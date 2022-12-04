import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gym
from torch.optim import Adam
from matplotlib import pyplot as plt
from torch.distributions import Normal

def reward(obs, r):
    return np.float32(r + 10*np.abs(obs[2]))

def dot(t1, t2, dim=-1):
    return (t1 * t2).sum(dim=dim)

def to_tensor(s, a, r, sp, mask, w, device):
    st = torch.FloatTensor(s).squeeze().to(device)
    at = torch.LongTensor(a).to(device)
    rt = torch.FloatTensor(r).to(device)
    spt = torch.FloatTensor(sp).squeeze().to(device)
    maskt = torch.Tensor(mask).to(device)
    wt = torch.FloatTensor(w).to(device)
    return st, at, rt, spt, maskt, wt

class ReplayBuffer(object):
    def __init__(self, maxlen, dimState, contAction=False):
        self.maxlen = maxlen
        self.dimState = dimState
        self.contAction = contAction
        self.alloc()

    def __len__(self):
        return self.filled

    def alloc(self):
        self.state_buff = np.zeros((self.maxlen, self.dimState), dtype=np.float32)
        if self.contAction:
            self.act_buff = np.zeros(self.maxlen, dtype=np.float32)
        else:
            self.act_buff = np.zeros(self.maxlen, dtype=np.int32)
        self.rew_buff = np.zeros(self.maxlen, dtype=np.float32)
        self.next_state_buff = np.zeros((self.maxlen, self.dimState), dtype=np.float32)
        self.mask_buff = np.zeros(self.maxlen, dtype=np.uint8)
        self.clear()

    def clear(self):
        self.filled = 0
        self.position = 0

    def push(self, s, a, r, sp, mask):
        self.state_buff[self.position] = s
        self.act_buff[self.position] = a
        self.rew_buff[self.position] = r
        self.next_state_buff[self.position] = sp
        self.mask_buff[self.position] = mask
        self.position = (self.position + 1) % self.maxlen

        if self.filled < self.maxlen:
            self.filled += 1

    def get_batch(self, batch_size, device):
        idx = np.random.choice(np.arange(self.filled), size=batch_size, replace=True)
        weights = np.ones(batch_size)
        return to_tensor(self.state_buff[idx], self.act_buff[idx], self.rew_buff[idx],
                         self.next_state_buff[idx], self.mask_buff[idx], weights, device), idx


class PrioritizedReplay(object):
    """
    Proportional Prioritization
    """

    def __init__(self, capacity, alpha=0.6, beta_start=0.4, beta_frames=100000):
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.frame = 1  # for beta calculation
        self.capacity = capacity
        self.buffer = []
        self.pos = 0
        self.priorities = np.zeros((capacity,), dtype=np.float32)

    def beta_by_frame(self, frame_idx):
        """
        Linearly increases beta from beta_start to 1 over time from 1 to beta_frames.

        3.4 ANNEALING THE BIAS (Paper: PER)
        We therefore exploit the flexibility of annealing the amount of importance-sampling
        correction over time, by defining a schedule on the exponent
        that reaches 1 only at the end of learning. In practice, we linearly anneal from its initial value 0 to 1
        """
        return min(1.0, self.beta_start + frame_idx * (1.0 - self.beta_start) / self.beta_frames)

    def push(self, state, action, reward, next_state, done):
        assert state.ndim == next_state.ndim
        state = np.expand_dims(state, 0)
        next_state = np.expand_dims(next_state, 0)

        max_prio = self.priorities.max() if self.buffer else 1.0  # gives max priority if buffer is not empty else 1

        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, next_state, done))
        else:
            # puts the new data on the position of the oldes since it circles via pos variable
            # since if len(buffer) == capacity -> pos == 0 -> oldest memory (at least for the first round?)
            self.buffer[self.pos] = (state, action, reward, next_state, done)

        self.priorities[self.pos] = max_prio
        self.pos = (self.pos + 1) % self.capacity  # lets the pos circle in the ranges of capacity if pos+1 > cap --> new posi = 0

    def get_batch(self, batch_size, device):
        N = len(self.buffer)
        if N == self.capacity:
            prios = self.priorities
        else:
            prios = self.priorities[:self.pos]

        # calc P = p^a/sum(p^a)
        probs = prios ** self.alpha
        P = probs / probs.sum()

        # gets the indices depending on the probability p
        idx = np.random.choice(N, batch_size, p=P)

        # samples = [self.buffer[idx] for idx in indices]
        samples = [self.buffer[i] for i in idx]

        use_weight = False
        if use_weight:
            beta = self.beta_by_frame(self.frame)
            self.frame += 1

            # Compute importance-sampling weight
            weights = (N * P[idx]) ** (-beta)
            # normalize weights
            weights /= weights.max()
            weights = np.array(weights, dtype=np.float32)
        else:
            weights = np.ones(batch_size)

        states, actions, rewards, next_states, dones = zip(*samples)
        states = np.array(states)
        actions = np.array(actions)
        rewards = np.array(rewards)
        next_states = np.array(next_states)
        dones = np.array(dones)
        return to_tensor(states, actions, rewards, next_states, dones, weights, device), idx


    def update_priorities(self, batch_indices, batch_priorities):
        for idx, prio in zip(batch_indices, batch_priorities):
            self.priorities[idx] = abs(prio)

    def __len__(self):
        return len(self.buffer)




class MLPCritic(nn.Module):
    def __init__(self, dimState, dimAction, hidden_dim):
        super(MLPCritic, self).__init__()
        self.fc1 = nn.Linear(dimState, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, dimAction)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def forward2(self, x):
        logit = self.forward(x)
        log_p = F.log_softmax(logit, dim=-1)
        p = torch.exp(log_p)
        return logit, p, log_p

    def gumbel_softmax(self, x, tau=1, hard=True):
        y = self.forward(x)
        p = F.gumbel_softmax(y, tau=tau, hard=hard)
        return p


class AgentSoftmax:
    def __init__(self, env, maxStep=1000, ent_max=True):
        self.name = "Softmax"
        self.entropy_max = ent_max
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Environment information
        self.env = env
        self.dimState = env.observation_space.shape[0]
        self.dimAction = env.action_space.n
        self.gamma = 0.99
        self.maxStep = maxStep

        # Experience replay buffer
        self.PER = True
        if self.PER:
            self.memory = PrioritizedReplay(capacity=100000)
        else:
            self.memory = ReplayBuffer(100000, self.dimState)
        self.batch_size = 256

        # Network initialization
        hidden_dim = 256

        # Network initialization
        self.alpha = 0.2
        CriticNet = MLPCritic
        self.qnet = CriticNet(self.dimState, self.dimAction, hidden_dim).to(self.device)
        self.qnet_target = CriticNet(self.dimState, self.dimAction, hidden_dim).to(self.device)
        self.qnet.alpha = self.alpha
        self.qnet_target.alpha = self.alpha
        self.value_lr = 3e-3  # 3e-4
        self.value_optimizer = Adam(self.qnet.parameters(), lr=self.value_lr)

        self.hard_update()
        self.tau = 1e-2  # 5e-3

        self.actor = self.qnet

    def save(self):
        torch.save(self.qnet.state_dict(), f"{self.name}_saved_value.pt")

    def load(self):
        self.qnet.load_state_dict(torch.load(f"{self.name}_saved_value.pt"))

    def hard_update(self):
        for target, source in zip(self.qnet_target.parameters(), self.qnet.parameters()):
            target.data.copy_(source.data)

    def soft_update(self):
        for target, source in zip(self.qnet_target.parameters(), self.qnet.parameters()):
            average = target.data * (1. - self.tau) + source.data * self.tau
            target.data.copy_(average)

    def getAction(self, state, test=False):
        if not isinstance(state, torch.Tensor):
            state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        if test:
            with torch.no_grad():
                logit = self.actor.forward(state)
            action = logit.argmax(dim=-1).item()
        else:
            with torch.no_grad():
                _, p, _ = self.actor.forward2(state)
            m = torch.distributions.Categorical(probs=p)
            action = m.sample().item()
        return action

    # Simulate 1 episode using current pnet and return a list of rewards
    def runEpisode(self, maxStep, render=False, test=False):
        s = self.env.reset()
        done = False
        rewards = []
        for et_i in range(maxStep):
            if render:
                self.env.render()
                # if et_i == 0:  # for video capture
                #     input()
            a = self.getAction(s, test)
            sp, r, done, _ = self.env.step(a)
            r = reward(sp, r)
            rewards.append(r)
            if not test:
                self.memory.push(s, a, r, sp, done)
                self.train()
                self.soft_update()
            s = sp
            if done:
                break
        return rewards

    # Test the current pnet and return a sum of rewards
    def runTest(self, maxStep=1000, render=True):
        rewards = self.runEpisode(maxStep=maxStep, render=render, test=True)
        ret, nStep = sum(rewards), len(rewards)
        print(f"Test episode, return = {ret:.1f} in {nStep} steps")
        return ret

    # Run multiple episodes to train the agent and give a learning plot
    def runMany(self, nEpisode=500):
        retHist = []
        maxk = 0
        for ep_i in range(nEpisode):
            rewards = self.runEpisode(maxStep=self.maxStep)
            ret = sum(rewards)
            print(f"Train episode i={ep_i + 1}, return = {ret:.1f}")
            retHist.append(ret)
            if ep_i < 4:
                continue
            k = 15
            avgk = sum(retHist[-k:]) / k  # average retn of the last 5 episodes
            if avgk > maxk: # and max(retHist[-k:]) == retHist[-1]:
                maxk = avgk
                print(f"iter {ep_i+1}, avgk = {avgk:.1f} updated")
                if avgk > 100:
                    self.save()
                if avgk >= self.maxStep * 2.2:
                    print(f"Average/{k} = {avgk} after {ep_i+1} episodes. Solution found")
                    break

        plt.plot(retHist)
        plt.show()

    def train(self):
        if len(self.memory) < self.batch_size:
            return
        self.value_optimizer.zero_grad()
        (s, a, r, sp, mask, weights), idx = self.memory.get_batch(self.batch_size, self.device)
        #######################################################################
        with torch.no_grad():
            qsp, psp, _ = self.qnet_target.forward2(sp)
            if self.entropy_max:
                # qsp = qsp - self.alpha * torch.log(psp)
                qsp = (1-self.alpha)*qsp
            exp_qsp = dot(psp, qsp)   # expected SARSA target
        y = r + (1 - mask) * self.gamma * exp_qsp

        qs = self.qnet(s)
        qa = qs.gather(1, a.reshape(-1,1)).squeeze()

        # value_loss = F.mse_loss(qa, y, reduction='mean')
        value_loss = ((y - qa).pow(2)*weights).mean()
        value_loss.backward()
        self.value_optimizer.step()



class MLPActor(nn.Module):
    def __init__(self, dimState, dimAction, hidden_dim):
        super(MLPActor, self).__init__()
        self.fc1 = nn.Linear(dimState, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, dimAction)

    def forward(self, x):
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        logit = self.fc3(h)
        return logit

    def forward2(self, x):
        logit = self.forward(x)
        # p = F.softmax(logit, dim=-1)
        # log_p = torch.log(p)
        log_p = F.log_softmax(logit, dim=-1)
        p = torch.exp(log_p)
        return logit, p, log_p

# No entropy max term in value update
class AgentSoftAC (AgentSoftmax):
    def __init__(self, env, maxStep=1000, ent_max=True):
        super().__init__(env, maxStep, ent_max)
        self.name = "SoftAC"
        # Network initialization
        hidden_dim = 256

        # Network initialization
        self.pnet = MLPActor(self.dimState, self.dimAction, hidden_dim).to(self.device)
        self.policy_lr = 3e-4
        self.policy_optimizer = Adam(self.pnet.parameters(), lr=self.policy_lr)
        self.hard_update()
        self.actor = self.pnet

        self.adjust_alpha = True
        if self.adjust_alpha:
            self.alpha_lr = 1e-3
            p = 0.2
            self.H0 = -(p*np.log(p) + (1-p)*np.log(1-p))
            # H([0.5,0.5)=0.693] : max entropy
            # self.H0 = 0.67  # H([0.4,0.6)]=0.67,
            # self.H0 = 0.6  # H([0.3,0.7)] = 0.61,
            # self.H0 = 0.5  # H([0.2,0.8)],
            # self.H0 = 0.32  # H([0.1,0.9)],
            # self.H0 = 0.2    # H([0.05,0.95)],
            # self.H0 = 0.1  # H([0.02,0.98)]
            self.alpha_list = []
            self.H_list = []

    def save(self):
        super().save()
        torch.save(self.pnet.state_dict(), f"{self.name}_saved_policy.pt")

    def load(self):
        super().load()
        self.pnet.load_state_dict(torch.load(f"{self.name}_saved_policy.pt"))

    def train(self):
        if len(self.memory) < self.batch_size:
            return
        self.value_optimizer.zero_grad()
        self.policy_optimizer.zero_grad()

        (s, a, r, sp, mask, weights), idx = self.memory.get_batch(self.batch_size, self.device)

        #######################################################################
        # update critic
        with torch.no_grad():
            qsp = self.qnet_target(sp)
            _, psp, logpi_sp = self.pnet.forward2(sp)
            if self.entropy_max:
                qsp = qsp - self.alpha*logpi_sp
            exp_qsp = dot(psp, qsp)   # expected SARSA target
        y = r + (1 - mask) * self.gamma * exp_qsp

        qs = self.qnet(s)
        qa = qs.gather(1, a.reshape(-1,1)).squeeze()

        # value_loss = F.mse_loss(qa, y, reduction='mean')
        value_loss = ((y - qa).pow(2)*weights).mean()
        value_loss.backward()
        self.value_optimizer.step()

        #######################################################################
        # update actor (policy)
        qs = qs.detach()
        _, pi, log_pi = self.pnet.forward2(s)
        pi = torch.exp(log_pi)

        policy_loss = (dot(pi, self.alpha * log_pi - qs)*weights).mean()
        policy_loss.backward()
        self.policy_optimizer.step()

        #######################################################################
        # entropy temperature tuning : bigger alpha --> bigger entropy --> more explore
        # if mean entropy > target entropy(H0), decrease alpha
        if self.adjust_alpha:
            mean_H = -(dot(pi, log_pi).mean().detach().cpu().item())
            grad_alpha = mean_H - self.H0
            self.alpha -= self.alpha_lr*grad_alpha
            self.alpha_list.append(self.alpha)
            self.H_list.append(mean_H)
        #######################################################################

        if self.PER:  # update PrioritizedMemory
            with torch.no_grad():
                qsp = self.qnet_target(sp)
                _, psp, logpi_sp = self.pnet.forward2(sp)
                if self.entropy_max:
                    qsp = qsp - self.alpha * logpi_sp
                exp_qsp = dot(psp, qsp)  # expected SARSA target
                y = r + (1 - mask) * self.gamma * exp_qsp

                qs = self.qnet(s)
                qa = qs.gather(1, a.reshape(-1, 1)).squeeze()

            bellman_error = abs(y - qa).squeeze().cpu().numpy()
            self.memory.update_priorities(idx, bellman_error)



def run(agent, mode='train'):
    if mode == 'train':
        agent.runMany()
        if "alpha_list" in dir(agent):
            plt.plot(agent.alpha_list, label='alpha')
            plt.plot(agent.H_list, label='mean ent')
            plt.legend()
            plt.title(f"H0 = {agent.H0:.2f}")
            plt.show()
    elif mode == 'test':
        agent.load()
    agent.runTest()
    env.close()


if __name__ == '__main__':
    """
    1) Select the agent you want to train or test
    2) Select whether to train the agent
       or test the best agent you already trained
    """
    env = gym.make('CartPole-v1').unwrapped
    # agent = AgentSoftmax(env, maxStep=1000, ent_max=False)
    agent = AgentSoftAC(env, maxStep=1000, ent_max=True)

    mode = "train"
    # mode = "test"

    run(agent, mode)