import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from cartpole import ContinuousCartPoleEnv
from torch.optim import Adam
from matplotlib import pyplot as plt
from torch.distributions import Normal


def to_tensor(s, a, r, sp, mask, w, device):
    s = np.array(s)
    st = torch.FloatTensor(s).squeeze().to(device)
    at = torch.FloatTensor(a).reshape((-1,1)).to(device)
    rt = torch.FloatTensor(r).reshape((-1,1)).to(device)
    sp = np.array(sp)
    spt = torch.FloatTensor(sp).squeeze().to(device)
    maskt = torch.Tensor(mask).reshape((-1,1)).to(device)
    wt = torch.FloatTensor(w).reshape((-1,1)).to(device)
    return st, at, rt, spt, maskt, wt


class ReplayBuffer(object):
    def __init__(self, maxlen, dimState):
        self.maxlen = maxlen
        self.dimState = dimState
        self.alloc()

    def __len__(self):
        return self.filled

    def alloc(self):
        self.state_buff = np.zeros((self.maxlen, self.dimState), 
                                   dtype=np.float32)
        self.act_buff = np.zeros((self.maxlen, 1), dtype=np.float32)
        self.rew_buff = np.zeros(self.maxlen, dtype=np.float32)
        self.next_state_buff = np.zeros((self.maxlen, self.dimState),
                                        dtype=np.float32)
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
        indices = np.random.choice(N, batch_size, p=P)
        samples = [self.buffer[idx] for idx in indices]

        beta = self.beta_by_frame(self.frame)
        self.frame += 1

        # Compute importance-sampling weight
        weights = (N * P[indices]) ** (-beta)
        # normalize weights
        weights /= weights.max()
        weights = np.array(weights, dtype=np.float32)

        states, actions, rewards, next_states, dones = zip(*samples)
        return to_tensor(states, actions, rewards, next_states, dones, weights, device), indices
        # return np.concatenate(states), actions, rewards, np.concatenate(next_states), dones, indices, weights

    def update_priorities(self, batch_indices, batch_priorities):
        for idx, prio in zip(batch_indices, batch_priorities):
            self.priorities[idx] = abs(prio)

    def __len__(self):
        return len(self.buffer)


# Initialize Policy weights
def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)

class MLPCritic(nn.Module):
    def __init__(self, dimState, dimAction, hidden_dim):
        super(MLPCritic, self).__init__()
        self.fc1 = nn.Linear(dimState + dimAction, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x, x, x


class MLPTwinCritic(nn.Module):
    def __init__(self, dimState, dimAction, hidden_dim):
        super(MLPTwinCritic, self).__init__()
        self.fc11 = nn.Linear(dimState + dimAction, hidden_dim)
        self.fc12 = nn.Linear(hidden_dim, hidden_dim)
        self.fc13 = nn.Linear(hidden_dim, 1)

        self.fc21 = nn.Linear(dimState + dimAction, hidden_dim)
        self.fc22 = nn.Linear(hidden_dim, hidden_dim)
        self.fc23 = nn.Linear(hidden_dim, 1)

    def forward1(self, x):
        x = F.relu(self.fc11(x))
        x = F.relu(self.fc12(x))
        x = self.fc13(x)
        return x

    def forward2(self, x):
        x = F.relu(self.fc21(x))
        x = F.relu(self.fc22(x))
        x = self.fc23(x)
        return x

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        q1 = self.forward1(x)
        q2 = self.forward2(x)
        q12 = torch.cat([q1,q2], dim=1)
        qmin = torch.min(q12, dim=1)[0]
        qmin = qmin.reshape(q1.shape)
        return q1, q2, qmin


class MLPActor(nn.Module):
    def __init__(self, dimState, dimAction, hidden_dim, act_limit=1):
        super(MLPActor, self).__init__()
        self.fc1 = nn.Linear(dimState, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

        self.fc3 = nn.Linear(hidden_dim, dimAction)
        self.fc4 = nn.Linear(hidden_dim, dimAction)
        self.act_limit = act_limit
        self.apply(weights_init_)
        self.log2pi = np.log(2*np.pi)/2

    def forward(self, x):
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        mean = self.fc3(h)
        log_std = torch.clamp(self.fc4(h), min = -20, max = 2)
        return mean, log_std

    def sample(self, x, device):
        mean, log_std = self.forward(x)
        std = log_std.exp()
        e = torch.randn(1).to(device)
        xt = mean + std*e
        act = self.act_limit * torch.tanh(xt)
        return act, e

    def sample2(self, x, device):
        act, e = self.sample(x, device)
        log_1 = -e*e/2 - self.log2pi
        log_2 = torch.log(self.act_limit * (1 - act.pow(2)) + 1e-6)
        log_pi = (log_1 - log_2).sum(1, keepdim=True)
        return act, log_pi


class AgentSAC:
    def __init__(self, env, maxStep=1000):
        self.name = "SAC"
        # self.device = torch.device("cuda")
        # self.device = torch.device("cpu")
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Environment information
        self.env = env
        self.dimState = env.observation_space.shape[0]
        self.dimAction = env.action_space.shape[0]
        self.act_limit = env.action_space.high[0]
        self.gamma = 0.99
        self.maxStep = maxStep
        
        # Experience replay buffer
        self.PER = True
        if self.PER:
            self.memory = PrioritizedReplay(capacity=100000)
        else:
            self.memory = ReplayBuffer(100000, self.dimState)
        self.batch_size = 256

        # alpha initialization
        self.alpha = 1.0
        self.adjust_alpha = False
        if self.adjust_alpha:
            self.alpha_lr = 1e-3
            p = 0.05
            self.H0 = -(p*np.log(p) + (1-p)*np.log(1-p))
            self.alpha_list = []
            self.H_list = []

        # Network initialization
        hidden_dim = 256
        self.pnet = MLPActor(self.dimState, self.dimAction, hidden_dim, self.act_limit).to(self.device)
        # CriticNet = MLPCritic
        CriticNet = MLPTwinCritic
        self.qnet = CriticNet(self.dimState, self.dimAction, hidden_dim).to(self.device)
        self.qnet_target = CriticNet(self.dimState, self.dimAction, hidden_dim).to(self.device)
        self.policy_lr = 3e-4 # 3e-4
        self.policy_optimizer = Adam(self.pnet.parameters(), lr=self.policy_lr)
        self.value_lr = 3e-4 # 3e-4
        self.value_optimizer = Adam(self.qnet.parameters(), lr=self.value_lr)
        self.hard_update()

        self.tau = 1e-2 # 5e-3
        
    def save(self):
        torch.save(self.pnet.state_dict(), f"{self.name}_saved_policy.pt")
        torch.save(self.qnet.state_dict(), f"{self.name}_saved_value.pt")

    def load(self):
        self.pnet.load_state_dict(torch.load(f"{self.name}_saved_policy.pt"))
        self.qnet.load_state_dict(torch.load(f"{self.name}_saved_value.pt"))

    def hard_update(self):
        for target, source in zip(self.qnet_target.parameters(), self.qnet.parameters()):
            target.data.copy_(source.data)

    def soft_update(self):
        for target, source in zip(self.qnet_target.parameters(), self.qnet.parameters()):
            average = target.data * (1. - self.tau) + source.data * self.tau
            target.data.copy_(average)

    def getAction(self, state, test=False):
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        with torch.no_grad():
            action, _ = self.pnet.sample(state, self.device)
        return action.cpu().item()

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
            rewards.append(r.item())
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
        rewards = self.runEpisode(maxStep=maxStep, render=render,
                                  test=True)
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
            print(f"Train episode i={ep_i+1}, return = {ret:.1f}")
            retHist.append(ret)
            if ep_i < 4:
                continue
            k = 10
            avgk = sum(retHist[-k:])/k  # average return of the last 5 episodes
            if avgk > maxk and max(retHist[-k:]) == retHist[-1]:
                maxk = avgk
                print(f"iter {ep_i+1}, avgk = {avgk:.1f} updated")
                if avgk > 100:
                    self.save()
                if avgk >= self.maxStep * 2:
                    print(f"Average reached {self.maxStep} after {ep_i+1} episodes. Solution found")
                    break
        plt.plot(retHist)
        plt.show()

    def train(self):
        if len(self.memory) < self.batch_size:
            return
        self.policy_optimizer.zero_grad()
        self.value_optimizer.zero_grad()
        (s, a, r, sp, mask, weights), idx = self.memory.get_batch(self.batch_size, self.device)
        r = r.reshape((-1,1))
        mask = mask.reshape((-1,1))
        #######################################################################
        with torch.no_grad():
            ap, ap_log = self.pnet.sample2(sp, self.device)
            ap = ap.reshape((-1,1))
            _, _, qsp = self.qnet_target(sp, ap)
        y = r + (1 - mask) * self.gamma * (qsp - self.alpha * ap_log).detach()
        q1, q2, _ = self.qnet(s, a)
        loss1 = ((y - q1).pow(2)*weights).mean()
        loss2 = ((y - q2).pow(2)*weights).mean()
        value_loss = (loss1 + loss2)/2
        value_loss.backward()
        self.value_optimizer.step()

        #######################################################################
        b, log_pi = self.pnet.sample2(s, self.device)
        # with torch.no_grad():
        _, _, qb = self.qnet(s, b)
        policy_loss = ((self.alpha * log_pi - qb) * weights).mean()
        policy_loss.backward()
        self.policy_optimizer.step()

        #######################################################################
        if self.PER:  # update PrioritizedMemory
            with torch.no_grad():
                ap, ap_log = self.pnet.sample2(sp, self.device)
                _, _, qsp = self.qnet_target(sp, ap)
                q1, q2, _ = self.qnet(s, a)
            y = r + (1 - mask) * self.gamma * (qsp - self.alpha * ap_log)
            bellman_error = abs(y - (q1+q2)/2).squeeze().cpu().numpy()
            self.memory.update_priorities(idx, bellman_error)

        #######################################################################
        # entropy temperature tuning : bigger alpha --> bigger entropy --> more explore
        # if mean entropy > target entropy(H0), decrease alpha
        if self.adjust_alpha:
            log_pi = log_pi.detach().cpu()
            pi = log_pi.exp()
            mean_H = -((pi*log_pi).mean().item())
            grad_alpha = mean_H - self.H0
            self.alpha -= self.alpha_lr*grad_alpha
            self.alpha_list.append(self.alpha)
            self.H_list.append(mean_H)
        #######################################################################


def run(AgentClass=AgentSAC, mode='train'):
    env = ContinuousCartPoleEnv()
    agent = AgentClass(env)
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
    agentClass = AgentSAC

    mode = "train"
    # mode = "test"

    run(agentClass, mode)