import gym
import math
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as img

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from itertools import count
from torch.distributions import Categorical

def cartpole_step(self, action):
    x, x_dot, theta, theta_dot = self.state
    force = self.force_mag * (2*action - 1)
    costheta = math.cos(theta)
    sintheta = math.sin(theta)

    # For the interested reader:
    # https://coneural.org/florian/papers/05_cart_pole.pdf
    temp = (force + self.polemass_length * theta_dot ** 2 * sintheta) / self.total_mass
    thetaacc = (self.gravity * sintheta - costheta * temp) / (self.length * (4.0 / 3.0 - self.masspole * costheta ** 2 / self.total_mass))
    xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass

    if self.kinematics_integrator == 'euler':
        x = x + self.tau * x_dot
        x_dot = x_dot + self.tau * xacc
        theta = theta + self.tau * theta_dot
        theta_dot = theta_dot + self.tau * thetaacc
    else:  # semi-implicit euler
        x_dot = x_dot + self.tau * xacc
        x = x + self.tau * x_dot
        theta_dot = theta_dot + self.tau * thetaacc
        theta = theta + self.tau * theta_dot

    self.state = (x, x_dot, theta, theta_dot)

    done = bool(
        x < -self.x_threshold
        or x > self.x_threshold
        or theta < -self.theta_threshold_radians
        or theta > self.theta_threshold_radians
    )

    if not done:
        reward = 1.0
    elif self.steps_beyond_done is None:
        # Pole just fell!
        self.steps_beyond_done = 0
        reward = 1.0
    else:
        self.steps_beyond_done += 1
        reward = 0.0

    return np.array(self.state), reward, done, {}


def plotReturn(retHist, m=0):
    plt.plot(retHist)
    if m>1 and m<len(retHist):
        cumsum = [0]
        movAvg = []
        for i, x in enumerate(retHist, 1):
            cumsum.append(cumsum[i-1] + x)
            if i < m:
                i0 = 0
                n = i - i0
            else:
                i0 = i - m
                n = m
            ma = (cumsum[i] - cumsum[i0]) / n
            movAvg.append(ma)
        plt.plot(movAvg)
    plt.show()



class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.clear()

    def increase(self):
        self.S.append(None)
        self.A.append(None)
        self.R.append(None)
        self.SP.append(None)
        self.Mask.append(None)

    def push(self, s, a, r, sp, mask):
        if len(self.S) < self.capacity:
            self.increase()
        self.S[self.position] = s
        self.A[self.position] = a
        self.R[self.position] = r
        self.SP[self.position] = sp
        self.Mask[self.position] = mask
        self.position = (self.position + 1) % self.capacity

    @staticmethod
    def getSub(L, idx):
        return [L[i] for i in idx]

    def minibatch(self, batchSize, dimState, device):
        idx = random.sample(range(len(self.S)), batchSize)
        L = batchSize   # *batch : batch내의 data를 unpack
        L1 = [L,1]
        S = torch.cat(self.getSub(self.S,idx)).view([L, dimState]).to(device)
        A = torch.tensor(self.getSub(self.A,idx)).view(L1).to(device)
        R = torch.tensor(self.getSub(self.R,idx)).view(L1).to(device)
        SP = torch.cat(self.getSub(self.SP,idx)).view([L, dimState]).to(device)
        Mask = torch.tensor(self.getSub(self.Mask,idx)).view(L1).to(device)
        return L1, idx, S, A, R, SP, Mask


    def __len__(self):
        return len(self.S)

    def clear(self):
        self.position = 0
        self.S = []
        self.A = []
        self.R = []
        self.SP = []
        self.Mask = []

# SumTree
# a binary tree data structure where the parent’s value is the sum of its children
class SumTree:
    def __init__(self, capacity):
        self.capacity = capacity
        self.sum = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.nEntries = 0
        self.position = 0

    # update to the root node
    def propagate(self, idx, change):
        parent = (idx - 1) // 2
        self.sum[parent] += change
        if parent != 0:
            self.propagate(parent, change)

    # find sample on leaf node
    def retrieve(self, curr, s):
        left = 2 * curr + 1
        if left >= len(self.sum):  # curr is leaf
            return curr
        if s <= self.sum[left]:
            return self.retrieve(left, s)
        else:
            right = left + 1
            return self.retrieve(right, s - self.sum[left])

    def total(self):
        return self.sum[0]

    # append data with priority p
    def add(self, p, data):
        idx = self.position + self.capacity - 1
        self.data[self.position] = data
        self.update(idx, p)
        self.position += 1
        if self.position >= self.capacity:
            self.position = 0
        if self.nEntries < self.capacity:
            self.nEntries += 1

    # update priority
    def update(self, idx, p):
        change = p - self.sum[idx]
        self.sum[idx] = p
        self.propagate(idx, change)

    # get priority and sample
    def get(self, s):
        idx = self.retrieve(0, s)
        dataIdx = idx - self.capacity + 1
        return (idx, self.sum[idx], self.data[dataIdx])


class PerMemory:  # stored as ( s, a, r, s_ ) in SumTree
    def __init__(self, capacity):
        self.tree = SumTree(capacity)
        self.capacity = capacity
        self.e = 0.01
        self.a = 0.6
        self.beta = 0.4
        self.beta_increment_per_sampling = 0.001

    def getPriority(self, error):
        return (np.abs(error) + self.e) ** self.a

    def push(self, seg, error):
        p = self.getPriority(error)
        self.tree.add(p, seg)

    def sample(self, n):
        batch = []
        idxs = []
        segment = self.tree.total() / n
        priorities = []

        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])

        for i in range(n):
            a = segment * i
            b = segment * (i + 1)

            s = random.uniform(a, b)
            (idx, p, data) = self.tree.get(s)
            priorities.append(p)
            batch.append(data)
            idxs.append(idx)

        sampling_probabilities = priorities / self.tree.total()
        is_weight = np.power(self.tree.nEntries * sampling_probabilities, -self.beta)
        is_weight /= is_weight.max()

        return batch, idxs, is_weight

    def update(self, idx, error):
        p = self.getPriority(error)
        self.tree.update(idx, p)




class LinFA(torch.nn.Module):
    def __init__(self, D_in, D_out):
        super().__init__()
        self.fc = torch.nn.Linear(D_in, D_out)

    def forward(self, x):
        x = x.reshape(x.size(0), -1)
        y = self.fc(x)
        return y


class MLP(torch.nn.Module):
    def __init__(self, lsize):
        super().__init__()
        self.layers = nn.ModuleList()
        self.n_layers = len(lsize) - 1
        for i in range(self.n_layers):
            self.layers.append(torch.nn.Linear(lsize[i], lsize[i+1]))
            # self.layers.append(nn.BatchNorm2d(lsize[i]))

    def forward(self, x):
        x = x.reshape(x.size(0), -1)
        for i in range(self.n_layers):
            x = self.layers[i](x)
            if i < self.n_layers-1:
                # x = torch.tanh(x)
                x = F.relu(x)
        return x  # softmax is done within F.cross_entropy
        # return F.softmax(x, dim=-1)


class DuelNet(MLP):
    def __init__(self, lsize):
        lsize = lsize.copy()
        outSize = lsize.pop()
        super().__init__(lsize)
        H = lsize[-1]
        self.fcV1 = torch.nn.Linear(H, H)
        self.fcV2 = torch.nn.Linear(H, 1)
        self.fcA1 = torch.nn.Linear(H, H)
        self.fcA2 = torch.nn.Linear(H, outSize)

    def forward(self, x):
        h = super().forward(x)
        h = F.relu(h)
        hA = F.relu(self.fcA1(h))
        A = self.fcA2(hA)
        hV = F.relu(self.fcV1(h))
        V = self.fcV2(hV)
        Q = (V - A.mean(1, keepdim=True)) + A
        return Q

def angleReward(obs, r):
    # return np.float32(r)  # plain reward
    return np.float32(r + 10*np.abs(obs[2]))  # angle reward

class AgentBase:
    def __init__(self, env):
        self.env = env
        self.eps = 0.5
        nS = env.observation_space.shape[0]
        self.dimState = self.getStateDim()
        self.dimAction = env.action_space.n
        self.gamma = 1
        self.rb = ReplayMemory(10000)
        self.nEpisode = 0

    def getStateRep(self, obs):
        s = obs
        # quad = [s[0]*s[0], s[2]*s[2], s[0]*s[1], s[0]*s[2], s[1]*s[2]]
        # # quad = [s[0]*s[2]]
        # s = np.concatenate ((s, quad))
        return s

    def getStateDim(self):
        nS = self.env.observation_space.shape[0]
        s = self.getStateRep(np.zero(nS))
        return s.size

    def getQ(self, s):
        return self.Q[s]

    def piGreedy(self, state):
        q = self.getQ(state)
        a = q.argmax()
        return a

    def getAction(self, state):  # eps-greedy policy
        # eps-greedy
        if np.random.random() < self.eps:
            a = np.random.choice(self.dimAction)
            self.eps *= 0.99
            # print (f"exploration move, eps={self.eps}")
        else:
            a = self.piGreedy(state)
        return a

    @staticmethod
    def getSigmoid(z):
        exp_z = np.exp(z)
        return exp_z / (1+exp_z)

    @staticmethod
    def getSoftmax(z, tau=1):
        exp_z = np.exp((z - np.max(z)) / tau)
        return exp_z / exp_z.sum()

    def piSoftmax(self, state):
        q = self.getQ(state)
        p = self.getSoftmax(q).reshape(-1)
        a = np.random.choice(self.dimAction, p=p)
        return a


    def runEpisode1(self, saveRB=True, maxStep=200, render=False, preEpisodeCB=None, perStepCB=None, postEpisodeCB=None):
        obs = self.env.reset()
        s = self.getStateRep(obs)
        done = False
        self.nStep = 0
        ret = 0
        df = 1
        self.nEpisode += 1
        if preEpisodeCB:
            preEpisodeCB()
        while not done:
            if render:
                self.env.render()
            a = self.getAction(s)
            # a = self.piSoftmax(s)
            obs, r, done, info = self.env.step(a)
            r = angleReward(obs, r)
            ret += df*r
            df *= self.gamma
            sp = self.getStateRep(obs)
            if saveRB:
                self.rb.push(s, a, r, sp, not done)
            if perStepCB:
                perStepCB(s, a, r, sp, not done)
                # perStepCB(self.s, self.a, self.r, self.sp, self.done)
            s = sp
            self.nStep += 1
            if self.nStep >= maxStep:  # 500 for v1
                break
        if not done:
            if 'Q' in dir(self):
                ret += df*self.getQ(sp).max().item()
            else:
                ret += 100
        if postEpisodeCB:
            postEpisodeCB(done, self.nStep, ret)
        return ret


    def runTest(self, nEpisode=1, maxStep=1000, render=True):
        eps = self.eps
        self.eps = 0
        i = 0
        while i < nEpisode:
            i += 1
            ret = self.runEpisode1(saveRB=False, maxStep=maxStep, render=render)
            print(f"Test episode {self.nEpisode}, return = {ret:.1f} in {self.nStep} steps")
        self.eps = eps
        return ret


class AgentDQN(AgentBase):
    def __init__(self, env):
        super().__init__(env)
        self.prepareNN()
        self.tau = 0.05
        self.maxGrad = 0

    def prepareNN(self):
        # if torch.cuda.is_available():
        #     self.device = torch.device('cuda')
        # else:
        #     self.device = torch.device('cpu')
        self.device = torch.device('cpu')
        # self.qf = LinFA(self.dimState, self.dimAction)
        # self.qfTarget = LinFA(self.dimState, self.dimAction)

        H = 256
        lsize = [self.dimState, H, H, H, self.dimAction]
        # lsize = [self.dimState, self.dimAction]
        # ModelClass = MLP
        ModelClass = DuelNet
        self.qf = ModelClass(lsize).to(self.device)
        self.qfTarget = ModelClass(lsize).to(self.device)
        # self.hardUpdate()
        self.qfTarget.eval()

        # self.optimizer = optim.Adam(self.qf.parameters(), lr=0.001)
        self.optimizer = optim.Adam(self.qf.parameters(), lr=0.01, weight_decay=0.000001)
        # self.optimizer = optim.SGD(self.qf.parameters(), lr=0.01, weight_decay=0.000001)
        self.lrScheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=1, gamma=0.97)
        #
        # self.lossFunc = torch.nn.MSELoss(size_average=False, reduce=False)

    def load(self, fn):
        self.qf.load_state_dict(torch.load(fn, map_location=self.device))
        self.hardUpdate()

    def save(self, fn):
        torch.save(self.qf.state_dict(), fn)

    def hardUpdate(self):
        self.qfTarget.load_state_dict(self.qf.state_dict())

    def softUpdate(self):
        for pParam, cParam in zip(self.qfTarget.parameters(), self.qf.parameters()):
            pParam.data.copy_(self.tau*cParam.data + (1.0-self.tau)*pParam.data)

    def getStateRep(self, s):
        # quad = [s[0]*s[0], s[2]*s[2], s[0]*s[1], s[0]*s[2], s[1]*s[2]]
        # s = np.concatenate((s, quad))
        sTensor = torch.tensor(s, dtype=torch.float32).view(1,len(s))
        return sTensor

    def getStateDim(self):
        nS = self.env.observation_space.shape[0]
        s = self.getStateRep(np.zeros(nS))
        return s.size(1)

    def getQ(self, stateTensor):
        s = stateTensor.to(self.device)
        with torch.no_grad():
            q = self.qf(s)
        q = q.cpu().numpy()
        return q

    def trainBatch(self, batchSize):
        L1, idx, S, A, R, SP, notDone = self.rb.minibatch(batchSize, self.dimState, self.device)
        with torch.no_grad():
            qpTarget = self.qfTarget(SP)
            qpTarget = qpTarget * notDone  # mask qp to zero for final states
            # (maxqp,idx) = qpTarget.max(dim=-1)    # plain vanilla DQN
            # maxqp = maxqp.view(L1)                # plain vanilla DQN

            qp = self.qf(SP) * notDone              # double DQN
            (_,AP) = qp.max(dim=-1)                 # double DQN
            qpTarget = qpTarget.gather(1, AP.unsqueeze(-1))       # double DQN
            maxqp = qpTarget.view(L1)               # double DQN

            T = R + self.gamma * maxqp

        Q = self.qf(S).gather(1,A).view(L1)
        loss = F.mse_loss(Q, T)

        # loss clipping
        # loss = self.lossFunc(Q, T)
        # loss = torch.clamp(loss, min=-5, max=5)
        # loss = loss.mean()

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # for param in self.qf.parameters():
        #     gradNorm = param.grad.data.max().item()
        #     gradNorm = abs(gradNorm)
        #     if abs(gradNorm) > self.maxGrad:
        #         self.maxGrad = gradNorm
        #         print(f'maxGrad = {self.maxGrad}')
        #     # param.grad.data.clamp_(-0.1, 0.1)
        # grad clipping
        # torch.nn.utils.clip_grad_norm_(self.qf.parameters(), 5)

        self.optimizer.step()
        return loss.item()


    def trainFromRB(self, tol=0.05):
        L = len(self.rb)
        # BATCH_SIZE = 1024
        # if L < BATCH_SIZE:
        #     return 0.0, 0
        BATCH_SIZE = min(L, 1024)
        N = L // BATCH_SIZE
        minN = N // 10
        maxN = min(N*20, 1)
        lossAvg = tol*10
        lossSum = 0.0
        n = 0
        while n<maxN:
            lossSum += self.trainBatch(BATCH_SIZE)
            n += 1
            lossAvg = lossSum / n
            if n>minN and lossAvg<tol:
                break
        return lossAvg, n

    def DQNstepCB(self, s, a, r, sp, notDone):
        loss, niter = self.trainFromRB()
        self.tau = 0.001
        self.softUpdate()

    def DQNpostEpisodeCB(self, done, nStep, ret):
        i = self.nEpisode
        # if i % 1 == 0:
        #     loss, niter = self.trainFromRB()
        #     print (f"i={i}, n_iteration = {niter}, loss = {loss:.2f}, return = {ret:.1f}")
        #     self.softUpdate()
        #     pass
        # if i % 10 == 0:
        #     # self.hardUpdate()
        #     # self.lrScheduler.step()
        #     # print (f"param copied.  LR = {self.optimizer.param_groups[0]['lr']}, eps={self.eps}")
        #     pass
        # if i > 5:
        #     avg5 = sum(retHist[-5:]) / 5
        #     if avg5 > max5 and max(retHist[-5:]) == retHist[-1]:
        #         max5 = avg5
        #         print(f"iter {i}, avg5 = {avg5} updated")
        #         if avg5 > 100:
        #             self.save(f"saved/Cartpole-v1.best-{avg5:.0f}")
        #         # self.updateTarget()
        #         print(f"param copied.  LR = {self.optimizer.param_groups[0]['lr']}, eps={self.eps}")
        #         if avg5 == maxStep:
        #             print(f"averge reached {maxStep} after {i} episodes. Solution found")
        #             break
        tret = self.runTest(render=False)
        self.testHist.append(tret)

    def runMany(self, nEpisode=1000, maxStep=200):
        self.testHist = []

        # self.load("Cartpole-100")
        # self.eps = 0
        i = 0
        while i < nEpisode:
            i += 1
            ret = self.runEpisode1(maxStep=maxStep, perStepCB=self.DQNstepCB, postEpisodeCB=self.DQNpostEpisodeCB)


def trainAgentDQN():
    env = gym.make('CartPole-v1').unwrapped
    # env = gym.make('CartPole-v1').unwrapped
    agent = AgentDQN(env)
    # agent.load("Cartpole-v1.best-200")
    # agent.eps = 0.01
    # agent.optimizer = optim.Adam(agent.qf.parameters(), lr=0.0001, weight_decay=0.000001)
    agent.runMany(500, maxStep=2000)
    # plotQ(agent, 0, 2)
    env.close()


def testBestAgentDQN():
    env = gym.make('CartPole-v1').unwrapped
    # env = gym.wrappers.Monitor(env, 'saved/Cartpole-v1.DQN.video', force=True)
    env.render()
    print ("Press enter to start")
    y = input()
    agent = AgentDQN(env)
    # agent.load("saved/Cartpole-v1.best-5003")
    agent.load("saved/Cartpole-v1.best-778")
    agent.runTest()
    env.close()

def plotAgentQ():
    env = gym.make('CartPole-v1').unwrapped
    agent = AgentDQN(env)
    agent.load("Cartpole-v1.best-5003")
    plotQ(agent, 0, 2)
    env.close()

class AgentLinPolicy(AgentBase):
    def getW(self, mu, sigma):
        sz = (1, self.dimState)
        z = np.random.normal(size=sz)
        self.w = mu + sigma * z

    def getAction(self, state):  # policy function
        z = np.dot(self.w, state)
        if z < 0:
            a = 0
        else:
            a = 1
        return a

    def runEpisodeN(self, saveRB=True, maxStep=200, nEpisodes=1):
        retSum = 0.0
        for i in range(nEpisodes):
            retSum += self.runEpisode1(saveRB, maxStep)
        self.ret = retSum / nEpisodes

    def __lt__(self, other):
        return self.ret < other.ret


class CEM:
    def __init__(self, sigma=1.0, nPop=100, nGen=10):
        self.env = gym.make('CartPole-v1').unwrapped
        self.dimState = self.env.observation_space.shape[0]
        sz = (1, self.dimState)

        self.mu = np.zeros(sz)
        self.sigma = np.full(sz, sigma)
        self.nPop = nPop
        self.elitePortion = 0.2
        self.nGen = nGen

    def initPop(self):
        self.agentSet = []
        for i in range(self.nPop):
            # env = gym.make('CartPole-v1').unwrapped
            agent = AgentLinPolicy(self.env)
            agent.getW(self.mu, self.sigma)
            self.agentSet.append(agent)

    def sampleAgents(self):
        for a in self.agentSet:
            a.getW(self.mu, self.sigma)

    def runAgents(self, maxStep=200, nEpisodes=1):
        for a in self.agentSet:
            a.runEpisodeN(saveRB=False, maxStep=maxStep, nEpisodes=nEpisodes)

    def computeWeight(self, nElite):  # return proportional weight
        p = np.full(nElite, 1/nElite)
        return p

    def updatePopParam(self):
        self.agentSet.sort(reverse=True)
        nElite = math.floor(len(self.agentSet) * self.elitePortion)
        # print (f"{self.elitePortion*100:.0f}% return = {self.pop[nElite].ret}")
        p = self.computeWeight(nElite)

        w = self.agentSet[0].w
        muNew = p[0] * w
        sigmaNew = p[0] * (w - self.mu)**2
        sumRet = self.agentSet[0].ret
        for i in range(1, nElite):
            w = self.agentSet[i].w
            muNew += p[i] * w
            sigmaNew += p[i] * (w - self.mu) ** 2
            sumRet += self.agentSet[i].ret

        self.mu = muNew
        self.sigma = np.sqrt(sigmaNew)
        return sumRet / nElite, self.agentSet[0].ret, self.agentSet[nElite].ret

    def bestw(self):
        return self.agentSet[0].w

    def runMany(self, maxStep=500):
        retHist = []
        self.initPop()
        for i in range(self.nGen):
            self.sampleAgents()
            self.runAgents(maxStep=maxStep)
            rMean, rMax, rMin = self.updatePopParam()
            print (f"iteration {i+1}, 20% max, mean, min = ({rMax:.2f}, {rMean:.2f}, {rMin:.2f})")
            print (f"mu    = {self.mu}")
            print (f"sigma = {self.sigma}")
            retHist.append(rMean)
        plt.plot(retHist)
        plt.show()

    def runTest(self, w):
        agent = AgentLinPolicy(self.env)
        # agent.w = self.mu
        agent.w = w
        return agent.runTest(maxStep=4000)

class CEM2(CEM):  # Weighted averga
    def computeWeight(self, nElite):  # return proportional weight
        W = np.zeros(nElite)
        sum = 0
        for i in range(nElite):
#             v = self.agentSet[i].ret  # return proportional weight
            v = math.log((1+nElite)/(1+i)) # rank based weight
            W[i] = v
            sum += W[i]
        W /= sum
        return W

def trainCEM(train=True):
    pop = CEM(1.0, 100, 20)
    if train:
        pop.runMany(1000)
        np.save(f"saved/CEM-mu", pop.mu)
        print ("mu = ", pop.mu)
        np.save(f"saved/CEM-best", pop.bestw())
        print ("best w = ", pop.bestw())
    w = np.load("saved/CEM-best.npy")
    print (f"w = {w}")
    # pop.env = gym.wrappers.Monitor(pop.env, 'Cartpole-v1.CEM', force=True)
    pop.runTest(w)
    pop.env.close()




class AgentReinforce(AgentBase):
    def __init__(self, env):
        self.env = env
        self.dimState = env.observation_space.shape[0]
        self.dimAction = env.action_space.n
        self.gamma = 1
        self.tau = 1

        #         lsize = [self.dimState, self.dimAction] # linear opolicy
        H = 256
        lsize = [self.dimState, H, self.dimAction]
        self.device = torch.device('cpu')
        self.actor = MLP(lsize).to(self.device)
        # self.optimizerA = optim.SGD(self.actor.parameters(), lr=0.01)
        self.optimizerA = optim.Adam(self.actor.parameters(), lr=0.001)
        # self.optimizerA = optim.Adam(self.actor.parameters(), lr=0.01, weight_decay=0.000001)
        self.lrSchedulerA = torch.optim.lr_scheduler.StepLR(self.optimizerA, step_size=10, gamma=0.9)

    def optimizeState(self):
        obs = self.env.reset()
        x = torch.FloatTensor(obs).view(1, self.dimState).to(self.device)
        x.requires_grad = True
        print(x.detach())
        inputOptimizer = optim.Adam([x], lr=0.01)
        for i in range(100):
            y = self.actor(x)
            # loss = -y[0]
            loss = -y.mean()
            inputOptimizer.zero_grad()
            loss.backward()
            inputOptimizer.step()
            print (f"loss = {loss.item()}, x = {x.detach()}")


    def load(self, fn):
        self.actor.load_state_dict(torch.load(fn))

    def save(self, fn):
        torch.save(self.actor.state_dict(), fn)

    def getAction(self, state):  # policy function
        q = self.actor(state) * self.tau
        p = F.softmax(q, dim=1)
        m = Categorical(p)
        a = m.sample()
        self.logProb = m.log_prob(a)
        a = a.item()
        self.sumProb += p[0, a].item()
        #         print (self.sumProb)
        return a

    def computeReturns(self, rewards, finalValue):
        ret = []
        N = len(rewards)
        R = finalValue
        for t in reversed(range(0, N)):
            R = rewards[t] + self.gamma * R
            # R *= self.gamma ** t
            ret.insert(0, R)
        return ret

    def trainEpisode1(self, maxStep=200, render=False):
        LogP = []
        R = []
        masks = []
        obs = self.env.reset()
        s = torch.FloatTensor(obs).view(1, self.dimState).to(self.device)
        self.sumProb = 0

        ret = 0
        retMax = 0
        nStep = 0
        done = False
        self.env.contAction = True
        while not done:
            nStep += 1
            if render:  self.env.render()
            a = self.getAction(s)  # action selection prob. is saved in self.logProb
            obs, r, done, info = self.env.step(a)
            r = angleReward(obs, r)
            ret += r
            sp = torch.FloatTensor(obs).view(1, self.dimState).to(self.device)

            LogP.append(self.logProb)
            R.append(torch.tensor([r], dtype=torch.float, device=self.device))
            s = sp
            if nStep >= maxStep:
                break

        if done:
            finalValue = 0
        else:
            finalValue = r / (1 - self.gamma)
        returns = self.computeReturns(R, finalValue)
        LogP = torch.cat(LogP)
        returns = torch.cat(returns)
        actorLoss = -(LogP * returns).mean()
        self.optimizerA.zero_grad()
        actorLoss.backward()
        self.optimizerA.step()
        if ret > retMax:  # new record
            retMax = ret
        self.lrSchedulerA.step()
        p = self.sumProb / nStep
        print(f"mean p = {p}, actor loss = {actorLoss.item()}, LR={self.optimizerA.param_groups[0]['lr']}")

        return ret

    def runMany(self, nEpisode=1000, maxStep=200):
        retHist = []

        i = 0
        max5 = 0
        while i < nEpisode:
            i += 1
            ret = self.trainEpisode1(maxStep=maxStep)
            retHist.append(ret)
            print(f"iteration = {i}, return = {ret}")

        plt.plot(retHist)
        plt.show()

    def runTest(self, maxStep=1000, render=True, tau=10):
        if tau is not None:
            tauSave = self.tau
            self.tau = tau

        obs = self.env.reset()
        done = False
        ret = 0
        nStep = 0
        while not done:
            if render:
                self.env.render()
            s = torch.FloatTensor(obs).view(1, self.dimState).to(self.device)
            a = self.getAction(s)
            obs, r, done, info = self.env.step(a)
            r = angleReward(obs, r)
            ret += r
            nStep += 1
            if nStep >= maxStep:  # 500 for v1
                break

        print(f"Test episode, return = {ret} in {nStep} steps")

        if tau is not None:
            self.tau = tauSave


def trainAgentReinforce():
    env = gym.make('CartPole-v1').unwrapped
    agentR = AgentReinforce(env)
    agentR.runMany(nEpisode=50, maxStep=500)
    agentR.runTest()
    env.close()
    agentR.optimizeState()


class AgentAC(AgentReinforce):
    def __init__(self, env):
        super().__init__(env)
        self.prepareCritic()
        self.retMax = -9e49
        self.sumProb = 0

    def prepareCritic(self):
        H = 256
        lsize = [self.dimState, H, 1]
        self.critic = MLP(lsize).to(self.device)
        self.optimizerC = optim.Adam(self.critic.parameters(), lr=0.05)
        self.lrSchedulerC = torch.optim.lr_scheduler.StepLR(self.optimizerC, step_size=10, gamma=0.9)
        self.optimizerA = optim.Adam(self.actor.parameters(), lr=0.0001)
        # self.optimizerA = optim.Adam(self.actor.parameters(), lr=0.001, weight_decay=0.000001)
        self.lrSchedulerA = torch.optim.lr_scheduler.StepLR(self.optimizerA, step_size=10, gamma=0.9)

    def load(self, fn):
        super().load(fn + 'A')
        self.critic.load_state_dict(torch.load(fn + 'C'))

    def save(self, fn):
        super().save(fn + 'A')
        torch.save(self.critic.state_dict(), fn + 'C')

    def updateAC(self, s, logP, r, sp, done):
        Vs = self.critic(s)
        with torch.no_grad():
            Vsp = self.critic(sp)
        target = r + ((1 - done) * self.gamma) * Vsp
        advantage = target - Vs
        actorLoss = -(logP * advantage.detach())  # backprop toward LogP only

        self.optimizerA.zero_grad()
        actorLoss.backward()
        self.optimizerA.step()
        # self.lrSchedulerA.step()
        # print (f"actor loss = {actorLoss.item()}, LR={self.optimizerA.param_groups[0]['lr']}")

        criticLoss = (advantage * advantage)  # backprop toward advantages
        self.optimizerC.zero_grad()
        criticLoss.backward()
        self.optimizerC.step()
        # self.lrSchedulerC.step()
        # print (f"target={target.item()}, Vs={Vs.item()}, pi(a)={np.exp(logP.item()):.2f}, actor loss = {actorLoss.item()}, critic loss = {criticLoss.item()}")

    def trainEpisode1(self, maxStep=200, render=False):
        obs = self.env.reset()
        s = torch.FloatTensor(obs).view(1, self.dimState).to(self.device)
        self.sumProb = 0

        ret = 0
        nStep = 0
        done = False
        while not done:
            nStep += 1
            if render:  self.env.render()
            a = self.getAction(s)
            obs, r, done, info = self.env.step(a)
            r = angleReward(obs, r)
            ret += r
            sp = torch.FloatTensor(obs).view(1, self.dimState).to(self.device)
            self.updateAC(s, self.logProb, r, sp, done)
            s = sp
            if nStep >= maxStep:  # 500 for v1
                break

        if ret > self.retMax:
            self.save("saved/Cartpole-AC")
            self.retMax = ret
        print(f'mean prob = {self.sumProb / nStep}')
        return ret


def trainAgentAC():
    env = gym.make('CartPole-v1').unwrapped
    agent = AgentAC(env)
    agent.runMany(nEpisode=500, maxStep=500)
    env.close()


class AgentPPO(AgentAC):
    def __init__(self, env):
        super().__init__(env)
        self.rb = ReplayMemory(20000)

    def trainBatch(self, batchSize):

        L1, idx, S, A, R, SP, notDone = self.rb.minibatch(batchSize, self.dimState, self.device)
        with torch.no_grad():
            # qp = self.qf(SP)
            qp = self.qfTarget(SP)
            qp = qp * notDone  # mask qp to zero for final states
            (maxqp,idx) = qp.max(dim=-1)
            T = R + self.gamma * maxqp.view(L1)

        Q = self.qf(S).gather(1,A).view(L1)
        loss = F.mse_loss(Q, T)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # for param in self.qf.parameters():
        #     param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
        return loss.item()


    def trainFromRB(self):
        loss = 0.0
        n = 0
        BATCH_SIZE = 128
        if len(self.rb) < BATCH_SIZE:
            return loss
        N = 10 * len(self.rb) // BATCH_SIZE
        N = min(N, 128)
        for i in range(N):
            loss += self.trainBatch(BATCH_SIZE)
            n += 1
        return loss/n

def getArange(r, n):
    d = (r[1]-r[0])/n
    x = np.arange(r[0], r[1]+0.01, d)
    return x


def getAllStateTensor(agent, x0, i1, r1, i2, r2):
    x1 = getArange(r1, 20)
    n1 = len(x1)
    x2 = getArange(r2, 20)
    n2 = len(x2)
    S = np.zeros((n1*n2, agent.dimState))
    k = 0
    for i in range(n1):
        for j in range(n2):
            S[k] = x0
            S[k][i1] = x1[i]
            S[k][i2] = x2[j]
            k += 1

    ST = torch.tensor(S, dtype=torch.float32)
    return ST, x1, x2


def plotQ(agent, i1, i2):
    x0 = np.zeros(agent.dimState)
    low = agent.env.observation_space.low
    high = agent.env.observation_space.high
    ii1 = i1 - i1%2
    r1 = (low[ii1], high[ii1])
    ii2 = i2 - i2%2
    r2 = (low[ii2], high[ii2])
    S, x1, x2 = getAllStateTensor(agent, x0, i1, r1, i2, r2)
    S = S.to(agent.device)
    with torch.no_grad():
        Q = agent.qf(S)
    Q = Q.cpu().numpy()
    Qdiff = Q[:,0] - Q[:,1]
    N = len(x1)
    M = len(x2)
    imQ0 = Q[:,0].reshape((N,M))
    imQ1 = Q[:,1].reshape((N,M))
    imQdiff = Qdiff.reshape((N,M))

    vmin = -100
    vmax = 100
    fig = plt.figure()
    ax = fig.add_subplot(1, 3, 1)
    ax.imshow(imQ0, vmin=vmin, vmax=vmax)
    ax = fig.add_subplot(1, 3, 2)
    ax.imshow(imQ1, vmin=vmin, vmax=vmax)
    ax = fig.add_subplot(1, 3, 3)
    ax.imshow(imQdiff, vmin=vmin/5, vmax=vmax/5)

    plt.show()
    # plt.draw()
    # plt.pause(1)




if __name__ == "__main__":
    # trainCEM()
    # trainCEM(train=False)
    # trainAgentReinforce()
    # trainAgentAC()

    # np.random.seed(300)
    trainAgentDQN()
    # testBestAgentDQN()
    # plotAgentQ()

