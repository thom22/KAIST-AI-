import numpy as np
from matplotlib import pyplot as plt


class AgentOnGW:
    def __init__(self, env):
        self.env = env
        self.nA = env.get_action_space()
        self.nr, self.nc = env.get_state_space()
        self.nS = self.nr * self.nc
        self.gamma = env.gamma   # discount factor
        self.reset_all()

    def use_double_Q(self):
        return 'td_method' in dir(self) and self.td_method=='DoubleQ'

    # self.V : value for each cell
    def reset_V(self, v=0.0):
        # v = 100
        if 'V' not in dir(self):  # locals()
            self.V = np.zeros(self.nS, float)
        self.V.fill(v)

    # self.Q : value for each state-action pair (s, a)
    def reset_Q(self, v=0.0):
        # v = 100
        if 'Q' not in dir(self):  # locals()
            nSA = self.get_nSA()
            self.Q = np.zeros(nSA, float)
        self.Q.fill(v)
        if self.use_double_Q():
            if 'QB' not in dir(self):  # locals()
                nSA = self.get_nSA()
                self.QB = np.zeros(nSA, float)
            self.QB.fill(v)

    # self.pi : policy for each state
    def reset_policy(self):
        if 'pi' not in dir(self):
            self.pi = np.zeros(self.nS, int)  # policy function
        else:
            self.pi.fill(0)

    def reset_all(self):
        self.reset_V()
        self.reset_Q()
        self.reset_policy()
        if 'optimalQ' in dir(self):
            del(self.optimalQ)

    def get_nSA(self):
        return (self.nS, self.nA)

    def rc_to_index(self, r, c):
        return r*self.nc + c

    def index_to_rc(self, idx):  # idx --> (r, c)
        return idx // self.nc, idx % self.nc

    def get_Q_2D(self):
        # reshape does not make a new copy of Q
        return self.Q.reshape((self.nr, self.nc, self.nA))

    def get_V_2D(self):
        # reshape does not make a new copy of Q
        return self.V.reshape((self.nr, self.nc))

    def get_V_from_Q(self):
        if self.use_double_Q():
            self.V = np.max((self.Q+self.QB)/2, axis=1)  # compute V from Q for display purpose
        else:
            self.V = np.max(self.Q, axis=1)  # compute V from Q for display purpose

    def greedy_gap(self):
        greedy_actions = np.argmax(self.Q, axis=1)
        self.curr_pi = greedy_actions
        self.curr_path = [[], []]
        step = 0
        v_sr, v_sc = 0, 0
        v_s = self.env.rc_to_index(v_sr, v_sc)
        while not self.env.is_terminal(v_sr, v_sc) and step < (self.env.nr+self.env.nc):
            action = self.curr_pi[v_s]
            self.curr_path[0].append(v_s)
            self.curr_path[1].append(action)
            v_sr, v_sc, _ = self.env.next_state(v_sr, v_sc, action, False)
            v_s = self.env.rc_to_index(v_sr, v_sc)
            step += 1

        curr_Q = self.Q[tuple(self.curr_path)]
        opt_Q = self.optimalQ[tuple(self.curr_path)]
        diff = (curr_Q - opt_Q)
        L2 = np.sqrt(np.mean(diff**2))
        return L2


    def optimality_gap(self):
        if 'Q' not in dir(self):
            return 0
        if 'optimalQ' not in dir(self):
            sr, sc = self.env.sr, self.env.sc  # save state of self.env
            dpa = DPAgentOnGW(self.env)
            dpa.gamma = self.gamma
            dpa.value_iteration(max_iter=100)
            self.optimalQ = dpa.Q
            self.optimalV = dpa.V
            self.env.sr, self.env.sc = sr, sc  # recover state of self.env

        # L2 = self.greedy_gap()
        diff = (self.V - self.optimalV)
        L2 = np.sqrt(np.mean(diff**2))
        return L2

    def getStatusMsg(self):
        return f"Opt. gap = {self.optimality_gap():.2f} "


# Agent with Dynamic Programming capability
class DPAgentOnGW(AgentOnGW):
    def __init__(self, env):
        super().__init__(env)
        self.T, self.R = self.get_model()

    def get_model(self):
        T = [None] * self.nA
        R = np.zeros((self.nS, self.nA))

        for action in range(self.nA):
            Ta = np.zeros((self.nS, self.nS))
            for sr in range(self.nr):
                for sc in range(self.nc):
                    s = self.rc_to_index(sr, sc)
                    self.env.move_to(sr, sc)
                    transition = self.env.get_transition(sr, sc, action)
                    for (tr, tc) in transition.keys():
                        tran = transition[(tr, tc)]
                        s_next = self.rc_to_index(tr, tc)
                        Ta[s, s_next] = tran[0]
                        R[s, action] += tran[1]

            T[action] = Ta

        return T, R

    def pe_step(self):
        max_dev = 0.0
        newQ = np.zeros((self.nS, self.nA))
        for s in range(self.nS):
            for a in range(self.nA):
                value = 0.
                for next_s in range(self.nS):
                    value += self.T[a][s, next_s] * self.Q[next_s, self.pi[next_s]]
                newQ[s, a] = self.R[s, a] + self.gamma * value
                dev = np.abs(newQ[s, a] - self.Q[s, a])
                if dev > max_dev:
                    max_dev = dev
        self.Q = newQ  # sync. backup
        self.get_V_from_Q()  # compute V from Q for display purpose
        return max_dev

    def greedy_policy_improve(self):
        greedy_actions = np.argmax(self.Q, axis=1)
        n_changed = (greedy_actions != self.pi).sum()
        self.pi = greedy_actions
        return n_changed > 0

    def policy_evaluation(self, max_iter_eval=100, eps=1e-4):
        for k in range(max_iter_eval):
            dev = self.pe_step()
            if dev < eps:
                return

    def policy_iteration(self, max_iter, max_iter_eval=100):
        self.greedy_policy_improve()   # initial policy
        for k in range(max_iter):
            self.policy_evaluation(max_iter_eval)
            policy_changed = self.greedy_policy_improve()
            if not policy_changed:
                break

    def vi_step(self):   # async backup
        max_dev = 0.0
        for a in range(self.nA):
            for s in range(self.nS):
                maxQ = np.max(self.Q, axis=1)
                newQ = self.R[s, a] + self.gamma*np.dot(self.T[a][s, :], maxQ)
                dev = np.abs(newQ - self.Q[s, a])
                if dev > max_dev:
                    max_dev = dev
                self.Q[s, a] = newQ
        self.get_V_from_Q()   # compute V from Q for display purpose
        return max_dev

    def value_iteration(self, max_iter=10000, eps=1e-4):
        self.greedy_policy_improve()  # initial policy
        for k in range(max_iter):
            dev = self.vi_step()
            if dev < eps:
                break


# Agent with Monte-Carlo control capability
class MCControlOnGW(AgentOnGW):
    def __init__(self, env):
        super().__init__(env)
        self.eps = 0.5
        self.reset_episode()

    def get_state(self):
        return self.env.observe()

    def reset_episode(self):
        self.env.reset()
        # self.env.reset(0.7)  # 70% random start, 30% start at (0,0)
        if 'n_episode' not in dir(self):
            self.n_episode = 0
        self.n_episode += 1

    # self.NV : visit count for each state (grid)
    # self.NQ : visit count for each state-action pair
    def reset_NV(self):
        if 'NV' not in dir(self): #locals()
            self.NV = np.zeros(self.nS, int)
        else:
            self.NV.fill(0)
        if 'NQ' not in dir(self): #locals()
            self.NQ = np.zeros(self.get_nSA(), int)
        else:
            self.NQ.fill(0)

    def reset_all(self):
        super().reset_all()
        self.reset_NV()

    def set_alpha(self, const_alpha=True, p_or_v=0.5):
        self.const_alpha = const_alpha
        # const_alpha = True -> learning rate: self.alpha
        # const_alpha = False  -> learning rate: (1/N)**p
        self.alpha = p_or_v
        self.p = p_or_v

    '''
    Task 1: Epsilon-greedy policy function

    Get action from epsilon-greedy policy
    '''
    def eps_greedy_policy(self, eps):
        p = np.random.random()
        ##########################################
        # Task 1: fill your code here
        # Compute the epsilon-greedy actions for the current state from self.Q
        # and return it
        # Solution
        if p < eps:
            return np.random.randint(0, self.nA)
        else:
            s = self.get_state()
            return self.Q[s].argmax()
        ##########################################

    # policy function
    def get_action(self):
        s = self.get_state()
        # if self.n_episode <= 1 or self.NV[s] == 0:  # epsilon is state-dependent
        #     self.eps = 1.0
        # else:
        #     self.eps = 1 / (self.NV[s] ** 0.5)
        self.eps = 1 / (self.n_episode + 1)  # epsilon is dependent not on state but on the # of episode
        # self.eps = 0.1  # constant epsilon : good for Q learning
        return self.eps_greedy_policy(self.eps)

    # Simulate 1 episode through current policy & value function
    def get_episode(self, max_step=1000):
        self.reset_episode()
        S = []
        A = []
        R = []
        S.append(self.get_state())
        self.n_step = 0
        while self.n_step < max_step:
            a = self.get_action()
            A.append(a)
            ns, reward, done = self.env.step(a)
            R.append(reward)
            S.append(ns)
            self.n_step += 1
            if done: break
        return S, A, R, done

    # Calculate G(t) for each t
    def calc_return(self, R):
        n = len(R)
        G = np.zeros(n, float)
        g = 0.0
        for i in range(n-1, -1, -1):  # calc. return backward
            g = self.gamma*g + R[i]
            G[i] = g
        return G

    '''
    Task 2: MC prediction & control

    Update the value functions through MC algorithm
    '''
    def run_episode(self, max_step=1000):
        S, A, R, terminated = self.get_episode(max_step)
        G = self.calc_return(R)
        n = len(R)

        ##########################################
        # Task 2: fill your code here
        # Update self.V and self.Q through every visit MC algorithm
        # given 1 episode (S, A, R)
        # 
        # 1) Please use incremental arithmetic average
        #     : Learning rate alpha = 1/self.NQ[s][a]
        # 2) You also need to update self.NV and self.NQ
        #     : self.NV[s] is used for epsilon-greedy
        # Solution
        for t in range(n):
            St = S[t]
            At = A[t]
            self.NV[St] += 1
            self.NQ[St, At] += 1
            # alpha = 1 / self.NV[St]
            # self.V[St] += alpha * (G[t] - self.V[St])
            if self.const_alpha:
                alpha = self.alpha  # learning rate
            else:
                alpha = 1 / (self.NQ[St][At] ** self.p)
            # alpha = 1 / self.NQ[St][At]
            self.Q[St][At] += alpha * (G[t] - self.Q[St][At])
        ##########################################


    def run_simulation(self, n_episode, n_step):
        gaps = []
        for _ in range(n_episode):
            self.run_episode(max_step=n_step)
            self.get_V_from_Q()
            if n_episode >= 1000:
                gaps.append(self.optimality_gap())

        if n_episode >= 1000:
            plt.figure(figsize=(8, 6))
            plt.title('Learning plot of MC', fontsize=15)
            plt.plot(gaps)
            plt.xlabel('Episodes', fontsize=15)
            plt.ylabel('Opt. gap', fontsize=15)
            plt.show()

    def getStatusMsg(self):
        s1 = super().getStatusMsg()
        s1 += f"# of episodes = {self.n_episode-1}, n_step={self.n_step}"
        return s1


# Agent with Temporal-Difference control capability
class TDControlOnGW(MCControlOnGW):
    def __init__(self, env):
        super().__init__(env)
        self.alpha = 0.5
        self.p = 1.0
        self.td_method = 'Q-Learn'  # can be 'SARSA' or 'ExpSARSA'
        self.const_alpha = True

    def set_td_method(self, td_method='Q-Learn'):
        self.td_method = td_method
        if td_method == 'DoubleQ' and 'QB' not in dir(self):
            self.reset_all()


    def reset_all(self):
        super().reset_all()

    # policy function
    # def get_action(self):
    #     if self.td_method == 'Q-Learn':
    #         self.eps = 0.1
    #         return self.eps_greedy_policy(self.eps)
    #     else:
    #         return super().get_action()

    '''
    Task 3: TD prediction & control

    Update the value functions through TD algorithms
    '''
    def run_episode(self, max_step=1000):
        act = self.get_action()
        self.n_step = 0
        while self.n_step < max_step:
            self.n_step += 1
            s = self.get_state()
            self.NV[s] += 1
            self.NQ[s][act] += 1
            sp, reward, done = self.env.step(act)
            if self.env.is_terminal():  # terminal node
                self.Q[sp] = 0.0
            if self.const_alpha:
                alpha = self.alpha  # learning rate
            else:
                alpha = 1 / (self.NQ[s][act] ** self.p)  # learning rate
            #########################################
            # Task 3: fill your code here
            # 1) Update self.Q through SARSA & Q learning & Expected SARSA
            #    (you need to implement all the update methods
            #     by using if/elif/else statements conditioned on self.update)
            # 2) Assign next action to "act"
            # Solution
            if done:
                nextQ = 0.0
                ap = 0
            elif self.td_method == 'SARSA':      # SARSA
                ap = self.get_action()
                nextQ = self.Q[sp][ap]
            elif self.td_method == 'ExpSARSA':  # Expected SARSA for eps-greedy policy
                nextQ = self.Q[sp].max() * (1 - self.eps) + self.Q[sp].mean() * self.eps
            elif self.td_method == 'Q-Learn':    # Q learning
                nextQ = self.Q[sp].max()
            elif self.td_method == 'DoubleQ':  # double Q learning
                if  np.random.random() > 0.5:  # swap Q and QB w.p. 0.5
                    self.Q, self.QB = self.QB, self.Q
                aa = self.Q[sp].argmax()
                nextQ = self.QB[sp][aa]
                # aa = self.QB[sp].argmax()
                # nextQ = self.Q[sp][aa]

            target = reward + self.gamma * nextQ
            delta = target - self.Q[s][act]

            self.Q[s][act] += alpha * delta
            if self.td_method != 'SARSA':  # Q or Exp Sarsa
                ap = self.get_action()   # use updated Q

            act = ap
            #########################################
            if done:
                self.reset_episode()
                break

        # if not done:
        #     self.reset_episode()

    def run_simulation(self, n_episode, max_step=1000):
        gaps = []
        for i in range(n_episode):
            self.run_episode(max_step=max_step)
            self.get_V_from_Q()
            if n_episode >= 1000:
                gaps.append(self.optimality_gap())
        
        if n_episode >= 1000:
            plt.figure(figsize=(8, 6))
            title = f"Learning plot of {self.td_method}"
            plt.title(title, fontsize=15)
            plt.plot(gaps)
            plt.xlabel('Episodes', fontsize=15)
            plt.ylabel('Opt. gap', fontsize=15)
            plt.show()


class TDLambdaControlOnGW(TDControlOnGW):
    def __init__(self, env):
        super().__init__(env)
        self.lam = 0.5
        self.gam_lam = self.gamma * self.lam
        self.acc_trace = True

    def reset_E(self, new_size=1000):
        if 'E' not in dir(self): #locals()
            self.E = np.zeros(self.get_nSA(), float)
        else:
            self.NQ.fill(0)


    def reset_all(self):
        super().reset_all()
        self.reset_E()

    def run_episode(self, max_step=1000):
        act = self.get_action()
        self.n_step = 0
        while self.n_step < max_step:
            self.n_step += 1
            s = self.get_state()
            self.NV[s] += 1
            self.NQ[s][act] += 1

            sp, reward, done = self.env.step(act)
            if self.env.is_terminal():  # terminal node
                self.Q[sp] = 0.0
            if self.const_alpha:
                alpha = self.alpha  # learning rate
            else:
                alpha = 1 / (self.NQ[s][act] ** self.p)  # learning rate
            #########################################
            # Task 3: fill your code here
            # 1) Update self.Q through SARSA & Q learning & Expected SARSA
            #    (you need to implement all the update methods
            #     by using if/elif/else statements conditioned on self.update)
            # 2) Assign next action to "act"
            # Solution
            if done:
                nextQ = 0.0
                ap = 0
            elif self.td_method == 'SARSA':  # SARSA
                ap = self.get_action()
                nextQ = self.Q[sp][ap]
            elif self.td_method == 'ExpSARSA':  # Expected SARSA for eps-greedy policy
                nextQ = self.Q[sp].max() * (1 - self.eps) + self.Q[sp].mean() * self.eps
            elif self.td_method == 'Q-Learn':  # Q learning
                nextQ = self.Q[sp].max()
            elif self.td_method == 'DoubleQ':  # double Q learning
                if np.random.random() > 0.5:  # swap Q and QB w.p. 0.5
                    self.Q, self.QB = self.QB, self.Q
                aa = self.Q[sp].argmax()
                nextQ = self.QB[sp][aa]
                # aa = self.QB[sp].argmax()
                # nextQ = self.Q[sp][aa]

            target = reward + self.gamma * nextQ
            delta = target - self.Q[s][act]

            self.E[s][act] += 1  # acc trace
            # self.E[s][act] = 1  # replacing trace
            for ss in range(self.nS):
                for aa in range(self.nA):
                    if self.E[ss][aa] > 0.0:
                        self.Q[ss][aa] += alpha * self.E[ss][aa] * delta
                        self.E[ss][aa] *= self.gam_lam

            if self.td_method != 'SARSA':  # Q or Exp Sarsa
                ap = self.get_action()  # use updated Q

            act = ap

            #########################################
            if done:
                self.reset_episode()
                break

        # if not done:
        #     self.reset_episode()
