import numpy as np


# Generic Grid World
class GridWorldEnv:
    def __init__(self):
        self.nr = 5
        self.nc = 5
        self.nA = 4   # 0:Left(West), 1:Right(East), 2:Up(North), 3:Down(South)
        self.gamma = 0.9
        self.episodic = True
        self.p_trans_rand = 0.2
        # self.p_trans_rand = 0
        self.reset()

    def rc_to_index(self, sr, sc):  # (sr, sc) --> index
        return sr*self.nc + sc

    def index_to_rc(self, idx):  # idx --> (sr, sc)
        sr = idx // self.nc
        sc = idx % self.nc
        return sr, sc

    # reset state
    def reset(self, rand_start=0.0, ir=0, ic=0):
        q = np.random.random()
        if q < rand_start:
            self.sr = np.random.randint(0, self.nr)
            self.sc = np.random.randint(0, self.nc)
        else:
            self.sr = ir
            self.sc = ic

    def observe(self):  # get current state
        return self.rc_to_index(self.sr, self.sc)

    def get_action_space(self):
        return self.nA

    def get_state_space(self):
        return (self.nr, self.nc)

    def get_nS(self):
        return self.nr * self.nc

    def get_state(self):
        return self.observe()

    def is_valid_rc(self, r, c):
        return (r >= 0 and r < self.nr and c >= 0 and c < self.nc)

    def is_valid_index(self, idx):
        return (idx >= 0 and idx < self.get_nS())

    def windy_action(self, action):
        return action       # no wind effect

    def stochastic_transition(self):
        a = np.array([-1, 0, 1])
        p = np.array([self.p_trans_rand/2,
                      1-self.p_trans_rand,
                      self.p_trans_rand/2])
        return np.random.choice(a, p=p)

    def next_state(self, cr, cc, action, stochastic=True):
        tr, tc = cr, cc
        if action == 0:  # Left, West
            tc -= 1
            if stochastic:
                tr += self.stochastic_transition()

        elif action == 1:  # Right, East
            tc += 1
            if stochastic:
                tr += self.stochastic_transition()

        elif action == 2:  # Up, North
            tr -= 1
            if stochastic:
                tc += self.stochastic_transition()

        else:  # if action == 3:  # Down, South
            tr += 1
            if stochastic:
                tc += self.stochastic_transition()

        if self.is_valid_rc(tr, tc):
            return tr, tc, True
        else:
            return cr, cc, False

    def get_transition(self, cr, cc, action):
        transition = dict()
        if self.is_terminal(cr, cc):
            transition[(cr, cc)] = [1., 0.]
            return transition
        tr, tc, moved = self.next_state(cr, cc, action, False)
        p = 1 - self.p_trans_rand
        done = self.is_terminal(tr, tc)
        r = p*self.get_reward(tr, tc, done)
        transition[(tr, tc)] = [p, r]
        for i in range(2):
            if moved:
                if action < 2:
                    cand_tc = tc
                    cand_tr = tr + (-1) ** i
                else:
                    cand_tc = tc + (-1) ** i
                    cand_tr = tr
            else:
                if action == 0:
                    cand_tc = tc - 1
                    cand_tr = tr + (-1) ** i
                elif action == 1:
                    cand_tc = tc + 1
                    cand_tr = tr + (-1) ** i
                elif action == 2:
                    cand_tc = tc + (-1) ** i
                    cand_tr = tr - 1
                else:
                    cand_tc = tc + (-1) ** i
                    cand_tr = tr + 1
            if self.is_valid_rc(cand_tr, cand_tc):
                sto_tr = cand_tr
                sto_tc = cand_tc
            else:
                sto_tr = cr
                sto_tc = cc
            p = self.p_trans_rand / 2
            done = self.is_terminal(sto_tr, sto_tc)
            r = p*self.get_reward(sto_tr, sto_tc, done)
            try:
                transition[(sto_tr, sto_tc)][0] += p
                transition[(sto_tr, sto_tc)][1] += r
            except:
                transition[(sto_tr, sto_tc)] = [p, r]
        return transition

    def move_to(self, tr, tc):
        self.sr = tr
        self.sc = tc

    def step(self, action):
        raise NotImplementedError

    def is_terminal(self, *args, **kwargs):
        raise NotImplementedError

    def get_reward(self, *args, **kwargs):
        pass


class GWE_Trap(GridWorldEnv):  # One goal, one trap
    def __init__(self, goal_reward=0, trap_penalty=-10, step_penalty=-1):
        super().__init__()
        self.nr = 5
        self.nc = 5
        self.sr = 0
        self.sc = 0
        self.er = 4
        self.ec = 4
        self.trap_r = self.nr // 2
        self.trap_c = self.nc // 2
        self.goal_reward = goal_reward
        self.trap_penalty = trap_penalty
        self.step_penalty = step_penalty
        self.gamma = 0.9
        self.p_trans_rand = 0.2
        # self.continuing = True

    def is_terminal(self, sr=None, sc=None):
        if sr is None and sc is None:
            return (self.sc == self.ec and self.sr == self.er)
        return (sc == self.ec and sr == self.er)

    def is_trap(self, sr, sc):
        return (sc == self.trap_c and sr == self.trap_r)

    def step(self, action):
        if self.is_terminal(self.sr, self.sc):
            return self.rc_to_index(self.sr, self.sc), 0, True
        tr, tc, _ = self.next_state(self.sr, self.sc, action)

        done = self.is_terminal(tr, tc)
        # if self.continuing and done: # make it a continuing task
        #     tr = 0
        #     tc = 0
        #     done = False

        reward = self.get_reward(tr, tc, done)
        self.move_to(tr, tc)
        return self.rc_to_index(tr, tc), reward, done

    def get_reward(self, sr, sc, done):
        if done:
            return self.goal_reward
        elif self.is_trap(sr, sc):
            return self.trap_penalty
        else:
            return self.step_penalty
