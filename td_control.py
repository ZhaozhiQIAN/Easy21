from utils import *


class TD_Control(object):
    """
    Temporal difference control
    """
    def __init__(self, lam):
        self.Q = QFunction()
        self.state_action_count = StateAction()
        self.eligibility = EligibilityTrace()
        self.N0 = 10.0
        self.lam = lam

    def start_new_episode(self):
        self.eligibility.reset()

    def selection_action_only(self, state):
        proposed_action = self.Q.get_action(state)
        n_min = min(
                self.state_action_count.get_value(state, 0),
                self.state_action_count.get_value(state, 1)
        )
        eps_t = self.N0 / (self.N0 + n_min + 1)
        action = np.random.choice((proposed_action, 1 - proposed_action), size=1, p=(1 - eps_t, eps_t))[0]
        return action

    def next_action(self, state):
        action = self.selection_action_only(state)
        self.state_action_count.inc_value(state, action, 1.0)
        self.eligibility.inc_value(state, action, 1.0 / self.state_action_count.get_value(state, action))
        return action

    def update_policy_online(self, state, state_new, action, reward):
        if state_new.is_terminal:
            delta = reward - self.Q.get_value(state, action)
        else:
            action_new = self.selection_action_only(state_new)
            delta = reward + self.Q.get_value(state_new, action_new) - self.Q.get_value(state, action)
        self.Q.inc_batch(self.eligibility, delta)
        self.eligibility.scale(self.lam)

    def update_policy_batch(self, state, state_new, action, reward):
        pass


def td_simulation(n_episode, lam):
    policy = TD_Control(lam)
    mc_q = QFunction()
    mc_q.load_from_buffer('mc_Q.npy')
    sse = []
    for i in xrange(n_episode):
        one_episode(policy)
        # if i % 100 == 0:
        sse.append(policy.Q.mse(mc_q))
    # policy.Q.save_buffer('td_Q')
    # np.savetxt('V_star_td.csv', policy.Q.get_V(), delimiter=',')
    # np.savetxt('Q_star_0_td.csv', policy.Q.get_Q()[:, :, 0], delimiter=',')
    # np.savetxt('Q_star_1_td.csv', policy.Q.get_Q()[:, :, 1], delimiter=',')
    print 'Lambda = {}, Mean square error in episode {} is {}'.format(lam, n_episode, sse[-1])
    if lam == 1 or lam == 0:
        np.savetxt('mse_trace{}.csv'.format(int(lam)), np.asarray(sse))
    return sse[-1]


if __name__ == '__main__':
    sse = [td_simulation(10000, k) for k in np.arange(0.0, 1.1, 0.1)]
    sse_arr = np.asarray(sse)
    np.savetxt('lambda_vs_mse.csv', sse_arr)
    # td_simulation(1000000, 1)