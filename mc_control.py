from utils import *
from collections import Counter


class MC_Control(object):
    def __init__(self):
        self.Q = QFunction()
        self.episode_history = []
        self.state_action_count = StateAction()
        self.N0 = 10.0
        # self.alpha = 0.1

    def start_new_episode(self):
        self.episode_history = []

    def next_action(self, state):
        proposed_action = self.Q.get_action(state)
        n_min = min(
                self.state_action_count.get_value(state, 0),
                self.state_action_count.get_value(state, 1)
        )
        eps_t = self.N0 / (self.N0 + n_min + 1)
        action = np.random.choice((proposed_action, 1 - proposed_action), size=1, p=(1 - eps_t, eps_t))[0]
        self.state_action_count.inc_value(state, action, 1.0)
        self.episode_history.append((state, action))
        # debug
        # if state.get_signature() == (10, 4):
        #     print n_min
        #     print proposed_action
        #     print eps_t
        #     print action
        return action

    # called each time a new action is selected
    def update_policy_online(self, state, state_new, action, reward):
        pass

    # episode has ended: the result of (state, action) => new state is terminal
    def update_policy_batch(self, state, state_new, action, reward):
        state_action_lst = list(set(self.episode_history))
        for state, action_p in state_action_lst:
            q_old = self.Q.get_value(state, action_p)
            sa_n = self.state_action_count.get_value(state, action_p)
            self.Q.inc_value(state, action_p, (reward - q_old) / sa_n)


def mc_simulation(n_episode):
    policy = MC_Control()
    result = [one_episode(policy) for i in xrange(n_episode)]
    # print policy.state_action_count.buffer
    # V = policy.Q.get_V()
    print Counter(result)
    np.savetxt('mc_V.csv', policy.Q.get_V(), delimiter=',')
    # np.savetxt('Q_star_0.csv', policy.Q.get_Q()[:, :, 0], delimiter=',')
    # np.savetxt('Q_star_1.csv', policy.Q.get_Q()[:, :, 1], delimiter=',')
    policy.Q.save_buffer('mc_Q')
    policy.Q.check_q('checkQ.csv')



if __name__ == '__main__':
    mc_simulation(1000000)

    # TODO: plot V function
