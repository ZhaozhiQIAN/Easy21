from utils import QFunction, one_episode
from draw_step import State
import numpy as np


class Approx_TD_Control(object):
    def __init__(self, lam, eps=0.1, alpha=0.01):
        self.theta = np.zeros(36)
        self.eligibility = np.zeros(36)
        self.N0 = 10.0
        self.lam = lam
        self.eps = eps
        self.alpha = alpha
        self.dealer_first_predicates = [
            lambda x: 1 <= x <= 4,
            lambda x: 4 <= x <= 7,
            lambda x: 7 <= x <= 10
        ]
        self.player_sum_predicates = [
            lambda x: 1 <= x <= 6,
            lambda x: 4 <= x <= 9,
            lambda x: 7 <= x <= 12,
            lambda x: 10 <= x <= 15,
            lambda x: 13 <= x <= 18,
            lambda x: 16 <= x <= 21
        ]

    def get_feature(self, state, action):
        dealer_feature = np.asarray([f(state.get_signature()[0]) for f in self.dealer_first_predicates], dtype=np.float64)
        player_feature = np.asarray([f(state.get_signature()[1]) for f in self.player_sum_predicates], dtype=np.float64)
        action_feature = np.asarray([action == 0, action != 0], dtype=np.float64)
        feature = np.reshape(np.outer(np.outer(dealer_feature, player_feature), action_feature), newshape=36)
        return feature

    def get_value(self, state, action):
        return np.dot(self.theta, self.get_feature(state, action))

    def start_new_episode(self):
        self.eligibility = np.zeros(36)

    def selection_action_only(self, state):
        value0 = self.get_value(state, 0)
        value1 = self.get_value(state, 1)
        if value0 > value1:
            proposed_action = 0
        elif value0 < value1:
            proposed_action = 1
        else:
            proposed_action = np.random.choice(2, 1)[0]
        action = np.random.choice((proposed_action, 1 - proposed_action), size=1, p=(1 - self.eps, self.eps))[0]
        return action

    def next_action(self, state):
        action = self.selection_action_only(state)
        self.eligibility = self.lam * self.eligibility + self.get_feature(state, action)
        return action

    def update_policy_online(self, state, state_new, action, reward):
        if state_new.is_terminal:
            delta = reward - self.get_value(state, action)
        else:
            action_new = self.selection_action_only(state_new)
            delta = reward + self.get_value(state_new, action_new) - self.get_value(state, action)
        self.theta += self.alpha * delta * self.eligibility

    def update_policy_batch(self, state, state_new, action, reward):
        pass

    def get_Q(self):
        mat = np.zeros((10, 21, 2), dtype=np.float64)
        for dealer_card in range(10):
            for player_sum in range(21):
                for action in range(2):
                    mat[dealer_card, player_sum, action] = self.get_value(State(dealer_card, player_sum, False), action)
        Q = QFunction()
        Q.buffer = mat
        return Q


def approx_td_simulation(n_episode, lam):
    policy = Approx_TD_Control(lam)
    mc_q = QFunction()
    mc_q.load_from_buffer('mc_Q.npy')
    sse = []
    for i in xrange(n_episode):
        one_episode(policy)
        # if i % 100 == 0:
        sse.append(mc_q.mse(policy.get_Q()))
    policy.get_Q().save_buffer('td_Q')
    # np.savetxt('V_star_td.csv', policy.Q.get_V(), delimiter=',')
    # np.savetxt('Q_star_0_td.csv', policy.Q.get_Q()[:, :, 0], delimiter=',')
    # np.savetxt('Q_star_1_td.csv', policy.Q.get_Q()[:, :, 1], delimiter=',')
    print 'Lambda = {}, Mean square error in episode {} is {}'.format(lam, n_episode, sse[-1])
    if lam == 1 or lam == 0:
        np.savetxt('approx_td_mse_trace{}.csv'.format(int(lam)), np.asarray(sse))
    return sse[-1]

if __name__ == '__main__':
    sse = [approx_td_simulation(10000, k) for k in np.arange(0.0, 1.1, 0.1)]
    sse_arr = np.asarray(sse)
    np.savetxt('lambda_vs_mse_approx.csv', sse_arr)