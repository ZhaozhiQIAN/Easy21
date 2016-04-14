from utils import *


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
        self.episode_history.append((state.get_signature(), action))
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
        for state_sig, action_p in state_action_lst:
            q_old = self.Q.get_value_sig(state_sig, action_p)
            sa_n = self.state_action_count.get_value_sig(state_sig, action_p)
            self.Q.inc_value_sig(state_sig, action_p, (reward - q_old) / sa_n)


def mc_simulation(n_episode):
    policy = MC_Control()
    for i in xrange(n_episode):
        one_episode(policy)
    # print policy.state_action_count.buffer
    # V = policy.Q.get_V()
    # data = [go.Surface(z=V)]
    # layout = go.Layout(title='MC Value', scene=go.Scene(
    #     xaxis = dict(title='PlayerSum'),
    #     yaxis = dict(title='dealerShow')
    # ))
    # # fig = dict(data=data, layout=layout)
    # # iplot(fig)
    # fig = go.Figure(data=data, layout=layout)
    # py.sign_in('Krasusz', 'imt0cfohkl')
    #
    # plot_url = py.plot(fig, filename='surface_plot_V')

    np.savetxt('mc_V.csv', policy.Q.get_V(), delimiter=',')
    # np.savetxt('Q_star_0.csv', policy.Q.get_Q()[:, :, 0], delimiter=',')
    # np.savetxt('Q_star_1.csv', policy.Q.get_Q()[:, :, 1], delimiter=',')
    policy.Q.save_buffer('mc_Q')
    policy.Q.check_q('checkQ.csv')



if __name__ == '__main__':
    mc_simulation(1000000)

    # TODO: plot V function
