from mc_control import *


class Q_Learning_Control(object):
    def __init__(self):
        self.Q = QFunction()
        self.state_action_count = StateAction()
        self.in_use = False

    def set_in_use(self, is_in_use):
        self.in_use = is_in_use

    def start_new_episode(self):
        pass

    def next_action(self, state):
        """
        Behaviour policy: compeletely random
        Learnt policy: greedy wrt. Q
        :return: an action
        """
        if self.in_use:
            proposed_action = self.Q.get_action(state)
        else:
            proposed_action = np.random.choice((0, 1), size=1, p=(0.5, 0.5))[0]
            self.state_action_count.inc_value(state, proposed_action, 1.0)
        return proposed_action

    def update_policy_online(self, state, state_new, action, reward):
        if self.in_use:
            pass
        else:
            q_old = self.Q.get_value(state, action)
            sa_n = self.state_action_count.get_value(state, action)
            if state_new.is_terminal:
                alternative_q = 0.0
            else:
                alternative_action = self.Q.get_action(state_new)
                alternative_q = self.Q.get_value(state_new, alternative_action)
            self.Q.inc_value(state, action, (reward + alternative_q - q_old) / sa_n)

    def update_policy_batch(self, state, state_new, action, reward):
        pass


def Q_Learning_simulation(learning_episode, evaluation_episode):
    policy = Q_Learning_Control()
    for i in xrange(learning_episode):
        one_episode(policy)
    policy.Q.save_buffer('Q_Learning_Q')
    policy.set_in_use(True)
    success = [one_episode(policy) for i in xrange(evaluation_episode)]
    print sum(success)

if __name__ == '__main__':
    Q_Learning_simulation(1000000, 10000)
