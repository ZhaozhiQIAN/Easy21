from utils import *
from collections import Counter


class Dyna_Q_Control(object):
    def __init__(self):
        self.Q = QFunction()
        self.state_action_count = StateAction()
        self.in_use = False
        self.simulate_rounds = 100
        self.state_transition = StateTransition()
        self.expected_reward = StateAction()

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

    def update_policy_from_model(self):
        for i in xrange(self.simulate_rounds):
            state0 = State(
                    np.random.randint(low=1, high=11, size=1)[0],
                    np.random.randint(low=1, high=22, size=1)[0],
                    False
            )
            action0 = np.random.randint(low=0, high=2, size=1)[0]
            state1 = self.state_transition.get_next_state(state0, action0)
            reward1 = self.expected_reward.get_value(state0, action0)
            q_old = self.Q.get_value(state0, action0)
            sa_n = self.state_action_count.get_value(state0, action0)
            if sa_n == 0:
                return
            if state1.is_terminal:
                alternative_q = 0.0
            else:
                alternative_action = self.Q.get_action(state1)
                alternative_q = self.Q.get_value(state1, alternative_action)
            self.Q.inc_value(state0, action0, (reward1 + alternative_q - q_old) / sa_n)

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
            # update model
            self.state_transition.update_model(state, action, state_new)
            # update reward
            old_reward = self.expected_reward.get_value(state, action)
            new_reward = (old_reward * (sa_n - 1) + reward) / sa_n
            self.expected_reward.set_value(state, action, new_reward)
            # update with simulated experience
            if self.state_action_count.buffer.sum() < 10000:
                self.simulate_rounds = 0
            else:
                self.simulate_rounds = 10
            self.update_policy_from_model()

    def update_policy_batch(self, state, state_new, action, reward):
        pass


def Dyna_Q_simulation(learning_episode, evaluation_episode):
    policy = Dyna_Q_Control()
    for i in xrange(learning_episode):
        one_episode(policy)
    policy.Q.save_buffer('Dyna_Q_Q')
    print policy.state_transition.to_terminal.buffer[:, :, 0]
    print policy.state_transition.to_terminal.buffer[:, :, 1]
    np.save('transition_mat', policy.state_transition.buffer)
    policy.set_in_use(True)
    result = [one_episode(policy) for i in xrange(evaluation_episode)]
    print Counter(result)

if __name__ == '__main__':
    Dyna_Q_simulation(100000, 10000)
