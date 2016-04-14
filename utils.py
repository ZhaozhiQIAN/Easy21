from draw_step import *


class StateAction(object):
    """
    The implementation of (state, action) => value
    For Q function, eligibility trace and counts
    """
    def __init__(self):
        self.buffer = np.zeros([10, 21, 2])

    # get value of a given state-action pair
    def get_value(self, state, action):
        state_sig = state.get_signature()
        return self.buffer[state_sig[0] - 1, state_sig[1] - 1, action]

    def get_value_sig(self, state_sig, action):
        return self.buffer[state_sig[0] - 1, state_sig[1] - 1, action]

    # increment the value by inc
    def inc_value(self, state, action, inc):
        state_sig = state.get_signature()
        self.buffer[state_sig[0] - 1, state_sig[1] - 1, action] += inc

    def inc_value_sig(self, state_sig, action, inc):
        self.buffer[state_sig[0] - 1, state_sig[1] - 1, action] += inc

    # set the value to be val
    def set_value(self, state, action, val):
        state_sig = state.get_signature()
        self.buffer[state_sig[0] - 1, state_sig[1] - 1, action] = val

    def set_value_sig(self, state_sig, action, val):
        self.buffer[state_sig[0] - 1, state_sig[1] - 1, action] = val

    def reset(self):
        self.buffer = np.zeros([10, 21, 2])

    def save_buffer(self, path):
        np.save(path, self.buffer)

    def load_from_buffer(self, path):
        self.buffer = np.load(path)

    def check_q(self, path):
        rows = []
        for dealer_card in range(10):
            for player_sum in range(21):
                for action in range(2):
                    rows.append((dealer_card + 1, player_sum + 1, action, self.buffer[dealer_card, player_sum, action]))
        mat = np.asarray(rows)
        np.savetxt(path, mat, delimiter=',')


class QFunction(StateAction):
    # get the action with maximum value given a state
    def get_action(self, state):
        state_sig = state.get_signature()
        return np.argmax(self.buffer[state_sig[0] - 1, state_sig[1] - 1, :])

    def get_Q(self):
        return self.buffer

    def get_V(self):
        return np.max(self.buffer, axis=2)

    def inc_batch(self, state_action, mul):
        self.buffer += state_action.buffer * mul

    def mse(self, Q):
        return np.mean((self.buffer - Q.buffer) ** 2)


class EligibilityTrace(StateAction):
    def scale(self, lam):
        self.buffer *= lam


def one_episode(policy):
    card_for_dealer = Card()
    card_for_player = Card()
    state = State(card_for_dealer.value, card_for_player.value, False)
    policy.start_new_episode()
    while True:
        action = policy.next_action(state)
        state_new = step(state, action)
        reward = get_reward_from_state(state_new)
        if state_new.is_terminal:
            policy.update_policy_online(state, state_new, action, reward)
            policy.update_policy_batch(state, state_new, action, reward)
            break
        else:
            policy.update_policy_online(state, state_new, action, reward)
        state = state_new
    return get_reward_from_state(state_new)

