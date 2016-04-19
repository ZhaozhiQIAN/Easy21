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

    # increment the value by inc
    def inc_value(self, state, action, inc):
        state_sig = state.get_signature()
        self.buffer[state_sig[0] - 1, state_sig[1] - 1, action] += inc

    # set the value to be val
    def set_value(self, state, action, val):
        state_sig = state.get_signature()
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


def sample_indices(prob_mat):
    probs = np.reshape(prob_mat, prob_mat.size)
    location = np.random.choice(np.arange(prob_mat.size), size=1, p=probs)[0]
    row = location / 21
    column = location % 21
    return row, column


class StateTransition(object):
    """
    modeling state transition
    """
    def __init__(self):
        # conditional transition probability to non-terminal state
        self.buffer = np.ones((21, 21), dtype=np.float64) / 21.0
        # transition to terminal state
        self.to_terminal = StateAction()
        self.to_terminal.buffer[:, :, 0] = 0.5
        self.to_terminal.buffer[:, :, 1] = 1.0
        # state-action count
        # self.count = StateAction()
        # self.count.buffer += 1
        self.player_sum_count = np.ones(21)

    def get_next_state(self, state, action):
        prob_terminal = self.to_terminal.get_value(state, action)
        next_terminal = np.random.choice((False, True), size=1, p=(1.0 - prob_terminal, prob_terminal))[0]
        if next_terminal:
            next_state = state.copy()
            next_state.set_terminal()
            if action == 0:
                next_state.player_sum = -1
            return next_state
        else:
            state_sig = state.get_signature()
            prob = self.buffer[state_sig[1] - 1, :]
            next_player_sum = np.random.choice(np.arange(1, 22), size=1, p=prob)[0]
            return State(state_sig[0], next_player_sum, False)

    def update_model(self, state, action, new_state):
        # t = self.count.get_value(state, action)
        state_sig = state.get_signature()
        t = self.player_sum_count[state_sig[1] - 1]
        if new_state.is_terminal:
            # update terminal probability
            old_prob = self.to_terminal.get_value(state, action)
            new_prob = (old_prob * t + 1.0) / (t + 1.0)
            self.to_terminal.set_value(state, action, new_prob)
        else:
            # update conditional probability
            new_state_sig = new_state.get_signature()
            term_prob = self.to_terminal.get_value(state, action)
            old_prob = self.buffer[state_sig[1] - 1, new_state_sig[1] - 1]
            new_prob = (old_prob * (1.0 - term_prob) * t + 1) / ((1.0 - term_prob) * t + 1.0)
            self.buffer[state_sig[1] - 1, new_state_sig[1] - 1] = new_prob
            self.buffer[state_sig[1] - 1, :] *= 1 / self.buffer[state_sig[1] - 1, :].sum()
            # update terminal probability
            new_term_prob = (term_prob * t) / (t + 1.0)
            self.to_terminal.set_value(state, action, new_term_prob)
        # self.count.inc_value(state, action, 1)
        self.player_sum_count[state_sig[1] - 1] += 1

