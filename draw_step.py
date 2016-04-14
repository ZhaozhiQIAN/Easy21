import numpy as np


class Card(object):
    def __init__(self):
        self.value = np.random.randint(low=1, high=11, size=1)[0]
        self.is_black = np.random.choice([True, False], size=1, p=[2.0/3, 1.0/3])[0]

    def __hash__(self):
        return hash((self.value, self.is_black))

    def __eq__(self, other):
        return (self.value, self.is_black) == (other.value, other.is_black)


class State(object):
    def __init__(self, dealer_first, player_sum, is_terminal):
        self.dealer_first = dealer_first
        self.player_sum = player_sum
        self.is_terminal = is_terminal
        self.burst = False

    def __hash__(self):
        return hash((self.dealer_first, self.player_sum))

    def __eq__(self, other):
        return (self.dealer_first, self.player_sum) == (other.dealer_first, other.player_sum)

    def copy(self):
        return State(self.dealer_first, self.player_sum, self.is_terminal)

    def get_signature(self):
        return self.dealer_first, self.player_sum

    def set_terminal(self):
        self.is_terminal = True

    def update_player_sum(self, a_card):
        if a_card.is_black:
            self.player_sum += a_card.value
        else:
            self.player_sum -= a_card.value
        # player go burst
        if self.player_sum > 21 or self.player_sum < 1:
            self.is_terminal = True
            self.burst = True

    def get_dealer_final_sum(self):
        dealer_sum = self.dealer_first
        while 0 < dealer_sum < 17:
            a_card = Card()
            if a_card.is_black:
                dealer_sum += a_card.value
            else:
                dealer_sum -= a_card.value
        return dealer_sum


def step(state, action):
    if state.is_terminal:
        Exception('Step on terminated state')

    state_ret = state.copy()
    if action == 0:  # hit
        state_ret.update_player_sum(Card())
    elif action == 1:  # stick
        state_ret.set_terminal()
    else:
        Exception('Unknown action')
    return state_ret


def get_reward_from_state(state):
    # non-terminal state: no reward
    if not state.is_terminal:
        return 0.0
    else:
        # player burst
        if state.burst:
            return -1.0
        else:
            dealer_sum = state.get_dealer_final_sum()
            if 0 < dealer_sum < 22:
                # dealer is greater
                if dealer_sum > state.player_sum:
                    return -1.0
                elif dealer_sum == state.player_sum:
                    return 0.0
                else:
                    return 1.0
            else:
                # dealer burst
                return 1.0


def check_draw(file_name, n_calls):
    cards = dict()
    for i in range(n_calls):
        a_card = Card()
        if a_card in cards:
            cards[a_card] += 1
        else:
            cards[a_card] = 1

    with open(file_name, mode='w') as f:
        for c in cards.keys():
            card_value = c.value
            card_color = int(2 * (c.is_black - 0.5))
            count = cards[c]*1.0/n_calls
            f.write(','.join((str(card_value), str(card_color), str(count)))+'\n')
    f.close()


def check_step(n_calls):
    # input
    config_lst = ((1, 1, 0), (1, 10, 0), (1, 18, 1), (10, 15, 1))
    for config in config_lst:
        init_state = State(config[0], config[1], False)
        state_reward = dict()
        for i in range(n_calls):
            cp_state = init_state.copy()
            a_state = step(cp_state, config[2])
            a_reward = get_reward_from_state(a_state)
            if (a_state.get_signature(), a_reward) in state_reward:
                state_reward[(a_state.get_signature(), a_reward)] += 1
            else:
                state_reward[(a_state.get_signature(), a_reward)] = 1

        with open('checkStepDealer{}Player{}Action{}.csv'.format(config[0], config[1], config[2]), mode='w') as f:
            for c in sorted(state_reward.keys()):
                dealer_first, player_sum = c[0]
                a_reward = c[1]
                freq = state_reward[c] * 1.0 / n_calls
                f.write(','.join((str(dealer_first), str(player_sum), str(a_reward), str(freq))) + '\n')
        f.close()


if __name__ == '__main__':
    check_draw('checkDraw.csv', 1000)
    check_step(1000)