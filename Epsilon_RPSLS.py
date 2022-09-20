import numpy as np


class Epsilon_Strategy_RPSLS:

    def __init__(self, alpha=0.5, gamma=0.2, epsilon=0.1, episodes_to_min_eps=100, min_eps=0.01):
        # Counter relationships dictionary ----------------------------------------
        self.counter_dictionary = {
            "Rock": ["Scissors", "Lizard"],  # Rock crushes Scissors, Rock crushes Lizard
            "Paper": ["Rock", "Spock"],  # Paper covers Rock，Paper disproves Spock
            "Scissors": ["Paper", "Lizard"],  # Scissors cuts Paper, Scissors decapitates Lizard
            "Lizard": ["Paper", "Spock"],  # Lizard eats Paper, Lizard poisons Spock
            "Spock": ["Rock", "Scissors"]  # Spock vaporizes Rock, Spock smashes Scissors
        }
        # Keep track of competition history ----------------------------------------
        self.opponent_history = []
        self.self_history = []
        # Q Matrix and parameters --------------------------------------------------
        self.q_matrix = np.random.uniform(low=0, high=0, size=(5, 5))  # random matrix with size 3*3
        self.alpha = alpha  # Used for updating q matrix
        self.gamma = gamma  # Used for updating q matrix
        # epsilon is reduced to min epsilon in episodes ----------------------------
        self.epsilon = epsilon
        self.cur_epsilon = epsilon
        self.episodes_to_min_eps = episodes_to_min_eps  # Used for reducing epsilon
        self.min_eps = min_eps  # Used for reducing epsilon
        # --------------------------------------------------------------------------
        self.RPSLS_list = ["Rock", "Paper", "Scissors", "Lizard", "Spock"]
        self.action_space = 5
        self.action = 0
        self.result = ""

    def Next_Move(self, ):
        """
        In each round, Epsilon_Strategy.Next_Move will be called
        :return: Return a string "R" or "P" or "S" indicating your play in this round,
        Your are allowed to play randomly in the first TWO rounds,
        opponent play in this round is not provided.
        """
        assert len(self.opponent_history) == len(self.self_history)
        history_length = len(self.self_history)
        if history_length <= 1:  # Play randomly in the first TWO rounds
            self.action = np.random.randint(0, self.action_space)
        else:  # Play Epsilon greedy in the further rounds
            opponent_prev_move = self.RPSLS_list.index(self.opponent_history[-1])
            if np.random.random() < 1 - self.cur_epsilon:  # 1-epsilon chances to Exploit
                self.action = np.argmax(self.q_matrix[opponent_prev_move])
            else:  # epsilon chances to Explore
                self.action = np.random.randint(0, self.action_space)
            # Decay epsilon
            eps_reduction = (self.epsilon - self.min_eps) / self.episodes_to_min_eps
            if self.cur_epsilon > self.min_eps:
                self.cur_epsilon -= eps_reduction
        return self.RPSLS_list[self.action]

    def Learn(self, opponent_history, self_history):
        """
        After each round, Another_Strategy.Learn will be called, provided with opponent_history & self_history to learn.
        :param opponent_history: Most recent play history of opponent, from the very beginning.
        :param self_history: Most recent play history of Another_Strategy, from the very beginning.
        """
        assert len(opponent_history) == len(self_history)
        history_length = len(self_history)
        if history_length >= 2:  # Play randomly in the first TWO rounds
            np.random.randint(0, self.action_space)
            self.opponent_history = opponent_history
            self.self_history = self_history

            reward = self.Get_Reward(self.opponent_history[-2], self.self_history[-2])
            state = self.RPSLS_list.index(self.opponent_history[-2])  # opponent previous, previous play index as state
            next_state = self.RPSLS_list.index(self.opponent_history[-1])  # opponent previous play index as next state
            action = self.RPSLS_list.index(self.self_history[-2])  # self previous play index as action

            self.Update_Q_Matrix(state, action, reward, next_state)
            return self.q_matrix

    def Update_Q_Matrix(self, state, action, reward, next_state):
        """
        Called by Epsilon_Strategy.Learn, help update the Q matrix
        :param state: opponent previous, previous play index as state
        :param action: self previous play index as action
        :param reward: reward from previous, previous play
        :param next_state: opponent previous play index as next state
        :return:
        """
        reward_from_next_state = np.max(self.q_matrix[next_state])
        delta = self.alpha * (reward + self.gamma * reward_from_next_state - self.q_matrix[state, action])
        self.q_matrix[state, action] += delta

    def Get_Reward(self, opponent_play, self_play):
        """Called by Epsilon_Strategy.Learn, get the reward for agent"""
        assert type(opponent_play) == type(self_play) == str
        if opponent_play in self.counter_dictionary[self_play]:
            self.result, reward = 'WIN', 5
        elif opponent_play == self_play:
            self.result, reward = 'TIE', -2
        else:
            self.result, reward = 'LOSS', -4
        return reward
