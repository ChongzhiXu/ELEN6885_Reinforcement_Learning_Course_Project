import tensorflow
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Dense, Input, Flatten, Add
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
import numpy as np
from tensorflow.python.framework.ops import disable_eager_execution
disable_eager_execution()


class AC_Strategy(object):
    def __init__(self, alpha, beta, gamma=0.99, n_actions=3, input_dims=10, win_reward=1, tie_reward=-0.2, loss_reward=-0.8):
        self.alpha = alpha  # actor learning rate
        self.beta = beta  # critic learning rate
        self.gamma = gamma  # discount factor
        self.n_actions = n_actions  # 0,1,2 for Rock, Paper, Scissors
        self.input_dims = input_dims
        self.actor, self.policy = self.create_actor_model()
        self.critic = self.create_critic_model()
        self.action_space = [i for i in range(self.n_actions)]
        self.win_reward = win_reward
        self.tie_reward = tie_reward
        self.loss_reward = loss_reward

        self.counter_dictionary = {
            "R": "S",  # rock counters scissors
            "P": "R",  # paper counters rock
            "S": "P"  # scissors counters paper
        }
        self.RPS_list = ["R", "P", "S"]
        self.current_state = None
        self.reward = 0

    def create_actor_model(self):
        state_input = Input(shape=(self.input_dims,))
        delta_input = Input(shape=(self.n_actions,))
        dense1 = Dense(24, activation='relu')(state_input)
        dense2 = Dense(48, activation='relu')(dense1)
        dense3 = Dense(24, activation='relu')(dense2)
        output = Dense(self.n_actions, activation='softmax')(dense3)

        def custom_loss(y_true, y_pred):
            # Clip the y prediction so that won't be 1 or 0
            y_pred_clip = K.clip(y_pred, 1e-8, 1-1e-8)
            log_lik = y_true * K.log(y_pred_clip)
            loss = K.sum(-log_lik * delta_input)
            return loss

        adam = Adam(learning_rate=self.alpha)
        actor_model = Model(inputs=[state_input, delta_input], outputs=output)
        actor_model.compile(loss=custom_loss, optimizer=adam)
        # Policy model no needs for compile!!!
        policy_model = Model(inputs=[state_input], outputs=output)

        return actor_model, policy_model

    def create_critic_model(self):
        state_input = Input(shape=(self.input_dims,))
        state_dense1 = Dense(24, activation='relu')(state_input)
        state_dense2 = Dense(48, activation='relu')(state_dense1)

        action_input = Input(shape=(self.n_actions,))
        action_dense1 = Dense(48)(action_input)

        merged = Add()([state_dense2, action_dense1])
        merge_dense1 = Dense(24, activation='relu')(merged)
        output = Dense(3, activation='relu')(merge_dense1)

        model = Model(inputs=[state_input, action_input], outputs=output)
        adam = Adam(learning_rate=self.beta)
        model.compile(loss='mse', optimizer=adam)
        return model

    def Get_Reward(self, opponent_play, self_play):
        """Called by Epsilon_Strategy.Learn, get the reward for agent"""
        assert type(opponent_play) == type(self_play) == str
        if self.counter_dictionary[self_play] == opponent_play:
            self.reward = self.win_reward
        elif opponent_play == self_play:
            self.reward = self.tie_reward
        else:
            self.reward = self.loss_reward

    def Next_Move(self):
        """
        In each round, Epsilon_Strategy.Next_Move will be called
        :return: Return a string "R" or "P" or "S" indicating your play in this round,
        Your are allowed to play randomly in the first several rounds,
        opponent play in this round is not provided.
        """
        if self.current_state is None:  # Play randomly in the first 10 rounds
            action_index = np.random.randint(0, self.n_actions)
        else:
            probabilities = self.policy.predict(self.current_state)[0]
            # select action follows the Model result probabilities
            action_index = np.random.choice(self.action_space, p=probabilities)
        next_move = self.RPS_list[action_index]
        return next_move

    def Learn(self, opponent_history_str, self_history_str):
        """
        After each round, Another_Strategy.Learn will be called, provided with opponent_history & self_history to learn.
        :param opponent_history_str: Most recent play history of opponent, from the very beginning, in string.
        :param self_history_str: Most recent play history of Another_Strategy, from the very beginning, in string.
        """
        assert len(opponent_history_str) == len(self_history_str)
        history_length = len(self_history_str)
        # Start learning when length of history longer than 10 rounds --------------------------------------------------
        if history_length > 10:
            self.Get_Reward(opponent_history_str[-2], self_history_str[-2])
            # Convert 'R','P','S' to 0,1,2 -----------------------------------------------------------------------------
            opponent_history = [0 for _ in range(history_length)]
            self_history = [0 for _ in range(history_length)]
            for i in range(history_length):
                opponent_history[i] = self.RPS_list.index(opponent_history_str[i])
                self_history[i] = self.RPS_list.index(self_history_str[i])
            # e.g. opponent_history = [0,1,2,2,2,1,0,0,1,0]
            # Update state, next state, action according to history ----------------------------------------------------
            opponent_history = np.array(opponent_history)
            self_history = np.array(self_history)
            state = opponent_history[-self.input_dims-1:-1] #+ 7*self_history[-self.input_dims-1:-1]
            next_state = opponent_history[-self.input_dims:] #+ 7*self_history[-self.input_dims:]
            # Most recent 10 rounds of competition history, e.g.[[0,0],[1,0],...[1,2]]
            state = state[np.newaxis, :]
            next_state = next_state[np.newaxis, :]
            self.current_state = next_state
            # Instead of rewards in other Reinforce Learning algorithm, AC use critic value ----------------------------
            critic_value_next_state = self.policy.predict(next_state)
            critic_value = self.policy.predict(state)
            print('critic_value',critic_value)
            target = self.reward + self.gamma * critic_value_next_state  # Lack reward!!!
            delta = target - critic_value

            # one_hot format of actions --------------------------------------------------------------------------------
            action = self.RPS_list.index(self.Next_Move())
            actions = np.zeros([1, self.n_actions])
            actions[0, action] = 1.0

            # Finally, use Model.fit to learn!!! -----------------------------------------------------------------------
            self.actor.fit([state, delta], actions, verbose=0)
            self.critic.fit([state, actions], target, verbose=0)


