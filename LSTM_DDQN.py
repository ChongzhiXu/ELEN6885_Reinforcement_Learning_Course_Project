import numpy as np
import random
import math
import tensorflow
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, GRU, TimeDistributed, Flatten
from tensorflow.keras.optimizers import Adam
from collections import deque
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')

# -------------------------- SETTING UP THE ENVIRONMENT --------------------------------------
# simple game, therefore we are not using the open gym custom set up
# --------------------------------------------------------------------------------------------
class RPS_environment():
    def __init__(self):
        '''
        Initialize the following:
        1. integer representation of r/p/s
        2. random seed: make it deterministic
        3. player 1,2 win tie lost count
        4. history size for rate trending calculation
        5. sigma for std distribution
        6. parameter used for calculating the win rate and store
        7. put all the observation state in here; shape in Keras input format
        '''
        self.action_space = [0, 1, 2]
        self.seed = random.seed(4)
        self.p1_statistic = [0, 0, 0]
        self.p2_statistic = [0, 0, 0]
        self.history = 10
        self.norm_sigma = 2.0
        self.current_WinRate, self.current_TieRate, self.current_LostRate = None, None, None
        self.last_WinRate, self.last_TieRate, self.last_LostRate = 0, 0, 0
        self.current_WinNum, self.current_TieNum, self.current_LostNum = None, None, None
        self.win_Trend, self.tie_Trend, self.lost_Trend = 0, 0, 0
        self.win_change_Avg, self.tie_change_Avg, self.lost_change_Avg = 0, 0, 0
        self.win_Buffer, self.tie_Buffer, self.lost_Buffer \
            = deque(maxlen=self.history), deque(maxlen=self.history), deque(maxlen=self.history)
        self.state = np.array([[None, None, None, self.win_Trend, self.tie_Trend, self.lost_Trend,
                                self.win_change_Avg, self.tie_change_Avg, self.lost_change_Avg]])

    def restoration(self):
        '''
        reset all the state to the initilalization state
        :return: empty list(array) for rate, number, trend
        '''
        self.current_WinRate, self.current_TieRate, self.current_LostRate = 0, 0, 0
        self.current_WinNum, self.current_TieNum, self.current_LostNum = 0, 0, 0
        self.win_Trend, self.tie_Trend, self.lost_Trend = 0, 0, 0
        self.win_change_Avg, self.tie_changing_Avg, self.lost_change_Avg = 0, 0, 0
        return np.array([0, 0, 0, 0, 0, 0, 0, 0, 0])

    def round_step(self, action, move_count, stage):
        '''
        Define a round of game, process.
        :param action: the action that computer give
        :param move_count: the number of round of game in an episode
        :param stage: distribution stage
        :return:
        '''
        print('The {0} round of the game:'.format(move_count))
        def get_key(dictionary, val):
            for key, value in dictionary.items():
                if val == value:
                    return key
            return "key doesn't exist"

        dict = {'r': 0, 'p': 1, 's': 2}
        while True:
            # human_play = 'r'
            human_play = input('please input your choice r/p/s:')
            if human_play in ['r', 'p', 's']:
                p2Move = dict[human_play]
                self.p2_statistic[p2Move] += 1
                p1Move = action
                p1_play = get_key(dict, action)
                print('Your rival choice:', p1_play)
                self.p1_statistic[p1Move] += 1
                break
            else:
                print('Please enter the right choice representation!')

        # check who won, set flag and assign reward
        win, tie, lost = 0, 0, 0
        if p1Move == p2Move:
            print('Result: TIE')
            self.current_TieNum, tie = self.current_TieNum + 1, 1
        elif (p1Move - p2Move == 1) or (p1Move - p2Move == -2):
            print('Result : YOU LOST')
            self.current_WinNum, win = self.current_WinNum + 1, 1
        else:
            self.current_LostNum, lost = self.current_LostNum + 1, 1
            print('Result : YOU WIN')

        # update the running rates
        self.current_WinRate = self.current_WinNum / move_count
        self.current_TieRate = self.current_TieNum / move_count
        self.current_LostRate = self.current_LostNum / move_count
        print('Your win rate is:', self.current_LostRate)
        print('LSTM win rate is:', self.current_WinRate)
        print('')
        # update moving avg buffer
        self.win_Buffer.append(self.current_WinRate)
        self.tie_Buffer.append(self.current_TieRate)
        self.lost_Buffer.append(self.current_LostRate)
        # calculate trend
        tmp = [0, 0, 0]
        self.win_Trend, self.tie_Trend, self.lost_Trend = 0, 0, 0
        if move_count >= self.history:
            tmp[0] = sum(self.win_Buffer[i] for i in range(self.history)) / self.history
            tmp[1] = sum(self.tie_Buffer[i] for i in range(self.history)) / self.history
            tmp[2] = sum(self.lost_Buffer[i] for i in range(self.history)) / self.history
            # Win rate trend analysis
            if self.win_change_Avg < tmp[0]:
                self.win_Trend = 1  # win rate trending up. That's good
            else:
                self.win_Trend = 0  # win rate trending down. That's bad
            # Tie rate trend analysis
            if self.tie_change_Avg < tmp[1]:
                self.tie_Trend = 1  # tie rate trending up. That's bad
            else:
                self.tie_Trend = 0  # tie rate trending down.  Neutral
            # Lost rate trend analysis
            if self.lost_change_Avg < tmp[2]:
                self.lost_Trend = 1  # lst rate trending up.  That's bad
            else:
                self.lost_Trend = 0  # lost rate trending down. That's good
            self.win_change_Avg, self.tie_change_Avg, self.lost_change_Avg = tmp[0], tmp[1], tmp[2]
        # assign the reward in this round of game
        reward = win
        # record the state and reshape it for keras input format
        dim = self.state.shape[1]
        self.state = np.array([win, tie, lost, self.win_Trend, self.tie_Trend, self.lost_Trend, self.win_change_Avg, self.tie_change_Avg, self.lost_change_Avg]).reshape(1, dim)
        # this game is done when it hits this goal
        done = False
        return self.state, reward, done, dim


# ------------------------- class for the Double-DQN agent ---------------------------------
# Double DQN networks: one for behavior policy, one for target policy
# -------------------------------------------------------------------------------------------
class DDQN:
    def __init__(self, env):
        '''
        Initialize the following:
        1. initialize the memory and auto drop when memory exceeds maxlength
        2. this controls how far out in history the "expeience replay" can select from
        3. future reward discount rate of the max Q of next state
        4. epsilon denotes the fraction of time dedicated to exploration (as oppse to exploitation)
        5. model learning rate (use in backprop SGD process)
        6. transfer learning proportion control between the target and action/behavioral NN
        7. hyyperparameters for LSTM
        8. create two models for double-DQN implementation
        9. some space to collect TD target for instrumentaion
        :param env: Used by the derivatives.
        '''
        self.env = env
        self.maxlength = 3000
        self.memory = deque(maxlen=self.maxlength)
        self.gamma = 0.9
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay_rate = 0.9910
        self.learning_rate = 0.005
        self.tau = .125
        self.look_back = 15
        self.hidden_nerve_cell = 50
        self.model = self.create_model()
        self.target_model = self.create_model()
        self.TDtargetdelta, self.TDtarget = [], []
        self.Qmax = []

    def create_model(self):
        '''
        This function defines the LSTM/GRU cell by tensorflow keras, remember not to use activation.
        :return: the  LSTM/GRU model with dense(fc layer)
        '''
        input_dimension = self.env.state.shape[1]
        output_dimension = len(self.env.action_space)
        model = Sequential()
        # model.add(GRU(self.hiddenUnits,\
        model.add(LSTM(self.hidden_nerve_cell, \
                       return_sequences=False, \
                       # activation = None, \
                       # recurrent_activation = None, \
                       input_shape=(self.look_back, input_dimension)))
        # let the output be the predicted target value.
        # NOTE: do not use activation to squash it!
        model.add(Dense(output_dimension))
        model.compile(loss="mean_squared_error", optimizer=Adam(learning_rate=self.learning_rate))
        print(model.summary())
        return model

    def act(self, state, step):
        '''
        This function is to take one action(from computer side)
        :param state: the current input line and win/lost trend/rate/numbers
        :param step: round step
        :return: the prediction the model made
        '''
        self.epsilon *= self.epsilon_decay_rate
        self.epsilon = max(self.epsilon_min, self.epsilon)
        # decide to take a random exploration or make a policy-based action (through NN prediction)
        # with a LSTM design, delay on policy prediction after at least lookback steps have accumlated
        if np.random.random() < self.epsilon or step < self.look_back + 1:
            # return a random move from action space
            return random.choice(self.env.action_space)
        else:
            # return a policy move
            state_set = np.empty((1, self.env.state.shape[1]))  # iniitial with 2 dims
            for j in range(self.look_back):
                state_tmp, _, _, _, _ = self.memory[-(j + 1)]  # get the most recent state and the previous N states
                if j == 0:
                    state_set[0] = state_tmp  # iniitalize the first record
                else:
                    # get a consecutive set of states for LSTM prediction
                    state_set = np.concatenate((state_set, state_tmp), axis=0)
            state_set = state_set[None, :, :]  # make the tensor 3 dim to align with Keras reqmt
            self.Qmax.append(max(self.model.predict(state_set)[0]))
            print('Q value for this round:',max(self.model.predict(state_set)[0]))
            return np.argmax(self.model.predict(state_set)[0])

    def remember_memory(self, state, action, reward, new_state, done):
        '''
        Store up a big pool of memory
        :param state: the current input line and win/lost trend/rate/numbers
        :param action: the action that computer give
        :param reward: the reward assigned based on round result
        :param new_state: define the next state, initialize for it
        :param done: judge if a episode is over
        :return: a backup for the current state
        '''
        self.memory.append([state, action, reward, new_state, done])

    def DeepMind_backup(self):
        '''
        DeepMind "experience replay" method, do the training (learning);
        This is DeepMind tricks of using "Double" model the sample size from memory to learn from
        This function will do nothing untl the memory is large enough
        It first get the samples; each sample is a sequence of consecutive states with same lookback length as LSTM definition
        Then check if memory is large enough to retrieve the time sequence
        Then get a random location and retrieve a sample from memory at that location; latest element at the end of deque
        At last, get a consecutive set of states for LSTM prediction
        :return: the choice of how to give the output
        '''
        RL_batch_size = 24  # this is experience replay batch_size (not the LSTM fitting batch size)
        if len(self.memory) < RL_batch_size: return
        #
        for i in range(RL_batch_size):
            state_set = np.empty((1, self.env.state.shape[1]))
            new_state_set = np.empty((1, self.env.state.shape[1]))
            if len(self.memory) <= self.look_back:  #
                return
            else:
                # get a random location
                a = random.randint(0, len(self.memory) - self.look_back)
            state, action, reward, new_state, done = self.memory[-(a + 1)]

            for j in range(self.look_back):
                # get a consecutive set of states
                state_tmp, _, _, new_state_tmp, _ = self.memory[-(a + j + 1)]
                if j == 0:
                    state_set[0] = state_tmp
                    new_state_set[0] = new_state_tmp
                else:
                    # get a consecutive set of states for LSTM prediction
                    state_set = np.concatenate((state_set, state_tmp), axis=0)
                    new_state_set = np.concatenate((new_state_set, new_state_tmp), axis=0)
                    # do the prediction from current state
            state_set = state_set[None, :, :]  # make the tensor 3 dimension to align with Keras requirement
            new_state_set = new_state_set[None, :, :]  # make the tensor 3 dimension to align with Keras requirment
            target = self.target_model.predict(state_set)
            # Q leanring
            if done:
                target[0][action] = reward
            else:
                Q_future = max(self.target_model.predict(new_state_set)[0])
                TDtarget = reward + Q_future * self.gamma
                self.TDtarget.append(TDtarget)
                self.TDtargetdelta.append(TDtarget - target[0][action])
                target[0][action] = TDtarget
            # do one pass gradient descend using target as 'label' to train the action model
            self.model.fit(state_set, target, batch_size=1, epochs=1, verbose=0)

    def target_weight_update(self):
        '''
        Transfer weights  proportionally from the action/behave model to the target model
        :return: Noting but the update of target prediction weights
        '''
        weights = self.model.get_weights()
        target_weights = self.target_model.get_weights()
        for i in range(len(target_weights)):
            target_weights[i] = weights[i] * self.tau + target_weights[i] * (1 - self.tau)
        self.target_model.set_weights(target_weights)
    # def save_model(self, fn):
    #     self.model.save(fn)


# ------------------------- MAIN BODY ----------------------------------------

def main():
    '''
    Give the:
    1. length of game play
    2. stages with change distribution
    3. sigma change amount at each stage
    4. init for intrumentation
    5. parameters to record and caculate the Q value
    6. declare the game play environment and AI agent
    :return: Initialization of the main function
    '''
    episodes, length_game = 2, 30
    stage, sum_stages = 0, 2
    sigma_reduce = -0.1
    current_reward, argmax = 0, 0
    steps, rate_tracker, sum_rate_tracker = [], [], []
    avg_Q_value_max_List = []
    avg_Current_rewardList = []
    player_1_rate, player_2_rate = [], []
    env = RPS_environment()
    dqn_agent = DDQN(env=env)
    # ------------------------------------------ start the game -----------------------------------------
    print('STARTING THE GAME with %s episodes each with %s moves' % (episodes, length_game), '\n')
    for episode in range(episodes):
        # first reset and get initial state in keras shape
        cur_state = env.restoration().reshape(1, env.state.shape[1])
        current_reward = 0

        for step in range(length_game):
            # AI agent take one action
            action = dqn_agent.act(cur_state, step)
            # play the one move and see how the environment reacts to it
            new_state, reward, done, info = env.round_step(action, step + 1, stage)
            current_reward += reward
            # record the play into memory pool
            dqn_agent.remember_memory(cur_state, action, reward, new_state, done)
            # perform Q-learning from using |"experience replay": learn from random samples in memory
            dqn_agent.DeepMind_backup()
            # apply tranfer learning from actions model to the target model.
            dqn_agent.target_weight_update()
            # update the current state with environment new state
            cur_state = new_state
            if done:  break
        # -------------------------------- INSTRUMENTAL AND PLOTTING -------------------------------------------
        # the instrumental are performed at the end of each episode
        # store epsiode #, winr rate, tie rate, lost rate, etc. etc.
        # ------------------------------------------------------------------------------------------------------
        rate_tracker.append([episode + 1, env.current_WinRate, env.current_TieRate, env.current_LostRate])
        env.last_WinRate += env.current_WinRate
        env.last_TieRate += env.current_TieRate
        env.last_LostRate += env.current_LostRate
        sum_rate_tracker.append([episode + 1,env.last_WinRate / (episode + 1), env.last_TieRate / (episode + 1), env.last_LostRate / (episode + 1), ])

        if True:
            print('EPISODE ', episode + 1),

        if True:
            player_1_rate.append([env.p1_statistic[0] / length_game, env.p1_statistic[1] / length_game, env.p1_statistic[2] / length_game])
            player_2_rate.append([env.p2_statistic[0] / length_game, env.p2_statistic[1] / length_game, env.p2_statistic[2] / length_game])
            print(' P1 rock rate: %.2f paper rate: %.2f scissors rate: %.2f' % (
                player_1_rate[-1][0], player_1_rate[-1][1], player_1_rate[-1][2]))
            print(' P2 rock rate: %.2f paper rate: %.2f scissors rate: %.2f' % (
                player_2_rate[-1][0], player_2_rate[-1][1], player_2_rate[-1][2]))
            env.p1_statistic, env.p2_statistic = [0, 0, 0], [0, 0, 0]

        # summarize Qmax from action model and reward
        avg_Q_max_value = sum(dqn_agent.Qmax) / length_game  # from action model
        avg_Q_value_max_List.append(avg_Q_max_value)
        avgCurrnet_reward = current_reward / length_game
        avg_Current_rewardList.append(avgCurrnet_reward)
        if True:
            print(' Avg reward: %.2f Avg Qmax: %.2f' % (avgCurrnet_reward, avg_Q_max_value))
        dqn_agent.Qmax = []  # reset for next episode


if __name__ == "__main__":
    main()