from __future__ import division
import random
import itertools

beat = {'R': 'P', 'P': 'S', 'S': 'R'}
class MarkovChain():

    def __init__(self, type, beat, level, memory, score=0, score_memory=0.9):
        self.type = type
        self.matrix = self.create_matrix(beat, level, memory)
        self.memory = memory
        self.level = level
        self.beat = beat
        self.score = score
        self.score_memory = score_memory
        self.prediction = ''
        self.name = 'level: {}, memory: {}'.format(self.level, self.memory)
        self.last_updated_key = ''

    @staticmethod
    # The @staticmethod decorator can be called from an uninstantiated class object,
    # although in this case there is no cls parameter passed to its method.
    def create_matrix(beat, level, memory):
        def create_keys(beat, level):
            keys = list(beat)
            if level > 1:
                for i in range(level - 1):
                    key_len = len(keys)
                    # itertools.product(arg*,repeat=1): cartesian product of input iterables
                    # '*'.join(beat) means add * to every interval of string 'beat'
                    for i in itertools.product(keys, ''.join(beat)):
                        keys.append(''.join(i))
                    keys = keys[key_len:]
            # 'keys' is 'level-1' rounds combination of all results sequence
            return keys

        # according to call self function to create the keys
        keys = create_keys(beat, level)
        # this matrix is in form of 3 to the power 'level-1' rows of {'R': , 'P': , 'S': }
        matrix = {}
        for key in keys:
            matrix[key] = {'R': 1 / (1 - memory) / 3,
                           'P': 1 / (1 - memory) / 3,
                           'S': 1 / (1 - memory) / 3}
        # RPS's parameter, depending on memory choosing and the keys update
        return matrix

    def update_matrix(self, key_stored, response_choice):

        for key in self.matrix[key_stored]:
            self.matrix[key_stored][key] = self.memory * self.matrix[key_stored][key]

        # use the stored keys to change weights in matrix
        self.matrix[key_stored][response_choice] += 1
        self.last_updated_key = key_stored
        # print(self.matrix)

    def update_score(self, input_choice, output_choice):

        # add or minus rewards depending on results
        if self.beat[output_choice] == input_choice:
            self.score = self.score * self.score_memory - 1
        elif output_choice == input_choice:
            self.score = self.score * self.score_memory
        else:
            self.score = self.score * self.score_memory + 1

    def predict(self, key_current_value):

        # get the latest probabilities
        probabilities = self.matrix[key_current_value]
        # print(probabilities)

        # judge if the matrix is updated with difference, no difference means no reference
        if max(probabilities.values()) == min(probabilities.values()):
            self.prediction = random.choice(list(beat.keys()))
        else:
            self.prediction = max([(i[1], i[0]) for i in probabilities.items()])[1]

        # choosing method according to rival of self
        if self.type == 'input_oriented':
            return self.prediction
        elif self.type == 'output_oriented':
            return self.beat[self.prediction]


class Ensembler():

    def __init__(self, type, beat, min_score=-10, score=0, score_memory=0.9):
        self.type = type
        self.matrix = {i: 0 for i in beat}
        self.beat = beat
        self.min_score = min_score
        self.score = score
        self.score_memory = score_memory
        self.prediction = ''

    def update_score(self, input_choice, output_choice):

        # add or minus rewards depending on results
        if self.beat[output_choice] == input_choice:
            self.score = self.score * self.score_memory - 1
        elif output_choice == input_choice:
            self.score = self.score * self.score_memory
        else:
            self.score = self.score * self.score_memory + 1

    def update_matrix(self, prediction_dictionary, prediction_score):

        # update the normal predicton matrix
        norm_dictionary = {key: prediction_dictionary[key] / sum(prediction_dictionary.values()) for key in
                           prediction_dictionary}
        for key in self.matrix:
            if prediction_score >= self.min_score:
                self.matrix[key] = self.matrix[key] + prediction_score * norm_dictionary[key]

    def predict(self):

        # judge if the matrix is updated with difference, no difference means no reference
        if max(self.matrix.values()) == min(self.matrix.values()):
            self.prediction = random.choice(list(beat.keys()))
        else:
            self.prediction = max([(i[1], i[0]) for i in self.matrix.items()])[1]
        return self.prediction


class HistoryCollection():

    # function used to collect historical data and archive it
    def __init__(self):
        self.history = ''

    def history_collector(self, input_act, output_act):
        self.history = self.history + input_act
        self.history = self.history + output_act
        # Preserve the latest 10 results to reduce the time influence
        if len(self.history) > 10:
            self.history = self.history[-10:]

    def create_keys(self, level):
        return self.history[-level:]

    def create_keys_hist(self, level):
        # input and output are stored in history as pairs, key preserves the 'level' rounds of pairs
        key_history = self.history[-level - 2:-2]
        input_latest = self.history[-2]
        output_latest = self.history[-1]
        return key_history, input_latest, output_latest


'''Initialization for main function'''
game_round = 0
win_round = 0
lost_round = 0
# first round use random method
output = random.choice(list(beat.keys()))
# first to initialize the history dict
history = HistoryCollection()
# initialize memory, level, and threshold for trigerring Ensemble model
memory = [0.5, 0.6, 0.7, 0.8, 0.9, 0.93, 0.95, 0.97, 0.99]
level = [1, 2, 3, 4]
ensemble_min_score = [5]
# combine models in dict for every-time selecting best model
models_inp = [MarkovChain('input_oriented', beat, i[0], i[1]) for i in itertools.product(level, memory)]
models_out = [MarkovChain('output_oriented', beat, i[0], i[1]) for i in itertools.product(level, memory)]
models_ens = [Ensembler('ensemble', beat, i) for i in ensemble_min_score]
models = models_inp + models_out + models_ens

'''Main function'''
for i in range (50):
    print('The {0} round of game'.format(i+1))
    input_1 = input('Enter your RPS choice:')
    if input_1 in ['R', 'P', 'S']:
        if len(history.history) == 10:
            # when runs to 5 rounds, call history_collector to preserve 10 latest results
            history.history_collector(input_1, output)
            max_score = 0

            for model in models:

                if model.type in ('input_oriented', 'output_oriented'):
                    key_hist, inp_latest, out_latest = history.create_keys_hist(model.level)
                    key_curr = history.create_keys(model.level)

                if model.prediction != '':
                    model.update_score(input_1, beat[model.prediction])

                if model.type == 'input_oriented':
                    model.update_matrix(key_hist, inp_latest)

                elif model.type == 'output_oriented':
                    model.update_matrix(key_hist, out_latest)

                elif model.type == 'ensemble':
                    for mod in models:
                        if mod.type in ('input_oriented', 'output_oriented'):
                            model.update_matrix(mod.matrix[mod.last_updated_key], model.score)

                if model.type in ('input_oriented', 'output_oriented'):
                    predicted_input = model.predict(key_curr)
                elif model.type == 'ensemble':
                    predicted_input = model.predict()

                if model.score > max_score:
                    best_model = model
                    max_score = model.score
                    output = beat[predicted_input]

            if max_score < 1:
                output = random.choice(list(beat.keys()))

        else:
            output = random.choice(list(beat.keys()))
            history.history_collector(input_1, output)

        game_round += 1
        if input_1 == output:
            win_round = win_round
        elif input_1 == 'R' and output == 'S':
            win_round += 1
        elif input_1 == 'S' and output == 'P':
            win_round += 1
        elif input_1 == 'P' and output == 'R':
            win_round += 1
        elif input_1 == 'P' and output == 'S':
            lost_round += 1
        elif input_1 == 'S' and output == 'R':
            lost_round += 1
        elif input_1 == 'R' and output == 'P':
            lost_round += 1
        win_rate = (win_round / game_round) * 100
        lost_rate = (lost_round / game_round) * 100
        print('The machine choice is {0}:'.format(output))
        print('Your win rate is {0}\n'.format(win_rate))
    else:
        print('Please enter the right choice R/P/S !\n')

print('Good Game!')
if win_rate > lost_rate:
    print('Congratulations! You Win!')
if win_rate == lost_rate:
    print('What a tough game! You make it even with Markov Chain!')
if win_rate < lost_rate:
    print('Sorry to say, you are loser.')
