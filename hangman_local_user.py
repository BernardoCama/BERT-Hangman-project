import numpy as np
import os 
import sys
import torch
import platform
import re
import collections

# Retrieve the path of the main.py file
cwd = os.path.split(os.path.abspath(__file__))[0]
# Import Classes and DB directories
DB_DIR =  os.path.join(cwd, 'DB')
CLASSES_DIR = os.path.join(cwd, 'Classes')
# Add the directories to the system path
sys.path.append(os.path.dirname(CLASSES_DIR))
sys.path.append(os.path.dirname(cwd))

# Import Classes
from Classes.solver.solver import Solver
from Classes.dataset.dataset import DatasetWords

# Create the model checkpoint directory
MODEL_DIR = os.path.join(CLASSES_DIR, 'model')
saved_models_dir = os.path.join(MODEL_DIR, 'Saved_models')
output_results_dir = os.path.join(MODEL_DIR, 'Output_results')

params = {}
# Folder parameters
params.update({'cwd':cwd, 'DB_DIR':DB_DIR, 'CLASSES_DIR':CLASSES_DIR, 'MODEL_DIR':MODEL_DIR, 'saved_models_dir':saved_models_dir, 'output_results_dir':output_results_dir})
# Training parameters
params.update({'epochs':1, 'batch_size':512, 'learning_rate':0, 'lr_scaling_factor':2, 'warmup_step':4000, 'seed':42, 'train_log_step':100, 'val_log_step':700})
# Model parameters
params.update({'load_model':1, 'save_model':0, 'save_results':0, 'hidden_size':256, 'n_encoders':4, 'n_heads':4, 'd_ff':1024})
# Dataset parameters
params.update({'load_dataset':0, 'save_dataset':0, 'valid_on_training_set':0, 'train_ratio':0.8, 'val_ratio':0.1, 'test_ratio':0.1, 'mask_char':'$', 'mask_ratio':0.5})
# Reproducibility
np.random.seed(params['seed'])
torch.manual_seed(params['seed'])
# OS
OS = platform.system()
params.update({'OS':OS})
# Device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
params.update({'device':device})


class Player:
    def __init__(self):
        # Solver and load model
        self.solver = Solver(params)
        # Load training dataset
        self.dataset = DatasetWords(params)
        
        # Load training words
        self.full_dictionary = self.dataset.read_words(os.path.join(params['DB_DIR'], 'words_250000_train.txt'))

        # For dumb guess with default strategy
        self.full_dictionary_common_letter_sorted = collections.Counter("".join(self.full_dictionary)).most_common()
        self.current_dictionary = []

    def guess(self, question):
        # Guess with default strategy
        # return self.guess_default(question)

        # Remove duplicates
        self.guessed_letters = list(set(self.guessed_letters))
        # Guess with trained model
        return self.solver.guess(question, self.guessed_letters, self.full_dictionary)


    def guess_default(self, word): # word input example: "_ p p _ e "
        ###############################################
        # Replace with your own "guess" function here #
        ###############################################

        # clean the word so that we strip away the space characters
        # replace "_" with "." as "." indicates any character in regular expressions
        clean_word = word.replace("_",".")
        
        # find length of passed word
        len_word = len(clean_word)
        
        # grab current dictionary of possible words from self object, initialize new possible words dictionary to empty
        current_dictionary = self.current_dictionary
        new_dictionary = []
        
        # iterate through all of the words in the old plausible dictionary
        for dict_word in current_dictionary:
            # continue if the word is not of the appropriate length
            if len(dict_word) != len_word:
                continue
                
            # if dictionary word is a possible match then add it to the current dictionary
            if re.match(clean_word,dict_word):
                new_dictionary.append(dict_word)
        
        # overwrite old possible words dictionary with updated version
        self.current_dictionary = new_dictionary
        
        
        # count occurrence of all characters in possible word matches
        full_dict_string = "".join(new_dictionary)
        
        c = collections.Counter(full_dict_string)
        sorted_letter_count = c.most_common()                   
        
        guess_letter = '!'
        
        # return most frequently occurring letter in all possible words that hasn't been guessed yet
        for letter, instance_count in sorted_letter_count:
            if letter not in self.guessed_letters:
                guess_letter = letter
                break
            
        # if no word matches in training dictionary, default back to ordering of full dictionary
        if guess_letter == '!':
            sorted_letter_count = self.full_dictionary_common_letter_sorted
            for letter, instance_count in sorted_letter_count:
                if letter not in self.guessed_letters:
                    guess_letter = letter
                    break            
        self.guessed_letters.append(guess_letter)
        return guess_letter
    
    def new_game(self):
        self.guessed_letters = []

def read_test_words():
    with open(os.path.join(params['DB_DIR'], 'words_alpha_test_unique.txt'), 'r') as f:
        words = f.read().split('\n')
    return words[:-1]

def data_iter(words):
    for word in words:
        if ',' in word:
            _, answer = word.split(',')
        else:
            answer = word
        question = params['mask_char'] * len(answer)
        yield question, answer

def run():

    player = Player()

    test_words = read_test_words()
    np.random.shuffle(test_words)

    # test_words = test_words[:1000]
    qa_pair = data_iter(test_words)
    num_correct_letters = 0
    num_total_letters = 0
    success = total = 0
    success_rate = 0
    print(f"Total Game Number: {len(test_words)}")
    for question, answer in qa_pair:
        player.new_game()
        tries = 6
        success_rate = 0 if total == 0 else success / total
        print("=" * 20, "Game %d" % (total + 1), '=' * 20, "Success Rate: %.2f" % success_rate)
        print('provided question: ', " ".join(question))
        while params['mask_char'] in question and tries > 0:
            num_total_letters += 1
            guess = player.guess(question)
            question_lst = []
            for q_l, a_l in zip(question, answer):
                if q_l == params['mask_char']:
                    if a_l == guess:
                        question_lst.append(a_l)
                    else:
                        question_lst.append(q_l)
                else:
                    question_lst.append(q_l)
            question = "".join(question_lst)
            if guess not in answer:
                tries -= 1
            else:
                num_correct_letters += 1
            print("provided question: ", " ".join(question), "your guess: %s" % guess, "left tries: %d" % tries, 'answer: %s' % answer)
            print('num_correct_letters: %d, num_total_letters: %d, rate: %.4f' % (num_correct_letters, num_total_letters, num_correct_letters / num_total_letters))


        if params['mask_char'] not in question:
            success += 1
            print("Success!")
        total += 1

    print(f"{success} success out of {total} tries, rate: {success / total:.4f}")


if __name__ == "__main__":
    run()

    
