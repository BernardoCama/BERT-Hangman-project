import os
import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
import string
import time

# Import Classes
from Classes.utils.utils import remove_one_length_word

class DatasetWords:  
    
    def __init__(self, params):
        self.params = params
        self.vocabulary = Vocabulary()

    def load_dataset(self):
        '''
        Load the dataset and return the train, validation and test loaders
        '''
        # Load dataset
        if self.params['load_dataset']:
            print('Loading dataset...')
            train_loader = torch.load(os.path.join(self.params['DB_DIR'], 'train_loader.pth'))
            val_loader = torch.load(os.path.join(self.params['DB_DIR'], 'val_loader.pth'))
            test_loader = torch.load(os.path.join(self.params['DB_DIR'], 'test_loader.pth'))
            print('Dataset loaded')
        else:
            train_loader, val_loader, test_loader = self.create_dataset()
        return train_loader, val_loader, test_loader
    
    def read_words(self, file_path, contains_masked_word=False):
        with open(file_path, 'r') as f:
            words = f.read().split('\n')
        if contains_masked_word:
            words = [word.split(',')[1] for word in words if len(word.split(',')) == 2]
        return words
    
    def create_dataset(self):
        '''
        Create the dataset and return the train, validation and test loaders
        '''
        print('Creating dataset...')

        if self.params['valid_on_training_set']:

            print('Validation on training set')
            # Read the words from the file
            words = self.read_words(os.path.join(self.params['DB_DIR'], 'words_250000_train.txt'))
            # Remove one length words
            words = remove_one_length_word(words)
            self.words_size = len(words)
            np.random.shuffle(words)

            train_words = words[:int(self.words_size * self.params['train_ratio'])]
            val_words = words[int(self.words_size * self.params['train_ratio']):int(self.words_size * (self.params['train_ratio'] + self.params['val_ratio']))]
            test_words = words[int(self.words_size * (self.params['train_ratio'] + self.params['val_ratio'])):]

            train_dataset = self.create_masks(train_words, self.vocabulary)
            val_dataset = self.create_masks(val_words, self.vocabulary)
            test_dataset = self.create_masks(test_words, self.vocabulary)
            # Storing direcly multiple mask ratio
            # train_dataset = self.create_masks_different_ratio(train_words)
            # val_dataset = self.create_masks_different_ratio(val_words)
            # test_dataset = self.create_masks_different_ratio(test_words)

            train_loader = self.create_loader(train_dataset, shuffle=False)
            val_loader = self.create_loader(val_dataset, shuffle=False)
            test_loader = self.create_loader(test_dataset, shuffle=False)

        else:
            print('Validation on independent set')

            # Read the training words from the file
            words = self.read_words(os.path.join(self.params['DB_DIR'], 'words_250000_train.txt'))
            # Remove one length words
            train_words = remove_one_length_word(words)
            np.random.shuffle(train_words)
            self.words_size = len(train_words)

            # Read the testing words from the file
            test_words = self.read_words(os.path.join(self.params['DB_DIR'], 'words_alpha_test_unique.txt'))
            # Remove one length words
            test_words = remove_one_length_word(test_words)
            np.random.shuffle(test_words)
            self.test_words_size = len(test_words)
            val_words = test_words[:int(self.test_words_size * self.params['test_ratio'])]

            train_dataset = self.create_masks(train_words, self.vocabulary)
            val_dataset = self.create_masks(val_words, self.vocabulary)
            test_dataset = self.create_masks(test_words, self.vocabulary)
            # Storing direcly multiple mask ratio
            # train_dataset = self.create_masks_different_ratio(train_words)
            # val_dataset = self.create_masks_different_ratio(val_words)
            # test_dataset = self.create_masks_different_ratio(test_words)

            train_loader = self.create_loader(train_dataset, shuffle=False)
            val_loader = self.create_loader(val_dataset, shuffle=False)
            test_loader = self.create_loader(test_dataset, shuffle=False)

        if self.params['save_dataset']:
            print('Saving dataset...')
            torch.save(train_loader, os.path.join(self.params['DB_DIR'], 'train_loader.pth'))
            torch.save(val_loader, os.path.join(self.params['DB_DIR'], 'val_loader.pth'))
            torch.save(test_loader, os.path.join(self.params['DB_DIR'], 'test_loader.pth'))
        
        return train_loader, val_loader, test_loader

    def create_training_dataset(self):
        '''
        Create the training dataset and return the train loader
        '''
        print('Creating training dataset...')
        # Read the words from the file
        words = self.read_words(os.path.join(self.params['DB_DIR'], 'words_250000_train.txt'))
        # Remove one length words
        words = remove_one_length_word(words)
        np.random.shuffle(words)
        self.words_size = len(words)

        train_words = words
        train_dataset = self.create_masks(train_words, self.vocabulary)
        train_loader = self.create_loader(train_dataset, shuffle=False)

        return train_loader
    
    def create_testing_dataset(self):
        '''
        Create the testing dataset and return the test loader
        '''
        print('Creating testing dataset...')
        # Read the words from the file
        words = self.read_words(os.path.join(self.params['DB_DIR'], 'words_alpha_test_unique.txt'))
        # Remove one length words
        words = remove_one_length_word(words)
        np.random.shuffle(words)
        self.words_size = len(words)

        test_words = words
        test_dataset = self.create_masks(test_words, self.vocabulary)
        test_loader = self.create_loader(test_dataset, shuffle=False)

        return test_loader

    def create_loader(self, data, shuffle=True):
        '''
        Create a DataLoader from a list of words and their masked versions
        '''
        dataset = CustomDataset(data, self.vocabulary)
        data_loader = DataLoader(
            dataset,
            batch_size=self.params['batch_size'],
            shuffle=shuffle,
            collate_fn=self.collate_fn,
            drop_last=True,
        )
        return data_loader

    def collate_fn(self, batch):
        '''
        Collate function for DataLoader that pads the words and converts the target to a distribution
        '''
        src_batch, tgt_batch, not_present_batch = zip(*batch)
        
        # Pad the words in the batch to the length of the longest word
        src_padded = torch.tensor(self.pad_words(src_batch, self.vocabulary.char2id['_']), device=self.params['device'])
        tgt_padded = torch.tensor(self.pad_words(tgt_batch, self.vocabulary.char2id['_']), device=self.params['device'])
        
        # Use the distribution of the characters in the target word as the target so that the model can predict the correct letters
        tgt_dist = self.convert_target_to_dist(tgt_padded, src_padded == self.vocabulary.char2id[self.params['mask_char']])
        tgt_dist = torch.div(tgt_dist, tgt_dist.sum(dim=1, keepdim=True))

        # Compute the boolean tensor indicating the non-presence of each letter
        # Compute the letters that are not present in the target word
        not_present_batch = torch.tensor(self.pad_words(not_present_batch, self.vocabulary.char2id['_']), device=self.params['device'])
        not_present_letters = self.return_not_present(mask_token=self.vocabulary.char2id[self.params['mask_char']], not_present_letters=not_present_batch)

        batch = Batch(
            src=src_padded, 
            tgt_word=tgt_padded,
            tgt=tgt_dist, 
            not_present=not_present_letters,
            mask_token=self.vocabulary.char2id[self.params['mask_char']], 
            pad_token=self.vocabulary.char2id['_']
        )
        
        return batch
    
    def return_not_present(self, not_present_letters = None, mask_token = '$', present_letters = None):
        '''
        Input:
        - not_present_letters: for sure non-present letters of size batch_size x max_word_length
        - present_letters: for sure present letters of size batch_size x max_word_length
        
        Returns:
        - output: boolean tensor indicating the non presence of each letter of size batch_size x vocabulary_size
        '''
        if not_present_letters is not None:
            output = torch.zeros(not_present_letters.shape[0], len(self.vocabulary.char2id), device=self.params['device'])
            output = output.scatter_(1, not_present_letters, 1)
            return output
        elif present_letters is not None:
            output = torch.ones(present_letters.shape[0], len(self.vocabulary.char2id), device=self.params['device'])
            output = output.scatter_(1, present_letters, 0)
            return output
        return None
    
    def convert_target_to_dist(self, target, mask):
        '''
        Converts the target tensor to a distribution tensor.
        mask is a tensor of the same shape as target, with 1s where the target is valid and 0s where it is not.
        '''
        dist_mask = torch.zeros(target.shape[0], len(self.vocabulary.char2id), device=self.params['device'], dtype=torch.long)
        dist_mask = dist_mask.scatter_(1, target * mask, 1)

        target_numpy = (target * mask).cpu().numpy()
        # Add an extra column to ensure that np.bincount generates an output array of the expected length
        extra_col = np.ones((target.shape[0], 1), dtype=target_numpy.dtype) * (self.vocabulary.char2id['z'] + 1)
        target_numpy = np.hstack((target_numpy, extra_col))
        target_np_dist = np.apply_along_axis(np.bincount, 1, target_numpy)[:, :-1]
        if self.params['device'].type == 'cuda':
            target_dist = torch.from_numpy(target_np_dist).cuda()
        else:
            target_dist = torch.from_numpy(target_np_dist)
        target_dist[:, 0] = 0

        return (target_dist * dist_mask).long()
    

    def create_masks(self, words, vocabulary):
        '''
        Masks all occurrences of selected unique characters. 
        If a character is selected, every instance of that character in the word is masked.
        '''
        result = []
        # Set the seed as the current time
        np.random.seed(int(time.time()))
        for i in range(len(words)):

            # Standard masked word
            word = list(words[i])
            word_chars = np.unique(word)
            word_len = len(word_chars)

            # Fixed mask ratio
            # number_mask = max(int(word_len * self.params['mask_ratio']), 1)
            # Random mask ratio
            number_mask = np.random.randint(1, word_len)

            indices = np.random.choice(np.arange(word_len), size=number_mask, replace=False)
            letters = word_chars[indices]
            for l in letters:
                for j, w in enumerate(word):
                    if w == l:
                        word[j] = self.params['mask_char']
            masked_word = ''.join(word)
            # Select from the alphabet the letters (randomly in [0, 5]) that are not present in the word
            # Uniform distribution    
            # not_present_letters = list(set(vocabulary.char_list) - set(words[i]))[:random.randint(0, 5)]
            # Exponential distribution
            not_present_letters = list(set(vocabulary.char_list) - set(words[i]))[:min(int(np.random.exponential(scale=1)), 5)]
            # Add the mask letter and "_" to the not_present_letters
            not_present_letters.append(self.params['mask_char'])
            not_present_letters.append('_')
            result.append([masked_word, words[i], not_present_letters])

            # Keep just one vowel and mask the rest
            word = list(words[i])
            word_chars = np.unique(word)
            word_len = len(word_chars)
            vowels = ['a', 'e', 'i', 'o', 'u']
            # Randomize vowels
            vowels = np.random.permutation(vowels)
            vowels_present = [v for v in vowels if v in word_chars]
            if len(vowels_present) > 0:
                for v in vowels_present[:1]:
                    for j, w in enumerate(word):
                        if w != v:
                            word[j] = self.params['mask_char']
                masked_word = ''.join(word)
                not_present_letters = list()
                not_present_letters.append(self.params['mask_char'])
                not_present_letters.append('_')
                result.append([masked_word, words[i], not_present_letters])
        return result
    
    def create_masks_different_ratio(self, words):
        '''
        Masks all occurrences of selected unique characters. 
        If a character is selected, every instance of that character in the word is masked.
        Use different mask ratio for each word.
        '''
        result = []
        # Set the seed as the current time
        np.random.seed(int(time.time()))
        for i in range(len(words)):
            word = list(words[i])
            word_chars = np.unique(word)
            word_len = len(word_chars)
            for number_mask in range(1, word_len):
                word = list(words[i])
                indices = np.random.choice(np.arange(word_len), size=number_mask, replace=False)
                letters = word_chars[indices]
                for l in letters:
                    for j, w in enumerate(word):
                        if w == l:
                            word[j] = self.params['mask_char']
                masked_word = ''.join(word)
                # Uniform distribution    
                # not_present_letters = list(set(vocabulary.char_list) - set(words[i]))[:random.randint(0, 5)]
                # Exponential distribution
                not_present_letters = list(set(self.vocabulary.char_list) - set(words[i]))[:min(int(np.random.exponential(scale=1)), 5)]
                not_present_letters.append(self.params['mask_char'])
                not_present_letters.append('_')
                result.append([masked_word, words[i], not_present_letters])
        return result

    def pad_words(self, words, pad_token):
        """
        Pads a list of sentences to the length of the longest sentence in the batch.

        :param sentences: list[list[int]]
            A list of sentences, where each sentence is represented as a list of word IDs.
        :param pad_token: int
            The token used for padding shorter sentences.
        :returns: list[list[int]]
            A list of sentences padded to the same length, with the shape (batch_size, max_sentence_length).
        """
        word_lens = [len(word) for word in words]
        max_len = max(word_lens)
        # Pad all sentences (words) to the length of the longest sentence
        words_padded = [word + [pad_token] * (max_len - word_lens[i]) for i, word in enumerate(words)]

        return words_padded

# Define the CustomDataset class
class CustomDataset(Dataset):
    '''
    Custom dataset class for the tokenized words
    '''
    def __init__(self, data, vocabulary):
        self.data = data
        self.vocabulary = vocabulary

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        src_word, tgt_word, not_present_letters = self.data[idx]
        # Convert words to lists of character IDs (tokens)
        src = [self.vocabulary.char2id[c] for c in src_word]
        tgt = [self.vocabulary.char2id[c] for c in tgt_word]
        not_present = [self.vocabulary.char2id[c] for c in not_present_letters]
        return src, tgt, not_present

# Define the Vocabulary class
class Vocabulary:
    def __init__(self, mask_char = '$'):
        self.mask_char = mask_char
        self.char2id = dict()
        self.char2id['_'] = 0
        self.char2id[self.mask_char] = 1
        self.char_list = string.ascii_lowercase
        for i, c in enumerate(self.char_list):
            self.char2id[c] = len(self.char2id)
        self.id2char = {v: k for k, v in self.char2id.items()}
        # Store the size of the vocabulary
        self.vocab_size = len(self.char2id)

# Define the Batch class
class Batch:
    def __init__(self, src, tgt, tgt_word, mask_token=None, pad_token=0, not_present=None):
        self.src = src
        self.tgt_word = tgt_word
        self.tgt = tgt
        self.src_mask = ((src != mask_token) & (src != pad_token)).unsqueeze(-2)
        self.randomly_mask = (src != mask_token)
        self.pad_token = pad_token
        self.not_present = not_present
