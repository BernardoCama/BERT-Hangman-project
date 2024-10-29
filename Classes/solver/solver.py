import os
import torch
import torch.nn as nn
import time
import numpy as np
import matplotlib.pyplot as plt

# Import Classes
from Classes.dataset.dataset import Vocabulary  
from Classes.model.model import Bert
from Classes.model.non_ML_model import guess_vowel
from Classes.optimizer.optimizer import NoamOpt
from Classes.utils.utils import nested_dict

class Solver:  
    
    def __init__(self, params):
        self.params = params
        self.vocabulary = Vocabulary()

        # Define the model
        self.model = Bert(params)
        for param in self.model.parameters():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)

        # To device
        self.model.to(params['device'])

        # Optimizer and loss
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.params['learning_rate'], betas=(0.9, 0.98), eps=1e-9)
        self.model_opt = NoamOpt(self.params['hidden_size'], self.params['lr_scaling_factor'], self.params['warmup_step'], optimizer)
        self.loss_fun = nn.KLDivLoss(reduction='sum')

        # Load model
        if params['load_model']:
            self.load_model()        


    #################################################################
    #################### LOADING AND SAVING #########################
    #################################################################
    def load_model(self):
        '''
        Load the model
        '''
        print('Loading model...')
        self.model = torch.load(os.path.join(self.params['saved_models_dir'], 'model.pth'), map_location=self.params['device'])
        # checkpoint = torch.load(os.path.join(self.params['saved_models_dir'], 'model.pth'), map_location=self.params['device'])
        # self.model.load_state_dict(checkpoint['model_state_dict'])
        # if 'generator_state_dict' in checkpoint:
        #     self.model.generator.load_state_dict(checkpoint['generator_state_dict'])
        # if 'optimizer_state_dict' in checkpoint:
        #     self.model_opt.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        #     self.model_opt._rate = checkpoint['_rate']
        #     self.model_opt._step = checkpoint['_step']
        self.model.eval()
        self.model.generator.eval()
        print('Model loaded')

    def save_model(self):
        '''
        Save the model
        '''
        print('Saving model...')
        torch.save(self.model, os.path.join(self.params['saved_models_dir'], 'model.pth'))
        # torch.save({'model_state_dict': self.model.state_dict(),
        #             'generator_state_dict': self.model.generator.state_dict(),
        #             'optimizer_state_dict': self.model_opt.optimizer.state_dict(),
        #             '_rate': self.model_opt._rate,
        #             '_step': self.model_opt._step}, os.path.join(self.params['saved_models_dir'], 'model.pth'))
        print('Model saved')

    def load_results(self):
        '''
        Load the results
        '''
        print('Loading results...')
        self.train_metrics = torch.load(os.path.join(self.params['output_results_dir'], 'train_metrics.pth'))
        self.val_metrics = torch.load(os.path.join(self.params['output_results_dir'], 'val_metrics.pth'))
        print('Results loaded')

    def plot_metrics(self):
        '''
        Plot the metrics
        '''
        print('Plotting metrics...')
        # Plot the training metrics
        fig = plt.figure()
        train_metrics = [self.train_metrics[epoch][i] for epoch in self.train_metrics for i in self.train_metrics[epoch]]
        # Each metric correspond to self.params['batch_size'] * self.params['train_log_step'] words
        plt.plot(np.arange(0, len(train_metrics) * self.params['batch_size'] * self.params['train_log_step'], self.params['batch_size'] * self.params['train_log_step']), train_metrics)
        plt.title('Train loss', fontsize=18)
        plt.xlabel('Words', fontsize=18)
        plt.ylabel('Loss', fontsize=18)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        # plt.show()
        # Save the plot
        plt.savefig(os.path.join(self.params['output_results_dir'], 'train_loss.png'))
        plt.savefig(os.path.join(self.params['output_results_dir'], 'train_loss.pdf'), bbox_inches='tight') 

        # Plot the validation metrics
        fig = plt.figure()
        val_metrics = [self.val_metrics[epoch][i] for epoch in self.val_metrics for i in self.val_metrics[epoch]]
        # Each metric correspond to self.params['batch_size'] * self.params['val_log_step'] words
        plt.plot(np.arange(0, len(val_metrics) * self.params['batch_size'] * self.params['val_log_step'], self.params['batch_size'] * self.params['val_log_step']), val_metrics)
        plt.title('Val accuracy', fontsize=18)
        plt.xlabel('Words', fontsize=18)
        plt.ylabel('Accuracy', fontsize=18)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        # plt.show()
        # Save the plot
        plt.savefig(os.path.join(self.params['output_results_dir'], 'val_accuracy.png')) 
        plt.savefig(os.path.join(self.params['output_results_dir'], 'val_accuracy.pdf'), bbox_inches='tight') 


    #################################################################
    #################### TRAINING AND TESTING #######################
    #################################################################
    def train(self, train_loader, val_loader = None, dataset = None):
        '''
        Train the model
        '''
        print('Training...')
        # Metrics dictionaries
        train_metrics = nested_dict()
        val_metrics = nested_dict()
        cumulated_words = 0
        # Track the best validation accuracy
        best_val_acc = 0
        for epoch in range(self.params['epochs']):
            total_loss = 0

            if dataset is not None:
                train_loader, val_loader, test_loader = dataset.load_dataset()

            start = time.time()

            for i, batch in enumerate(train_loader):

                self.model.train()
                self.model.generator.train()
                # Zero gradients to avoid accumulation
                self.model_opt.optimizer.zero_grad()

                # Forward pass through the encoder. out.shape = (batch_size, seq_len, hidden_size)
                out = self.model.forward(batch.src, batch.src_mask, batch.not_present)
                # Forward pass through the generator to generate the target distribution
                # generator_mask.shape = (batch_size, vocab_size)
                generator_mask = torch.zeros(self.params['batch_size'], len(self.vocabulary.char2id), device=self.params['device'])
                generator_mask = generator_mask.scatter_(1, batch.src, 1) 
                out = self.model.generator(out, generator_mask.bool() | batch.not_present.bool())

                # DEBUG
                # print('Input word:', self.convert_id_to_char(batch.src[0]))
                # print('Source mask:', batch.src_mask[0, 0, :].bool())
                # print('Target word:', self.convert_id_to_char(batch.tgt_word[0]))
                # print('Target letters:', self.convert_booleanid_to_char(batch.tgt[0]))
                # print('Generator mask:', self.convert_booleanid_to_char(generator_mask[0]))
                # print('Not present:', self.convert_booleanid_to_char(batch.not_present[0]))
                # print('Output (most probable letter):', self.convert_id_to_char(torch.argmax(out[0]).view(1)))
                # self.plot_distribution(out[0].exp())

                # Compute the loss
                loss = self.loss_fun(out, batch.tgt)
                # Compute the loss per sample
                loss = loss / self.params['batch_size']

                # Backpropagation
                loss.backward()
                # Update the parameters
                self.model_opt.step()

                loss = loss.data
                cumulated_words += self.params['batch_size']
                total_loss += loss.item()

                if i % self.params['train_log_step'] == 0 and i > 0:
                    elapsed = time.time() - start
                    print('Epoch: {} | Batch: {} | Loss (per sample): {:.2f} | Cumulative words: {} | Time: {:.2f}'.format(epoch, i, total_loss/self.params['train_log_step'], cumulated_words, elapsed))

                    # Save the metrics
                    train_metrics[epoch][i] = total_loss / self.params['train_log_step']
                    torch.save(train_metrics, os.path.join(self.params['output_results_dir'], 'train_metrics.pth'))

                    start = time.time()                                                                  
                    total_loss = 0

                if i % self.params['val_log_step'] == 0 and i > 0 and val_loader is not None:
                    val_metrics[epoch][i] = self.test(val_loader, dataset)
                    # If the current model is better than the previous ones (for all metrics in each epoch), save it
                    if val_metrics[epoch][i] > best_val_acc:
                        best_val_acc = val_metrics[epoch][i]
                        self.save_model()
                    # Save the metrics
                    torch.save(val_metrics, os.path.join(self.params['output_results_dir'], 'val_metrics.pth'))

            # Save the model at the end of each epoch
            self.save_model()
                    
        print('Training finished')

    def test(self, test_loader, dataset = None):
        '''
        Test the model
        '''
        print('Testing...')
        if dataset is not None:
            test_loader = dataset.load_dataset()[1]
        self.model.eval()
        self.model.generator.eval()
        total_loss = 0
        total_acc = 0
        cumulated_words = 0

        with torch.no_grad():
            for i, batch in enumerate(test_loader):

                # Forward pass through the encoder
                out = self.model.forward(batch.src, batch.src_mask, batch.not_present)
                # Forward pass through the generator to generate the target distribution
                generator_mask = torch.zeros(self.params['batch_size'], len(self.vocabulary.char2id), device=self.params['device'])
                generator_mask = generator_mask.scatter_(1, batch.src, 1)
                out = self.model.generator(out, generator_mask.bool() | batch.not_present.bool())

                # DEBUG
                # print('Input word:', self.convert_id_to_char(batch.src[0]))
                # print('Source mask:', batch.src_mask[0, 0, :].bool())
                # print('Target word:', self.convert_id_to_char(batch.tgt_word[0]))
                # print('Target letters:', self.convert_booleanid_to_char(batch.tgt[0]))
                # print('Generator mask:', self.convert_booleanid_to_char(generator_mask[0]))
                # print('Not present:', self.convert_booleanid_to_char(batch.not_present[0]))
                # print('Output (most probable letter):', self.convert_id_to_char(torch.argmax(out[0]).view(1)))
                # self.plot_distribution(out[0].exp())
                # self.plot_distribution(batch.tgt[0])

                # Compute the loss
                loss = self.loss_fun(out, batch.tgt)
                loss = loss / self.params['batch_size']
                loss = loss.data
                total_loss += loss.item()

                # Compute the accuracy
                most_prob_letter = torch.argmax(out, dim=1)
                most_prob_mask = torch.zeros(batch.tgt.shape, device=self.params['device'])
                most_prob_mask = most_prob_mask.scatter_(1, most_prob_letter.view(-1, 1), 1)
                batch_guess = (batch.tgt * most_prob_mask).sum(dim=1)
                # For each word, we gussed right if the most probable letter is in the target distribution
                total_acc += (batch_guess != 0).sum().item()

                cumulated_words += self.params['batch_size']
                
        acc = total_acc / cumulated_words
        loss = total_loss / cumulated_words
        print('Loss (per sample): {:.6f} | Accuracy: {:.2f}'.format(loss, acc))
        print('Testing finished')
        
        return acc
    

    #################################################################
    ######################### GUESSING ##############################
    #################################################################
    def get_best_first_char(self, question, guessed_letters, words):
        return guess_vowel(len(question), guessed_letters, words)

    def guess(self, question, guessed_letters, words):
        if question.count(self.params['mask_char']) == len(question):
            pred = self.get_best_first_char(question, guessed_letters, words)
            guessed_letters.append(pred)
            return pred

        guessed = [self.vocabulary.char2id[l] for l in (guessed_letters)]
        # Compute the wrongly guessed letters (letters in the guessed_letters list that are not in the question)
        wrongly_guessed = [l for l in guessed if l not in [self.vocabulary.char2id[c] for c in question]]
        wrongly_guessed = torch.tensor(wrongly_guessed, device=self.params['device']).view(1, -1)
        not_present = torch.zeros(1, len(self.vocabulary.char2id), device=self.params['device'], dtype=torch.int64)
        # Remove mask_char and _ from the not_present tensor
        not_present[:, self.vocabulary.char2id[self.params['mask_char']]] = 1
        not_present[:, self.vocabulary.char2id['_']] = 1
        if wrongly_guessed.shape[1] > 0:
            not_present = not_present.scatter_(1, wrongly_guessed, 1)
 
        p = self.get_most_prob(question, not_present)
        p[guessed] = -np.inf

        pred = self.vocabulary.id2char[np.argmax(p)]
        guessed_letters.append(pred)
        return pred
    
    def get_most_prob(self, masked_word, not_present):
        with torch.no_grad():
            src = torch.tensor([[self.vocabulary.char2id[c] for c in masked_word]], device=self.params['device'])
            src_mask = ((src != self.vocabulary.char2id[self.params['mask_char']]) & (src != self.vocabulary.char2id['_'])).unsqueeze(-2)
            out = self.model.forward(src, src_mask, not_present)
            generator_mask = torch.zeros(src.shape[0], len(self.vocabulary.char2id), device=self.params['device'])
            generator_mask = generator_mask.scatter_(1, src, 1)
            generator_mask[0, self.vocabulary.char2id[self.params['mask_char']]] = 1
            generator_mask[0, self.vocabulary.char2id['_']] = 1
            out = self.model.generator(out, generator_mask.bool() | not_present.bool())

            # DEBUG
            # print('Input word:', self.convert_id_to_char(src[0]))
            # print('Source mask:', src_mask[0, 0, :].bool())
            # print('Generator mask:', self.convert_booleanid_to_char(generator_mask[0]))
            # print('Not present:', self.convert_booleanid_to_char(not_present[0]))
            # print('Output (most probable letter):', self.convert_id_to_char(torch.argmax(out[0]).view(1)))
            # self.plot_distribution(out[0].exp())
            
            p = out.exp().squeeze(0)
            if self.params['device'].type == 'cuda':
                p = p.cpu()
            return p.detach().numpy()
        
    #################################################################
    ######################### DEBUG #################################
    #################################################################
    def plot_distribution(self, pdf):
        '''
        Plot the distribution of the probabilities
        Input:
            pdf: numpy or tensor array of probabilities of size (vocab_size,)
        '''
        # Convert tensor to numpy
        if torch.is_tensor(pdf):
            pdf = pdf.cpu().detach().numpy()

        fig = plt.figure()
        plt.bar(range(len(pdf)), pdf)
        plt.title('Probability distribution')
        plt.xlabel('Letter')
        plt.ylabel('Probability')
        plt.xticks(range(len(pdf)), self.vocabulary.id2char.values())
        plt.show()

    def convert_id_to_char(self, ids):
        if torch.is_tensor(ids):
            ids = ids.cpu().detach().numpy()
        return ''.join([self.vocabulary.id2char[i] for i in ids])
    
    def convert_booleanid_to_char(self, ids):
        if torch.is_tensor(ids):
            ids = ids.cpu().detach().numpy()
        return ''.join([self.vocabulary.id2char[i] if ids[i] > 0 else '' for i in range(len(ids))])