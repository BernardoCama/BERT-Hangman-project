import os 
import sys
import numpy as np
import torch
import platform
import time

# Retrieve the path of the main.py file
cwd = os.path.split(os.path.abspath(__file__))[0]
# Import Classes and DB directories
DB_DIR =  os.path.join(cwd, 'DB')
CLASSES_DIR = os.path.join(cwd, 'Classes')
# Add the directories to the system path
sys.path.append(os.path.dirname(CLASSES_DIR))
sys.path.append(os.path.dirname(cwd))

# Import Classes
from Classes.utils.utils import mkdir
from Classes.solver.solver import Solver
from Classes.dataset.dataset import DatasetWords

# Create the model checkpoint directory
MODEL_DIR = os.path.join(CLASSES_DIR, 'model')
saved_models_dir = os.path.join(MODEL_DIR, 'Saved_models')
output_results_dir = os.path.join(MODEL_DIR, 'Output_results')
mkdir(saved_models_dir)
mkdir(output_results_dir)

# Setting parameters
params = {'train':0, 'test':0}

# Folder parameters
params.update({'cwd':cwd, 'DB_DIR':DB_DIR, 'CLASSES_DIR':CLASSES_DIR, 'MODEL_DIR':MODEL_DIR, 'saved_models_dir':saved_models_dir, 'output_results_dir':output_results_dir})

# Training parameters
params.update({'epochs':1000, 'batch_size':512, 'learning_rate':0, 'lr_scaling_factor':2, 'warmup_step':4000, 'seed':int(time.time()), 'train_log_step':100, 'val_log_step':700})

# Model parameters                                                                             
params.update({'load_model':1, 'save_model':1, 'load_results':1, 'save_results':1, 'hidden_size':256, 'n_encoders':4, 'n_heads':4, 'd_ff':1024})

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


###########################################################################################
###########################################################################################
# IMPORTING DATASETS - SOLVER(MODEL)
# Dataset
dataset = DatasetWords(params)

# Solver
solver = Solver(params)

# Profiler of complexity
# solver.model.print_MACs_FLOPs()

# Print the results
if params['load_results']:
    solver.load_results()
    # Plot the metrics
    solver.plot_metrics()

###########################################################################################
###########################################################################################
# TRAINING AND TESTING
if params['train']:
    train_loader = dataset.create_training_dataset()
    solver.train(train_loader, dataset=dataset) 
if params['test']:
    dataset = DatasetWords(params)
    test_loader = dataset.create_testing_dataset()
    solver.test(test_loader)
