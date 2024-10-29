import os
import torch
from torch.autograd import Variable
import numpy as np
from collections import defaultdict
import copy
import torch.nn as nn

def nested_dict():
    return defaultdict(nested_dict)

def remove_one_length_word(words):
    return [w for w in words if len(np.unique(list(w))) > 1]

def to_var(x, volatile=False, use_cuda = True):
    if torch.cuda.is_available() and use_cuda:
        if isinstance(x, torch.Tensor):
            x = x.to("cuda:0", non_blocking=True)
        elif isinstance(x, dict):
            for key in x.keys():
                if isinstance(x[key], torch.Tensor):
                    x[key] = x[key].to("cuda:0", non_blocking=True)
    else:
        if isinstance(x, torch.Tensor):
            x = x.cpu()
        elif isinstance(x, dict):
            for key in x.keys():
                if isinstance(x[key], torch.Tensor):
                    x[key] = x[key].cpu()
    return Variable(x, volatile=volatile) if isinstance(x, torch.Tensor) else x

def mkdir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def return_numpy(variable, cpu=1):
    if isinstance(variable, torch.Tensor):
        if cpu:
            return variable.cpu().data.numpy()
        else:
            return variable.data.numpy()
    elif isinstance(variable, np.ndarray):
        return variable
    else:
        return variable

def return_tensor(variable, use_cuda = True):
    if isinstance(variable, torch.Tensor):
        return to_var(variable, use_cuda = use_cuda)
    elif isinstance(variable, np.ndarray):
        return to_var(torch.tensor(variable).float(), use_cuda = use_cuda)
    else:
        return variable
    
def dataloader_to_numpy(dataloader):
    # Create a dictionary to store each tensor
    tensor_dict = {}
    
    # Iterate over the DataLoader
    for batch in dataloader:
        # Each batch is a tuple of tensors
        for i, tensor in enumerate(batch):
            # Convert tensor to numpy array
            np_tensor = tensor.numpy()
            
            # Add to dictionary
            if i not in tensor_dict:
                tensor_dict[i] = []
            tensor_dict[i].append(np_tensor)
    
    # Concatenate the numpy arrays for each tensor
    for i in tensor_dict:
        tensor_dict[i] = np.concatenate(tensor_dict[i], axis=0)
        
    return tensor_dict

def serialize_ndarray(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")

def rsqueeze(arr):
    while np.any(np.array(arr.shape) == 1):
        arr = np.squeeze(arr)
    return arr

def get_last_value(nested_structure):
    """Get the last value from a nested structure."""
    while hasattr(nested_structure, '__getitem__'):
        try:
            nested_structure = nested_structure[0]
        except (TypeError, IndexError, KeyError):
            break
    return nested_structure

def get_value_at_layer(nested_structure, layer):
    """Get the value at a specific nested layer."""
    current_layer = 0
    while hasattr(nested_structure, '__getitem__'):
        if current_layer == layer:
            return nested_structure
        try:
            nested_structure = nested_structure[0]
        except (TypeError, IndexError, KeyError):
            break
        current_layer += 1
    return None

def clone(layer, N):
    return nn.ModuleList([copy.deepcopy(layer) for _ in range(N)])