import nn_layers.vp_layers as vp
import matplotlib.pyplot as plt
import torch.nn as nn
import torch
import pickle
import os
from classifier_models.utils import train_classifier, test_classifier, return_model_accuracy
from datasets import load_dataset
import logging
from torch.utils.tensorboard import SummaryWriter
import scipy.io as sio

class UnknownLayer(Exception):
    pass

def hook_fn(m, i, o):
    print(i)

def add_hook(net):
    hook_handles = []
    for m in net.modules():
        handle = m.register_forward_hook(hook_fn)
        hook_handles.append(handle)
    return hook_handles

def load_weights(name, path):
    model_path = os.path.join(path, f'{name}.dat')
    with open(model_path, 'rb') as model:
        return pickle.load(model)

def rebuild_network(params):
    modules = []
    for key, value in params['VPNet_params'].items():
        layername = key.split('_', 1)[0]
        if 'LinearLayer' == layername:
            out_features, in_features = value[0].shape
            linearlayer = nn.Linear(in_features, out_features)
            linearlayer.weight.data = torch.transpose(torch.tensor(value[0], dtype=torch.float32), -1, -2)
            linearlayer.bias.data = torch.tensor(value[1], dtype=torch.float32)
            modules.append(linearlayer)
        elif 'ReLU' == layername:
            modules.append(nn.ReLU())
        elif 'VPLayer' == layername:
            vp_layer = vp.vp_layer(vp.ada_hermite, params['n_in'], params['VP'], len(value), device='cpu', penalty=params['penalty'], init=value)
            modules.append(vp_layer)
        else:
            raise UnknownLayer
    modules.append(nn.Softmax(dim=-1))
    return nn.Sequential(*modules)

def test_signal(n):
    mat = sio.loadmat('C:/Users/Kovács Ottó/Documents/GitHub/SVPNN/NativeVPNet/data/ecg_test.mat')
    test_x = torch.tensor(mat['samples'], dtype=torch.float32)
    test_y = torch.tensor(mat['labels'], dtype=torch.float32)
    return test_x[n].unsqueeze(0).unsqueeze(0), test_y[n].unsqueeze(0).unsqueeze(0)

'''Checking the projection.'''
path = 'C:/Users/Kovács Ottó/Documents/GitHub/SVPNN/tests/manual/models'
name = 'pretrained_model'
params = load_weights(name, path)
classifier = rebuild_network(params)
logging.basicConfig(filename='pretrained_VP.log', level=logging.INFO)
writer = SummaryWriter(log_dir=f'{path}')

# Evaluate the model on a single test sigal
signal, target = test_signal(0)
#add_hook(classifier)
pred = classifier(signal)
criterion = nn.BCELoss()
loss = criterion(pred, target)

# Evaluate test accuracy
batch_size = 512
data_path = os.path.join(os.getcwd(), 'data', 'ECG')
train_loader, test_loader = load_dataset_ann.load_ecg_real(batch_size, data_path)
accuracy = return_model_accuracy(classifier, test_loader, device='cpu', logging=logging, writer=writer, n_steps=0)
print('Test accuracy:', accuracy.item())

