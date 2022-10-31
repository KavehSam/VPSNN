import torch
from classifier_models.utils import parse

SEED_NUM = 2
dtype = torch.float32
device = None
dataset_name = None
data_path = None
batch_size = 0
n_steps = 0
lr = 0
lr_decay = 0
load_best_state = None
vp_latent_dim = None
num_classes = None
L2penalty = None
init_vp = None
epochs_cls = None
classifier_name = None


def get(n_config):
    global device, dataset_name, data_path, batch_size, n_steps, lr, lr_decay, load_best_state, vp_latent_dim,\
    num_classes, L2penalty, init_vp, epochs_cls, classifier_name
    
    #device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    device = 'cpu'
    params = parse('../network_configs/ECG_VPNN.yaml')
    dataset_name = params['Network']['dataset']
    data_path = params['Network']['data_path']
    batch_size = params['Network']['batch_size']
    n_steps = params['Network']['n_steps']
    lr = params['Network']['lr']
    lr_decay = params['Network']['lr_decay']
    load_best_state = params['Network']['load_best_state']
    vp_latent_dim = params['Network']['vp_latent_dim']
    num_classes = params['Network']['n_class']
    L2penalty = params['Network']['penalty']
    init_vp = params['Network']['init_vp']
    epochs_cls = params['Network']['epochs_cls']
    classifier_name = params['Network']['classifier_name']
