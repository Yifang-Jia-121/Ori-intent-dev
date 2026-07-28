import os
from os.path import join
from warnings import simplefilter

import torch

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

ROOT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_PATH = join(ROOT_PATH, 'data')
FILE_PATH = join(ROOT_PATH, 'saved_models')
RESULT_PATH = join(ROOT_PATH, 'results')


if not os.path.exists(FILE_PATH):
    os.makedirs(FILE_PATH, exist_ok=True)

config = {}
all_dataset = ['Ciao', 'Epinions', 'Philadelphia', 'Tucson']
dataset = 'Ciao'
assert dataset in all_dataset
prepro = '2filter'
delete_ratio = 0
#uniform | xavier_uniform_ | kaiming_uniform_ | normal
config['initial_method'] = 'normal'

config['dec_ui'] = 'ui'
config['layer'] = 2
config['social_layer'] = 2
config['pop_num'] = 20
config['degree_num'] = 20
config['prior'] = True
config['pop_fading'] = True
config['ci_alpha'] = 0.2
# Counterfactual reference score used only during validation and test ranking.
config['k'] = 3
config['latent_dim_rec'] = 64

config['bpr_batch_size'] = 1024
config['test_u_batch_size'] = 100

config['droprate'] = 0.5
config['lr'] = 0.001
config['emb_l2rg'] = 0.0001


GPU = torch.cuda.is_available()
device = torch.device('cuda' if GPU else "cpu")
# device = torch.device("cpu")
seed = 23
LOAD = False
PATH = './saved_models'

config['device'] = device

TRAIN_epochs = 1500
PATIENCE = 10
REPEAT = 1
ng_num = 4
topks = [10, 20, 30, 50, 100]

# Validation-only search for the accuracy-constrained extension. The baseline
# is the fixed ci_alpha/k configuration above; test data is never used here.
CONSTRAINED_ALPHA_GRID = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
CONSTRAINED_K_GRID = [0, 1, 2, 3, 4, 5, 6, 7]
CONSTRAINED_METRIC_K = 50
CONSTRAINED_HR_RETENTION = 0.99
CONSTRAINED_PATIENCE = 10


# testMethod = 'tfo'

simplefilter(action="ignore", category=FutureWarning)
