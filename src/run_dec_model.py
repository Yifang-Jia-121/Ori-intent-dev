import time
import os
import shutil
import torch
import numpy as np
import Procedure
import utils
import sampler
from pprint import pprint
import dataloader
from models import my_graph_models
import world
import json


utils.set_seed(world.seed)
dataset = dataloader.DecGraphDataset(world.dataset)
print('===========config================')

print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
print("LOAD:", world.LOAD)
print("Weight path:", world.PATH)
print("Test Topks:", world.topks)
print("using bpr loss")
print('===========end===================')
config = world.config
print(">>SEED:", world.seed)


file = utils.getFileName('CISGNN')
weight_path = os.path.join(world.FILE_PATH, dataset.dataset_name)
if not os.path.exists(weight_path):
    os.makedirs(weight_path, exist_ok=True)
result_path = os.path.join(world.RESULT_PATH,  dataset.dataset_name)
if not os.path.exists(result_path):
    os.makedirs(result_path, exist_ok=True)
result_file = os.path.join(result_path, f'{file}.json')
weight_file = os.path.join(weight_path, f'{file}.pth.tar')

minimum_free_bytes = 64 * 1024 * 1024
free_bytes = shutil.disk_usage(weight_path).free
if free_bytes < minimum_free_bytes:
    raise RuntimeError(
        f'Insufficient disk space for checkpoints in {weight_path}: '
        f'{free_bytes / (1024 ** 2):.1f} MB free; at least '
        f'{minimum_free_bytes / (1024 ** 2):.0f} MB is required. '
        'Free disk space or change world.FILE_PATH before training.'
    )

print(f'#########Starting Experiment:{file}##############')
# ==============================
# torch.autograd.set_detect_anomaly(True)

# config['ci_alpha'] = 0.2
# config['k'] = 20
Recmodel = my_graph_models.CISGNN(config, dataset)
Recmodel = Recmodel.to(world.device)
bpr = sampler.BPRLoss(Recmodel, config)
pprint(world.config)
best_perf = {
    'hr@50': float('-inf'),
    'ndcg@50': 0,
    'best_epoch': 0,
    'counterfactual_k': config['k'],
}
# print(f"********** Run {repeat_num + 1} starts. **********")
for epoch in range(1, world.TRAIN_epochs + 1):
    start = time.time()
    loss = Procedure.BPR_train_original(dataset, Recmodel, bpr, epoch)
    print(f'EPOCH[{epoch}/{world.TRAIN_epochs}][BPR aver loss{loss:.6f}]')
    val_results = Procedure.Evaluate(dataset, Recmodel, epoch, False)
    print('\t \t  Validation hr{:.4f}, ndcg{:.4f},' 
          'niche_rate{:.4f}, novelty{:3f}'.format(val_results['hr@50'],
                                               val_results['ndcg@50'],
                                               val_results['niche_rate@50'],
                                               val_results['novelty@50']))
    if val_results['hr@50'] > best_perf['hr@50'] + 0.0001:
        best_perf['hr@50'] = val_results['hr@50']
        best_perf['ndcg@50'] = val_results['ndcg@50']
        best_perf['best_epoch'] = epoch
        torch.save(Recmodel.state_dict(), weight_file)
        print(f"\t [Increased] model saved with K={config['k']}")
    if epoch - best_perf['best_epoch'] >= world.PATIENCE:
        print("early stop at %d epoch" % epoch)
        break
print("[TEST]")
Recmodel.load_state_dict(torch.load(weight_file, map_location=world.device))
test_results = Procedure.Test(dataset, Recmodel, False, False)


def json_default(value):
    if isinstance(value, torch.device):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f'Object of type {type(value).__name__} is not JSON serializable')


with open(result_file, 'w') as file:
    json.dump(
        {
            'config': config,
            'split_diagnostics': dataset.split_diagnostics,
            'best_validation': best_perf,
            'test_results': test_results,
        },
        file,
        indent=4,
        default=json_default,
    )
