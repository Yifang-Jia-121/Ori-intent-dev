import copy
import hashlib
import json
import os
import shutil
import time
from pprint import pprint

import numpy as np
import torch

import Procedure
import dataloader
import sampler
import utils
import world
from models import my_graph_models


def json_default(value):
    if isinstance(value, torch.device):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f'Object of type {type(value).__name__} is not JSON serializable')


def save_checkpoint(state_dict, checkpoint_file):
    temporary_file = f'{checkpoint_file}.{os.getpid()}.tmp'
    try:
        torch.save(state_dict, temporary_file)
        os.replace(temporary_file, checkpoint_file)
    finally:
        if os.path.exists(temporary_file):
            try:
                os.remove(temporary_file)
            except OSError:
                pass


def accuracy_improved(metrics, best, metric_k, tolerance=0.0001):
    if best is None:
        return True
    hr_key = f'hr@{metric_k}'
    ndcg_key = f'ndcg@{metric_k}'
    hr_delta = metrics[hr_key] - best['metrics'][hr_key]
    if hr_delta > tolerance:
        return True
    return (
        hr_delta >= 0
        and abs(hr_delta) <= tolerance
        and metrics[ndcg_key] > best['metrics'][ndcg_key]
    )


def constrained_improved(candidate, best, metric_k, tolerance=1e-12):
    if best is None:
        return True
    niche_key = f'niche_rate@{metric_k}'
    hr_key = f'hr@{metric_k}'
    ndcg_key = f'ndcg@{metric_k}'
    candidate_metrics = candidate['metrics']
    best_metrics = best['metrics']

    comparisons = (
        (candidate_metrics[niche_key], best_metrics[niche_key], True),
        (candidate_metrics[hr_key], best_metrics[hr_key], True),
        (candidate_metrics[ndcg_key], best_metrics[ndcg_key], True),
        (-float(candidate['counterfactual_k']),
         -float(best['counterfactual_k']), True),
        (-int(candidate['epoch']), -int(best['epoch']), True),
        (-float(candidate['ci_alpha']), -float(best['ci_alpha']), True),
    )
    for candidate_value, best_value, higher_is_better in comparisons:
        delta = candidate_value - best_value
        if abs(delta) <= tolerance:
            continue
        return delta > 0 if higher_is_better else delta < 0
    return False


def build_candidate(ci_alpha, counterfactual_k, epoch, metrics):
    return {
        'ci_alpha': float(ci_alpha),
        'counterfactual_k': float(counterfactual_k),
        'epoch': int(epoch),
        'metrics': copy.deepcopy(metrics),
    }


def build_model(config, dataset):
    model = my_graph_models.CISGNN(config, dataset).to(world.device)
    loss = sampler.BPRLoss(model, config)
    return model, loss


def train_accuracy_reference(dataset, base_config, checkpoint_file, metric_k):
    utils.set_seed(world.seed)
    model_config = copy.copy(base_config)
    model, bpr = build_model(model_config, dataset)
    best = None

    print('\n========== Stage 1: validation HR reference ==========')
    print(f"baseline ci_alpha={model_config['ci_alpha']}, k={model_config['k']}")
    for epoch in range(1, world.TRAIN_epochs + 1):
        loss = Procedure.BPR_train_original(dataset, model, bpr, epoch)
        metrics = Procedure.Evaluate(
            dataset,
            model,
            epoch,
            False,
            counterfactual_k=model_config['k'],
            niche_items=dataset.validation_niche_items,
        )
        print(
            f"[reference][{epoch}/{world.TRAIN_epochs}] loss={loss:.6f} "
            f"HR@{metric_k}={metrics[f'hr@{metric_k}']:.6f} "
            f"NicheRatio@{metric_k}={metrics[f'niche_rate@{metric_k}']:.6f}"
        )
        if accuracy_improved(metrics, best, metric_k):
            best = build_candidate(
                model_config['ci_alpha'], model_config['k'], epoch, metrics)
            save_checkpoint(model.state_dict(), checkpoint_file)
            print(f"  [reference improved] checkpoint saved at epoch {epoch}")
        if epoch - best['epoch'] >= world.PATIENCE:
            print(f"  reference early stop at epoch {epoch}")
            break
    return best


def search_constrained_candidates(dataset, base_config, hr_floor,
                                  selected_checkpoint_file, metric_k,
                                  initial_best):
    global_best = copy.deepcopy(initial_best)
    alpha_summaries = []
    selection_trace = [copy.deepcopy(initial_best)]
    hr_key = f'hr@{metric_k}'

    print('\n========== Stage 2: constrained validation search ==========')
    print(f'fixed validation HR floor: {hr_floor:.6f}')
    for ci_alpha in world.CONSTRAINED_ALPHA_GRID:
        utils.set_seed(world.seed)
        model_config = copy.copy(base_config)
        model_config['ci_alpha'] = float(ci_alpha)
        model, bpr = build_model(model_config, dataset)

        local_best = None
        local_best_hr = None
        last_progress_epoch = 0
        print(f'\n----- ci_alpha={ci_alpha} -----')

        for epoch in range(1, world.TRAIN_epochs + 1):
            loss = Procedure.BPR_train_original(dataset, model, bpr, epoch)
            epoch_feasible = False
            epoch_best_hr = float('-inf')
            epoch_best_niche = float('-inf')

            metrics_by_k = Procedure.EvaluateKs(
                dataset,
                model,
                world.CONSTRAINED_K_GRID,
                niche_items=dataset.validation_niche_items,
            )

            for counterfactual_k in world.CONSTRAINED_K_GRID:
                metrics = metrics_by_k[counterfactual_k]
                candidate = build_candidate(
                    ci_alpha, counterfactual_k, epoch, metrics)
                epoch_best_hr = max(epoch_best_hr, metrics[hr_key])
                epoch_best_niche = max(
                    epoch_best_niche,
                    metrics[f'niche_rate@{metric_k}'],
                )

                if accuracy_improved(metrics, local_best_hr, metric_k):
                    local_best_hr = candidate
                    if local_best is None:
                        last_progress_epoch = epoch

                if metrics[hr_key] + 1e-12 < hr_floor:
                    continue

                epoch_feasible = True
                if constrained_improved(candidate, local_best, metric_k):
                    local_best = candidate
                    last_progress_epoch = epoch
                if constrained_improved(candidate, global_best, metric_k):
                    global_best = candidate
                    save_checkpoint(model.state_dict(), selected_checkpoint_file)
                    selection_trace.append(copy.deepcopy(candidate))
                    print(
                        f"  [selected] epoch={epoch}, k={counterfactual_k}, "
                        f"HR={metrics[hr_key]:.6f}, "
                        f"NicheRatio={metrics[f'niche_rate@{metric_k}']:.6f}"
                    )

            print(
                f"[search][alpha={ci_alpha}][{epoch}/{world.TRAIN_epochs}] "
                f"loss={loss:.6f}, best_HR={epoch_best_hr:.6f}, "
                f"best_NicheRatio={epoch_best_niche:.6f}, "
                f"feasible={epoch_feasible}"
            )

            if epoch - last_progress_epoch >= world.CONSTRAINED_PATIENCE:
                print(f'  constrained early stop at epoch {epoch}')
                break

        alpha_summaries.append({
            'ci_alpha': float(ci_alpha),
            'best_hr_candidate': local_best_hr,
            'best_feasible_candidate': local_best,
        })

    return global_best, alpha_summaries, selection_trace


def main():
    utils.set_seed(world.seed)
    dataset = dataloader.DecGraphDataset(world.dataset)
    base_config = copy.copy(world.config)
    metric_k = int(world.CONSTRAINED_METRIC_K)

    if metric_k not in world.topks:
        raise ValueError(
            f'CONSTRAINED_METRIC_K={metric_k} must be present in topks={world.topks}')
    if not 0 < world.CONSTRAINED_HR_RETENTION <= 1:
        raise ValueError('CONSTRAINED_HR_RETENTION must be in (0, 1].')
    if not world.CONSTRAINED_ALPHA_GRID or not world.CONSTRAINED_K_GRID:
        raise ValueError('The constrained alpha and k grids must not be empty.')

    weight_path = os.path.join(world.FILE_PATH, dataset.dataset_name)
    result_path = os.path.join(world.RESULT_PATH, dataset.dataset_name)
    os.makedirs(weight_path, exist_ok=True)
    os.makedirs(result_path, exist_ok=True)

    search_signature = json.dumps({
        'baseline_ci_alpha': base_config['ci_alpha'],
        'baseline_k': base_config['k'],
        'alpha_grid': world.CONSTRAINED_ALPHA_GRID,
        'k_grid': world.CONSTRAINED_K_GRID,
        'metric_k': metric_k,
        'hr_retention': world.CONSTRAINED_HR_RETENTION,
        'train_epochs': world.TRAIN_epochs,
        'patience': world.CONSTRAINED_PATIENCE,
    }, sort_keys=True)
    search_id = hashlib.sha1(search_signature.encode('ascii')).hexdigest()[:8]
    run_id = f"{time.strftime('%Y%m%d-%H%M%S')}-pid{os.getpid()}"
    search_name = (
        f"CISGNN-constrained-{base_config['latent_dim_rec']}-"
        f"{base_config['layer']}layer-{base_config['social_layer']}social_layer-"
        f"basea{base_config['ci_alpha']:g}-basek{base_config['k']:g}-"
        f"retain{world.CONSTRAINED_HR_RETENTION:g}-{search_id}-seed{world.seed}-"
        f"{run_id}"
    )
    reference_checkpoint_file = os.path.join(
        weight_path, f'{search_name}-hr-reference.pth.tar')
    selected_checkpoint_file = os.path.join(
        weight_path, f'{search_name}-selected.pth.tar')
    result_file = os.path.join(result_path, f'{search_name}.json')

    minimum_free_bytes = 64 * 1024 * 1024
    free_bytes = shutil.disk_usage(weight_path).free
    if free_bytes < minimum_free_bytes:
        raise RuntimeError(
            f'Insufficient disk space for checkpoints in {weight_path}: '
            f'{free_bytes / (1024 ** 2):.1f} MB free; at least '
            f'{minimum_free_bytes / (1024 ** 2):.0f} MB is required.')

    print('=========== constrained search config ===========')
    print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()))
    pprint(base_config)
    print('alpha grid:', world.CONSTRAINED_ALPHA_GRID)
    print('k grid:', world.CONSTRAINED_K_GRID)
    print('HR retention:', world.CONSTRAINED_HR_RETENTION)

    reference = train_accuracy_reference(
        dataset, base_config, reference_checkpoint_file, metric_k)
    hr_floor = (
        float(reference['metrics'][f'hr@{metric_k}'])
        * float(world.CONSTRAINED_HR_RETENTION)
    )
    shutil.copyfile(reference_checkpoint_file, selected_checkpoint_file)
    selected, alpha_summaries, selection_trace = search_constrained_candidates(
        dataset,
        base_config,
        hr_floor,
        selected_checkpoint_file,
        metric_k,
        reference,
    )

    selected_config = copy.copy(base_config)
    selected_config['ci_alpha'] = selected['ci_alpha']
    selected_config['k'] = selected['counterfactual_k']
    selected_model = my_graph_models.CISGNN(
        selected_config, dataset).to(world.device)
    selected_model.load_state_dict(torch.load(
        selected_checkpoint_file, map_location=world.device))

    print('\n[TEST] evaluating the selected configuration once')
    test_results = Procedure.Test(
        dataset,
        selected_model,
        False,
        False,
        counterfactual_k=selected['counterfactual_k'],
    )

    output = {
        'base_config': base_config,
        'selection_policy': {
            'run_id': run_id,
            'type': 'maximize_validation_niche_ratio_with_hr_constraint',
            'metric_k': metric_k,
            'hr_reference': 'best validation HR of the fixed baseline config',
            'validation_niche_labels': (
                'training-only item frequency; zero-frequency candidate items '
                'are classified explicitly'
            ),
            'test_niche_labels': 'full preprocessed interactions, matching the paper',
            'hr_retention': world.CONSTRAINED_HR_RETENTION,
            'hr_floor': hr_floor,
            'tie_breakers': [
                f'validation HR@{metric_k}',
                f'validation NDCG@{metric_k}',
                'smaller k',
                'earlier epoch',
                'smaller ci_alpha',
            ],
            'alpha_grid': world.CONSTRAINED_ALPHA_GRID,
            'k_grid': world.CONSTRAINED_K_GRID,
            'train_epochs': world.TRAIN_epochs,
            'reference_patience': world.PATIENCE,
            'constrained_patience': world.CONSTRAINED_PATIENCE,
        },
        'split_diagnostics': dataset.split_diagnostics,
        'reference_validation': reference,
        'selected_validation': selected,
        'alpha_summaries': alpha_summaries,
        'selection_trace': selection_trace,
        'test_results': test_results,
    }
    with open(result_file, 'w') as file:
        json.dump(output, file, indent=4, default=json_default)

    print(f'Result saved to {result_file}')
    print('Selected configuration:')
    pprint(selected)


if __name__ == '__main__':
    main()
