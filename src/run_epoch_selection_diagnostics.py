import argparse
import csv
import json
import os
import time

import torch

import Procedure
import dataloader
import sampler
import utils
import world
from models import my_graph_models


def to_builtin(value):
    if hasattr(value, "item"):
        return value.item()
    return value


def write_csv(path, rows):
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def prefixed_metrics(prefix, metrics):
    return {f"{prefix}{key}": to_builtin(value) for key, value in metrics.items()}


def strategy_scores(val_results, niche_weight):
    return {
        "best_hr": val_results["hr@50"],
        "best_ndcg": val_results["ndcg@50"],
        "best_novelty": val_results["novelty@50"],
        "best_niche_rate": val_results["niche_rate@50"],
        "best_hr_plus_niche": val_results["hr@50"] + niche_weight * val_results["niche_rate@50"] / 100,
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Record per-epoch validation metrics and test checkpoints selected by different strategies."
    )
    parser.add_argument("--dataset", default=world.dataset)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=world.TRAIN_epochs)
    parser.add_argument("--niche-weight", type=float, default=1.0)
    parser.add_argument("--early-stop", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    world.dataset = args.dataset
    world.TRAIN_epochs = args.epochs

    dataset = dataloader.DecGraphDataset(world.dataset)
    config = world.config
    base_file = utils.getFileName("CISGNN")
    weight_path = os.path.join(world.FILE_PATH, dataset.dataset_name)
    result_path = os.path.join(world.RESULT_PATH, dataset.dataset_name)
    os.makedirs(weight_path, exist_ok=True)
    os.makedirs(result_path, exist_ok=True)

    print("===========epoch selection diagnostics===========")
    print(f"dataset: {world.dataset}")
    print(f"repeat: {args.repeat}")
    print(f"epochs: {args.epochs}")
    print(f"niche_weight: {args.niche_weight}")
    print(f"early_stop: {args.early_stop}")
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    print("=================================================")

    epoch_rows = []
    selection_rows = []
    strategies = ["best_hr", "best_ndcg", "best_novelty", "best_niche_rate", "best_hr_plus_niche"]

    for repeat in range(1, args.repeat + 1):
        utils.set_seed(world.seed + repeat - 1)
        rec_model = my_graph_models.CISGNN(config, dataset).to(world.device)
        bpr = sampler.BPRLoss(rec_model, config)
        best = {
            strategy: {"score": float("-inf"), "epoch": 0, "val_results": {}, "weight_file": ""}
            for strategy in strategies
        }
        best_hr_epoch = 0

        print(f"********** diagnostic repeat={repeat} starts **********")
        for epoch in range(1, world.TRAIN_epochs + 1):
            loss = Procedure.BPR_train_original(dataset, rec_model, bpr, epoch)
            val_results = Procedure.Evaluate(dataset, rec_model, epoch, False)
            scores = strategy_scores(val_results, args.niche_weight)
            row = {
                "dataset": world.dataset,
                "repeat": repeat,
                "epoch": epoch,
                "loss": loss,
                **prefixed_metrics("val_", val_results),
                **{f"selection_score_{key}": to_builtin(value) for key, value in scores.items()},
            }
            epoch_rows.append(row)

            print(
                f"EPOCH[{epoch}/{world.TRAIN_epochs}] loss={loss:.6f} "
                f"val_hr@50={val_results['hr@50']:.4f} "
                f"val_ndcg@50={val_results['ndcg@50']:.4f} "
                f"val_niche_rate@50={val_results['niche_rate@50']:.4f} "
                f"val_novelty@50={val_results['novelty@50']:.4f}"
            )

            for strategy, score in scores.items():
                if score > best[strategy]["score"]:
                    weight_file = os.path.join(
                        weight_path,
                        f"{base_file}-epochdiag-{strategy}-r{repeat}.pth.tar",
                    )
                    torch.save(rec_model.state_dict(), weight_file)
                    best[strategy] = {
                        "score": to_builtin(score),
                        "epoch": epoch,
                        "val_results": {key: to_builtin(value) for key, value in val_results.items()},
                        "weight_file": weight_file,
                    }
                    if strategy == "best_hr":
                        best_hr_epoch = epoch

            if args.early_stop and epoch - best_hr_epoch > world.PATIENCE:
                print(f"early stop at {epoch} epoch")
                break

        for strategy in strategies:
            rec_model.load_state_dict(torch.load(best[strategy]["weight_file"], map_location=torch.device("cpu")))
            test_results = Procedure.Test(dataset, rec_model, False, False)
            result_row = {
                "dataset": world.dataset,
                "repeat": repeat,
                "strategy": strategy,
                "selected_epoch": best[strategy]["epoch"],
                "selection_score": best[strategy]["score"],
                **prefixed_metrics("selected_val_", best[strategy]["val_results"]),
                **prefixed_metrics("test_", test_results),
            }
            selection_rows.append(result_row)
            print(json.dumps(result_row, indent=4))

    prefix = f"{base_file}-epoch-selection-diagnostics"
    epoch_file = os.path.join(result_path, f"{prefix}-epochs.csv")
    selection_file = os.path.join(result_path, f"{prefix}-selections.csv")
    json_file = os.path.join(result_path, f"{prefix}.json")
    write_csv(epoch_file, epoch_rows)
    write_csv(selection_file, selection_rows)
    with open(json_file, "w") as file:
        file.write(json.dumps({
            "epoch_results": epoch_rows,
            "selection_results": selection_rows,
        }, indent=4))

    print("===========diagnostic files===========")
    print(f"epochs_csv: {epoch_file}")
    print(f"selections_csv: {selection_file}")
    print(f"json: {json_file}")


if __name__ == "__main__":
    main()
