import argparse
import csv
import json
import os
import time

import pandas as pd
import torch

import Procedure
import dataloader
import sampler
import utils
import world
from models import my_graph_models


def format_value(value):
    return str(value).replace(".", "p").replace("-", "m")


def train_and_test(dataset, config, weight_file):
    rec_model = my_graph_models.CISGNN(config, dataset).to(world.device)
    bpr = sampler.BPRLoss(rec_model, config)
    best_perf = {"hr@50": 0, "ndcg@50": 0, "best_epoch": 0}

    for epoch in range(1, world.TRAIN_epochs + 1):
        loss = Procedure.BPR_train_original(dataset, rec_model, bpr, epoch)
        val_results = Procedure.Evaluate(dataset, rec_model, epoch, False)
        print(
            f"EPOCH[{epoch}/{world.TRAIN_epochs}] loss={loss:.6f} "
            f"val_hr@50={val_results['hr@50']:.4f} "
            f"val_ndcg@50={val_results['ndcg@50']:.4f} "
            f"val_niche_rate@50={val_results['niche_rate@50']:.4f} "
            f"val_novelty@50={val_results['novelty@50']:.4f}"
        )

        if val_results["hr@50"] + 0.0001 > best_perf["hr@50"]:
            best_perf["hr@50"] = val_results["hr@50"]
            best_perf["ndcg@50"] = val_results["ndcg@50"]
            best_perf["best_epoch"] = epoch
            torch.save(rec_model.state_dict(), weight_file)
            print("[Increased] model saved")

        if epoch - best_perf["best_epoch"] > world.PATIENCE:
            print(f"early stop at {epoch} epoch")
            break

    rec_model.load_state_dict(torch.load(weight_file, map_location=torch.device("cpu")))
    test_results = Procedure.Test(dataset, rec_model, False, False)
    return best_perf, test_results


def write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def parse_args():
    parser = argparse.ArgumentParser(description="Train CISGNN with counterfactual r_star values.")
    parser.add_argument("--dataset", default=world.dataset)
    parser.add_argument("--values", nargs="+", type=float, default=[0.0, 0.1, 0.2, 0.5, 1.0])
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    world.dataset = args.dataset
    if args.epochs is not None:
        world.TRAIN_epochs = args.epochs

    dataset = dataloader.DecGraphDataset(world.dataset)
    config = world.config
    base_file = utils.getFileName("CISGNN")
    weight_path = os.path.join(world.FILE_PATH, dataset.dataset_name)
    result_path = os.path.join(world.RESULT_PATH, dataset.dataset_name)
    os.makedirs(weight_path, exist_ok=True)
    os.makedirs(result_path, exist_ok=True)

    print("===========r_star search config===========")
    print(f"dataset: {world.dataset}")
    print(f"values: {args.values}")
    print(f"repeat: {args.repeat}")
    print(f"TRAIN_epochs: {world.TRAIN_epochs}")
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    print("==========================================")

    rows = []
    for value in args.values:
        config["r_star"] = value
        for repeat in range(1, args.repeat + 1):
            utils.set_seed(world.seed + repeat - 1)
            tag = f"rstar_{format_value(value)}_r{repeat}"
            weight_file = os.path.join(weight_path, f"{base_file}-{tag}.pth.tar")
            print(f"********** r_star={value}, repeat={repeat} starts **********")
            best_perf, test_results = train_and_test(dataset, config, weight_file)
            row = {
                "dataset": world.dataset,
                "r_star": value,
                "repeat": repeat,
                "best_epoch": best_perf["best_epoch"],
                "best_val_hr@50": best_perf["hr@50"],
                "best_val_ndcg@50": best_perf["ndcg@50"],
                **test_results,
            }
            rows.append(row)
            print(json.dumps(row, indent=4))

    detail = pd.DataFrame(rows)
    numeric_cols = [
        col for col in detail.columns
        if col not in {"dataset"} and pd.api.types.is_numeric_dtype(detail[col])
    ]
    summary = detail.groupby("r_star", as_index=False)[numeric_cols].mean()

    prefix = f"{base_file}-rstar-search"
    detail_file = os.path.join(result_path, f"{prefix}-detail.csv")
    summary_file = os.path.join(result_path, f"{prefix}-summary.csv")
    json_file = os.path.join(result_path, f"{prefix}-detail.json")
    write_csv(detail_file, detail.to_dict("records"))
    summary.to_csv(summary_file, index=False)
    with open(json_file, "w") as file:
        file.write(json.dumps(rows, indent=4))

    print("===========r_star search summary===========")
    print(summary.to_string(index=False))
    print(f"detail_csv: {detail_file}")
    print(f"summary_csv: {summary_file}")
    print(f"detail_json: {json_file}")


if __name__ == "__main__":
    main()
