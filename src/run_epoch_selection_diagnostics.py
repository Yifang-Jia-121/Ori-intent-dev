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


def compact_epoch_rows(epoch_rows):
    return [
        {
            "dataset": row["dataset"],
            "repeat": row["repeat"],
            "epoch": row["epoch"],
            "val_hr@50": row["val_hr@50"],
            "val_niche_rate@50": row["val_niche_rate@50"],
        }
        for row in epoch_rows
    ]


def selection_summary_rows(selection_rows):
    rows = []
    by_key = {}
    for row in selection_rows:
        by_key[(row["dataset"], row["repeat"], row["strategy"])] = row

    dataset_repeats = sorted({(dataset_name, repeat) for dataset_name, repeat, _ in by_key})
    for dataset_name, repeat in dataset_repeats:
        hr_row = by_key.get((dataset_name, repeat, "best_hr"))
        niche_row = by_key.get((dataset_name, repeat, "best_niche_rate"))
        if hr_row is None or niche_row is None:
            continue
        rows.append({
            "dataset": dataset_name,
            "repeat": repeat,
            "hr_best_epoch": hr_row["selected_epoch"],
            "hr_best_val_hr@50": hr_row["selected_val_hr@50"],
            "hr_best_val_niche_rate@50": hr_row["selected_val_niche_rate@50"],
            "hr_best_test_hr@50": hr_row["test_hr@50"],
            "hr_best_test_niche_rate@50": hr_row["test_niche_rate@50"],
            "niche_best_epoch": niche_row["selected_epoch"],
            "niche_best_val_hr@50": niche_row["selected_val_hr@50"],
            "niche_best_val_niche_rate@50": niche_row["selected_val_niche_rate@50"],
            "niche_best_test_hr@50": niche_row["test_hr@50"],
            "niche_best_test_niche_rate@50": niche_row["test_niche_rate@50"],
            "test_hr_delta_niche_minus_hr": niche_row["test_hr@50"] - hr_row["test_hr@50"],
            "test_niche_delta_niche_minus_hr": niche_row["test_niche_rate@50"] - hr_row["test_niche_rate@50"],
        })
    return rows


def plot_epoch_trend(epoch_rows, selection_rows, output_file):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed; skip epoch trend plot.")
        return None

    by_repeat = {}
    for row in epoch_rows:
        by_repeat.setdefault(row["repeat"], []).append(row)

    selection_by_repeat = {}
    for row in selection_rows:
        selection_by_repeat.setdefault(row["repeat"], {})[row["strategy"]] = row

    repeat_count = len(by_repeat)
    fig, axes = plt.subplots(repeat_count, 1, figsize=(10, max(4, 3.5 * repeat_count)), squeeze=False)
    for axis, repeat in zip(axes[:, 0], sorted(by_repeat)):
        rows = sorted(by_repeat[repeat], key=lambda row: row["epoch"])
        epochs = [row["epoch"] for row in rows]
        val_hr = [row["val_hr@50"] * 100 for row in rows]
        val_niche = [row["val_niche_rate@50"] for row in rows]
        axis.plot(epochs, val_hr, marker="o", label="val HR@50 x100")
        axis.plot(epochs, val_niche, marker="o", label="val NicheRate@50")

        repeat_selection = selection_by_repeat.get(repeat, {})
        if "best_hr" in repeat_selection:
            axis.axvline(
                repeat_selection["best_hr"]["selected_epoch"],
                color="#2ca02c",
                linestyle="--",
                label="HR-best epoch",
            )
        if "best_niche_rate" in repeat_selection:
            axis.axvline(
                repeat_selection["best_niche_rate"]["selected_epoch"],
                color="#d62728",
                linestyle="--",
                label="Niche-best epoch",
            )
        axis.set_title(f"repeat {repeat}")
        axis.set_xlabel("epoch")
        axis.set_ylabel("value")
        axis.legend()

    fig.tight_layout()
    fig.savefig(output_file, dpi=160)
    plt.close(fig)
    return output_file


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
    compact_epoch_file = os.path.join(result_path, f"{prefix}-compact-epochs.csv")
    selection_file = os.path.join(result_path, f"{prefix}-selections.csv")
    summary_file = os.path.join(result_path, f"{prefix}-hr-vs-niche-summary.csv")
    json_file = os.path.join(result_path, f"{prefix}.json")
    trend_plot_file = os.path.join(result_path, f"{prefix}-trend.png")
    compact_rows = compact_epoch_rows(epoch_rows)
    summary_rows = selection_summary_rows(selection_rows)
    trend_plot = plot_epoch_trend(epoch_rows, selection_rows, trend_plot_file)
    write_csv(epoch_file, epoch_rows)
    write_csv(compact_epoch_file, compact_rows)
    write_csv(selection_file, selection_rows)
    write_csv(summary_file, summary_rows)
    with open(json_file, "w") as file:
        file.write(json.dumps({
            "epoch_results": epoch_rows,
            "compact_epoch_results": compact_rows,
            "selection_results": selection_rows,
            "hr_vs_niche_summary": summary_rows,
            "trend_plot": trend_plot,
        }, indent=4))

    print("===========diagnostic files===========")
    print(f"epochs_csv: {epoch_file}")
    print(f"compact_epochs_csv: {compact_epoch_file}")
    print(f"selections_csv: {selection_file}")
    print(f"hr_vs_niche_summary_csv: {summary_file}")
    print(f"json: {json_file}")
    if trend_plot is not None:
        print(f"trend_plot: {trend_plot}")


if __name__ == "__main__":
    main()
