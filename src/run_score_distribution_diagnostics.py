import argparse
import csv
import json
import os

import numpy as np
import torch

import Procedure
import dataloader
import utils
import world
from models import my_graph_models


def to_builtin(value):
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, np.generic):
        return value.item()
    return value


def write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def stats_row(name, values):
    percentiles = np.percentile(values, [1, 5, 10, 25, 50, 75, 90, 95, 99])
    return {
        "component": name,
        "count": int(len(values)),
        "min": float(np.min(values)),
        "p1": float(percentiles[0]),
        "p5": float(percentiles[1]),
        "p10": float(percentiles[2]),
        "p25": float(percentiles[3]),
        "median": float(percentiles[4]),
        "mean": float(np.mean(values)),
        "p75": float(percentiles[5]),
        "p90": float(percentiles[6]),
        "p95": float(percentiles[7]),
        "p99": float(percentiles[8]),
        "max": float(np.max(values)),
        "std": float(np.std(values)),
    }


def sample_values(values, max_points):
    if len(values) <= max_points:
        return values
    index = np.linspace(0, len(values) - 1, max_points).astype(int)
    return values[index]


def get_score_components(model, users):
    all_users, all_items, _, _ = model.computer()
    mediator_emb1, _ = model.computer_mediator(all_users, all_items)
    users_emb = all_users[users.long()]
    items_emb = all_items
    scores = torch.matmul(users_emb, items_emb.t())
    m_scores = torch.matmul(mediator_emb1[users], items_emb.t())
    sigmoid_m = torch.sigmoid(m_scores)
    return scores, m_scores, sigmoid_m


def collect_component_distributions(dataset, model, users, batch_size, max_sample_points):
    score_chunks = []
    m_score_chunks = []
    sigmoid_chunks = []

    with torch.no_grad():
        for batch_users in utils.minibatch(users, batch_size=batch_size):
            all_pos = dataset.getUserPosItems(batch_users, False)
            batch_users_gpu = torch.Tensor(batch_users).long().to(world.device)
            scores, m_scores, sigmoid_m = get_score_components(model, batch_users_gpu)
            candidate_mask = torch.ones_like(scores, dtype=torch.bool)
            for row, items in enumerate(all_pos):
                if len(items) > 0:
                    candidate_mask[row, torch.Tensor(items).long().to(world.device)] = False

            score_chunks.append(scores[candidate_mask].detach().cpu().numpy())
            m_score_chunks.append(m_scores[candidate_mask].detach().cpu().numpy())
            sigmoid_chunks.append(sigmoid_m[candidate_mask].detach().cpu().numpy())

    score_values = sample_values(np.concatenate(score_chunks), max_sample_points)
    m_score_values = sample_values(np.concatenate(m_score_chunks), max_sample_points)
    sigmoid_values = sample_values(np.concatenate(sigmoid_chunks), max_sample_points)
    return {
        "r_score": score_values,
        "m_score": m_score_values,
        "sigmoid_m_score": sigmoid_values,
    }


def recommendation_frequency_diagnostics(all_rating, item_counts, k):
    top_items = all_rating[:, :k].reshape(-1)
    top_item_counts = np.array([item_counts[int(item)] for item in top_items])
    return {
        f"rec_freq_mean@{k}": float(np.mean(top_item_counts)),
        f"rec_freq_median@{k}": float(np.median(top_item_counts)),
        f"rec_freq_le4_ratio@{k}": float(np.mean(top_item_counts <= 4) * 100),
        f"rec_freq_5_10_ratio@{k}": float(np.mean((top_item_counts >= 5) & (top_item_counts <= 10)) * 100),
        f"rec_freq_11_20_ratio@{k}": float(np.mean((top_item_counts >= 11) & (top_item_counts <= 20)) * 100),
        f"rec_freq_gt20_ratio@{k}": float(np.mean(top_item_counts > 20) * 100),
    }


def evaluate_rstar(dataset, model, users, r_star, batch_size, max_k, topk):
    item_counts = getattr(dataset, "eval_item_counts", dataset.item_counts)
    test_dict = dataset.testDict
    results = {"hr": np.zeros(len(world.topks)), "ndcg": np.zeros(len(world.topks))}
    rating_list = []
    ground_true_list = []

    with torch.no_grad():
        for batch_users in utils.minibatch(users, batch_size=batch_size):
            all_pos = dataset.getUserPosItems(batch_users, False)
            ground_true = [test_dict[u] for u in batch_users]
            batch_users_gpu = torch.Tensor(batch_users).long().to(world.device)
            scores, _, sigmoid_m = get_score_components(model, batch_users_gpu)
            rating = (scores - r_star) * sigmoid_m

            exclude_index = []
            exclude_items = []
            for row, items in enumerate(all_pos):
                exclude_index.extend([row] * len(items))
                exclude_items.extend(items)
            rating[exclude_index, exclude_items] = -1e9
            _, rating_k = torch.topk(rating, k=max_k)
            rating_list.append(rating_k.cpu())
            ground_true_list.append(ground_true)

    pre_results = [Procedure.test_one_batch(batch) for batch in zip(rating_list, ground_true_list)]
    for result in pre_results:
        results["hr"] += result["hr"]
        results["ndcg"] += result["ndcg"]
    results["hr"] /= float(len(users))
    results["ndcg"] /= float(len(users))

    final_results = {}
    for idx, k in enumerate(world.topks):
        final_results[f"hr@{k}"] = float(results["hr"][idx])
        final_results[f"ndcg@{k}"] = float(results["ndcg"][idx])

    all_rating = torch.cat(rating_list, dim=0).cpu().numpy()
    final_results.update(recommendation_frequency_diagnostics(all_rating, item_counts, topk))
    for k in world.topks:
        diversity = utils.diversity_at_k(all_rating, item_counts, dataset.niche_items, k, dataset.n_users)
        final_results[f"novelty@{k}"] = float(diversity["novelty"])
        final_results[f"niche_rate@{k}"] = float(diversity["niche_rate"])
    return final_results


def build_rstar_values(fixed_values, score_stats):
    values = [{"r_star_label": str(value), "r_star": float(value)} for value in fixed_values]
    for key in ["p25", "median", "mean", "p75", "p90"]:
        values.append({"r_star_label": f"r_score_{key}", "r_star": float(score_stats[key])})

    deduped = []
    seen = set()
    for row in values:
        marker = round(row["r_star"], 8)
        if marker not in seen:
            deduped.append(row)
            seen.add(marker)
    return deduped


def plot_outputs(output_prefix, distributions, comparison_rows, topk):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed; skip PNG plots.")
        return []

    plot_files = []
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for axis, (name, values) in zip(axes, distributions.items()):
        axis.hist(values, bins=80, color="#4C78A8", alpha=0.85)
        axis.set_title(name)
        axis.set_ylabel("count")
    fig.tight_layout()
    distribution_file = f"{output_prefix}-score-distribution.png"
    fig.savefig(distribution_file, dpi=160)
    plt.close(fig)
    plot_files.append(distribution_file)

    labels = [row["r_star_label"] for row in comparison_rows]
    x = np.arange(len(labels))
    fig, axis = plt.subplots(figsize=(max(10, len(labels) * 1.1), 5))
    axis.plot(x, [row[f"niche_rate@{topk}"] for row in comparison_rows], marker="o", label=f"NicheRate@{topk}")
    axis.plot(x, [row[f"novelty@{topk}"] for row in comparison_rows], marker="o", label=f"Novelty@{topk}")
    axis.plot(x, [row[f"hr@{topk}"] * 100 for row in comparison_rows], marker="o", label=f"HR@{topk} x100")
    axis.plot(x, [row[f"rec_freq_gt20_ratio@{topk}"] for row in comparison_rows], marker="o", label=f">20 ratio@{topk}")
    axis.set_xticks(x)
    axis.set_xticklabels(labels, rotation=35, ha="right")
    axis.set_ylabel("value")
    axis.legend()
    fig.tight_layout()
    comparison_file = f"{output_prefix}-rstar-comparison.png"
    fig.savefig(comparison_file, dpi=160)
    plt.close(fig)
    plot_files.append(comparison_file)
    return plot_files


def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze r_ai/m_ai score distributions and compare r_star rankings on a trained CISGNN checkpoint."
    )
    parser.add_argument("--dataset", default=world.dataset)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--values", nargs="+", type=float, default=[0.0, 0.1, 0.2, 0.5, 1.0])
    parser.add_argument("--topk", type=int, default=50)
    parser.add_argument("--max-users", type=int, default=None)
    parser.add_argument("--max-sample-points", type=int, default=500000)
    return parser.parse_args()


def main():
    args = parse_args()
    world.dataset = args.dataset
    dataset = dataloader.DecGraphDataset(world.dataset)
    base_file = utils.getFileName("CISGNN")
    checkpoint = args.checkpoint
    if checkpoint is None:
        checkpoint = os.path.join(world.FILE_PATH, dataset.dataset_name, f"{base_file}.pth.tar")
    if not os.path.exists(checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    model = my_graph_models.CISGNN(world.config, dataset).to(world.device)
    model.load_state_dict(torch.load(checkpoint, map_location=world.device))
    model.eval()
    model.forecast = True

    users = list(dataset.testDict.keys())
    if args.max_users is not None:
        users = users[:args.max_users]

    result_path = os.path.join(world.RESULT_PATH, dataset.dataset_name)
    os.makedirs(result_path, exist_ok=True)
    output_prefix = os.path.join(result_path, f"{base_file}-score-distribution-diagnostics")

    print("===========score distribution diagnostics===========")
    print(f"dataset: {world.dataset}")
    print(f"checkpoint: {checkpoint}")
    print(f"users: {len(users)}")
    print(f"fixed r_star values: {args.values}")
    print("====================================================")

    distributions = collect_component_distributions(
        dataset,
        model,
        users,
        world.config["test_u_batch_size"],
        args.max_sample_points,
    )
    stats_rows = [stats_row(name, values) for name, values in distributions.items()]
    r_score_stats = next(row for row in stats_rows if row["component"] == "r_score")
    rstar_rows = build_rstar_values(args.values, r_score_stats)

    comparison_rows = []
    for rstar_row in rstar_rows:
        print(f"Evaluating r_star={rstar_row['r_star_label']} ({rstar_row['r_star']:.6f})")
        metrics = evaluate_rstar(
            dataset,
            model,
            users,
            rstar_row["r_star"],
            world.config["test_u_batch_size"],
            max(world.topks),
            args.topk,
        )
        comparison_rows.append({
            "dataset": world.dataset,
            **rstar_row,
            **{key: to_builtin(value) for key, value in metrics.items()},
        })

    stats_file = f"{output_prefix}-score-stats.csv"
    comparison_file = f"{output_prefix}-rstar-comparison.csv"
    json_file = f"{output_prefix}.json"
    write_csv(stats_file, stats_rows)
    write_csv(comparison_file, comparison_rows)
    plot_files = plot_outputs(output_prefix, distributions, comparison_rows, args.topk)

    with open(json_file, "w") as file:
        file.write(json.dumps({
            "checkpoint": checkpoint,
            "score_stats": stats_rows,
            "rstar_comparison": comparison_rows,
            "plot_files": plot_files,
        }, indent=4))

    print("===========diagnostic files===========")
    print(f"score_stats_csv: {stats_file}")
    print(f"rstar_comparison_csv: {comparison_file}")
    print(f"json: {json_file}")
    for plot_file in plot_files:
        print(f"plot: {plot_file}")


if __name__ == "__main__":
    main()
