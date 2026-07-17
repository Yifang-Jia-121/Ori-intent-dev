REQUIRED_COLUMNS = [
    "epoch",
    "val_hr@50",
    "val_ndcg@50",
    "val_novelty@50",
    "val_niche_rate@50",
]


def _to_records(epoch_metrics):
    if hasattr(epoch_metrics, "to_dict"):
        return epoch_metrics.to_dict("records")
    return [dict(row) for row in epoch_metrics]


def _as_float(value):
    if hasattr(value, "item"):
        return float(value.item())
    return float(value)


def _validate_records(records):
    if not records:
        raise ValueError("epoch_metrics is empty.")
    missing = [column for column in REQUIRED_COLUMNS if column not in records[0]]
    if missing:
        raise ValueError(f"epoch_metrics missing required columns: {missing}")


def _score(row, mode, niche_weight):
    if mode == "hr":
        return _as_float(row["val_hr@50"])
    if mode == "ndcg":
        return _as_float(row["val_ndcg@50"])
    if mode == "novelty":
        return _as_float(row["val_novelty@50"])
    if mode == "niche":
        return _as_float(row["val_niche_rate@50"])
    if mode == "hr_plus_niche":
        return _as_float(row["val_hr@50"]) + niche_weight * _as_float(row["val_niche_rate@50"])
    raise ValueError(f"Unsupported checkpoint selection mode: {mode}")


def select_checkpoint_by_metrics(epoch_metrics, mode, min_hr_ratio=0.95, niche_weight=0.01):
    records = _to_records(epoch_metrics)
    _validate_records(records)

    supported_modes = {
        "hr",
        "ndcg",
        "novelty",
        "niche",
        "hr_plus_niche",
        "acc_constrained_niche",
        "acc_constrained_novelty",
    }
    if mode not in supported_modes:
        raise ValueError(f"Unsupported checkpoint selection mode: {mode}")

    best_val_hr = max(_as_float(row["val_hr@50"]) for row in records)
    eligible_records = records
    objective_mode = mode

    if mode.startswith("acc_constrained_"):
        min_hr = min_hr_ratio * best_val_hr
        eligible_records = [
            row for row in records
            if _as_float(row["val_hr@50"]) >= min_hr
        ]
        if not eligible_records:
            raise ValueError(
                f"No eligible checkpoints for mode={mode}, min_hr_ratio={min_hr_ratio}."
            )
        objective_mode = "niche" if mode == "acc_constrained_niche" else "novelty"

    selected = max(
        eligible_records,
        key=lambda row: _score(row, objective_mode, niche_weight),
    )
    selected_metrics = {
        key: value.item() if hasattr(value, "item") else value
        for key, value in selected.items()
    }

    return {
        "selected_epoch": int(selected["epoch"]),
        "selected_metrics": selected_metrics,
        "best_val_hr": best_val_hr,
        "eligible_epoch_count": len(eligible_records),
        "selection_mode": mode,
        "min_hr_ratio": min_hr_ratio,
        "selection_score": _score(selected, objective_mode, niche_weight),
    }
