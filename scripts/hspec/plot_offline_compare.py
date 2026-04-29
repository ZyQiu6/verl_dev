"""
Plot comparisons between offline current-HSpec and optim-HSpec simulations.

Features:
1. Prompt-level merged entry plots:
   - For up to `max_plot_num` shared (epoch, step, prompt) items,
     merge all shared entries within that prompt by slicing each entry to the
     maximum common prefix length shared by:
       * current entry_selected_accept_lengths
       * current entry_selected_draft_lengths
       * optim  entry_selected_accept_lengths
     Then plot the three merged arrays in one figure.
2. Prompt-level similarity plots:
   - For up to `max_plot_num` shared (epoch, step, prompt) items,
     plot:
       * current selected_similarities
       * optim   selected_similarities
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot comparisons between offline current and optim HSpec simulations.",
    )
    parser.add_argument(
        "--current-root",
        type=Path,
        required=True,
        help="Output root of offline_current_sim.py",
    )
    parser.add_argument(
        "--optim-root",
        type=Path,
        required=True,
        help="Output root of offline_optim_sim.py",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="Directory to save comparison plots.",
    )
    parser.add_argument(
        "--max-plot-num",
        type=int,
        default=20,
        help="Maximum number of shared prompt plots to generate for each plot type.",
    )
    parser.add_argument(
        "--plot-num-offset",
        type=int,
        default=0,
        help="Starting position of the data source interval in plot."
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="Figure dpi.",
    )
    return parser.parse_args()


def load_npz(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=True) as data:
        return {k: data[k] for k in data.files}


def decode_scalar(x: Any) -> Any:
    if isinstance(x, np.ndarray) and x.ndim == 0:
        return x.item()
    return x


def safe_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", text)


def flatten_object_array(values: Any, dtype: np.dtype) -> np.ndarray:
    """Flatten a rollout-level object array into a single 1D ndarray."""
    pieces: list[np.ndarray] = []
    if isinstance(values, np.ndarray) and values.dtype == object:
        iterable = values.tolist()
    elif isinstance(values, list):
        iterable = values
    else:
        iterable = [values]

    for item in iterable:
        if item is None:
            continue
        arr = np.asarray(item, dtype=dtype)
        if arr.size == 0:
            continue
        pieces.append(arr.reshape(-1))

    if not pieces:
        return np.empty((0,), dtype=dtype)
    return np.concatenate(pieces, axis=0)


def build_prompt_file_map(root: Path, decision_dir_name: str) -> dict[tuple[int, str, str], Path]:
    result: dict[tuple[int, str, str], Path] = {}
    for epoch_dir in sorted(root.glob("epoch_*")):
        if not epoch_dir.is_dir():
            continue
        try:
            epoch = int(epoch_dir.name.split("_")[-1])
        except ValueError:
            continue
        decision_dir = epoch_dir / decision_dir_name
        if not decision_dir.exists():
            continue
        for npz_path in sorted(decision_dir.glob("step_*/*.npz")):
            step = npz_path.parent.name
            data = load_npz(npz_path)
            prompt_id = str(decode_scalar(data["prompt_id"]))
            result[(epoch, step, prompt_id)] = npz_path
    return result


def build_shared_prompt_items(
    current_root: Path,
    optim_root: Path,
) -> list[tuple[tuple[int, str, str], Path, Path]]:
    current_map = build_prompt_file_map(current_root, "current_decisions")
    optim_map = build_prompt_file_map(optim_root, "optim_decisions")
    shared_keys = sorted(set(current_map.keys()) & set(optim_map.keys()))
    return [(key, current_map[key], optim_map[key]) for key in shared_keys]


def merge_prompt_entry_lengths(
    *,
    current_data: dict[str, Any],
    optim_data: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    current_accept_obj = current_data["entry_selected_accept_lengths"].tolist()
    current_draft_obj = current_data["entry_selected_draft_lengths"].tolist()
    optim_accept_obj = optim_data["entry_selected_accept_lengths"].tolist()

    merged_current_accept: list[np.ndarray] = []
    merged_current_draft: list[np.ndarray] = []
    merged_optim_accept: list[np.ndarray] = []

    num_entries = min(
        len(current_accept_obj),
        len(current_draft_obj),
        len(optim_accept_obj),
    )
    for entry_id in range(num_entries):
        current_accept = np.asarray(current_accept_obj[entry_id], dtype=np.int32).reshape(-1)
        current_draft = np.asarray(current_draft_obj[entry_id], dtype=np.int32).reshape(-1)
        optim_accept = np.asarray(optim_accept_obj[entry_id], dtype=np.int32).reshape(-1)
        common_len = min(len(current_accept), len(current_draft), len(optim_accept))
        if common_len <= 0:
            continue
        merged_current_accept.append(current_accept[:common_len])
        merged_current_draft.append(current_draft[:common_len])
        merged_optim_accept.append(optim_accept[:common_len])

    if not merged_current_accept:
        empty = np.empty((0,), dtype=np.int32)
        return empty, empty, empty

    return (
        np.concatenate(merged_current_accept, axis=0),
        np.concatenate(merged_current_draft, axis=0),
        np.concatenate(merged_optim_accept, axis=0),
    )


def plot_merged_prompt_entry_item(
    *,
    item_key: tuple[int, str, str],
    current_path: Path,
    optim_path: Path,
    output_dir: Path,
    dpi: int,
) -> dict[str, Any]:
    epoch, step, prompt_id = item_key
    current_data = load_npz(current_path)
    optim_data = load_npz(optim_path)

    current_accept, current_draft, optim_accept = merge_prompt_entry_lengths(
        current_data=current_data,
        optim_data=optim_data,
    )

    fig = plt.figure(figsize=(10, 5), dpi=dpi)
    ax = fig.add_subplot(111)
    ax.plot(np.arange(len(current_accept)), current_accept, label="current_accept_len", linewidth=1.8)
    ax.plot(np.arange(len(current_draft)), current_draft, label="current_draft_len", linewidth=1.8)
    ax.plot(np.arange(len(optim_accept)), optim_accept, label="optim_accept_len", linewidth=1.8)
    ax.set_title(f"Merged Entry Compare | epoch={epoch} step={step} prompt={prompt_id}")
    ax.set_xlabel("merged selection index")
    ax.set_ylabel("length")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()

    filename = f"epoch_{epoch:04d}__{safe_name(step)}__{safe_name(prompt_id)}.png"
    output_path = output_dir / filename
    fig.savefig(output_path)
    plt.close(fig)

    return {
        "epoch": epoch,
        "step": step,
        "prompt_id": prompt_id,
        "merged_current_accept_count": int(len(current_accept)),
        "merged_current_draft_count": int(len(current_draft)),
        "merged_optim_accept_count": int(len(optim_accept)),
        "output_path": str(output_path),
    }


def plot_prompt_item(
    *,
    item_key: tuple[int, str, str],
    current_path: Path,
    optim_path: Path,
    output_dir: Path,
    dpi: int,
) -> dict[str, Any]:
    epoch, step, prompt_id = item_key
    current_data = load_npz(current_path)
    optim_data = load_npz(optim_path)

    current_sims = flatten_object_array(current_data["selected_similarities"], np.float32)
    optim_sims = flatten_object_array(optim_data["selected_similarities"], np.float32)

    fig = plt.figure(figsize=(10, 5), dpi=dpi)
    ax = fig.add_subplot(111)
    ax.plot(np.arange(len(current_sims)), current_sims, label="current_selected_similarities", linewidth=1.8)
    ax.plot(np.arange(len(optim_sims)), optim_sims, label="optim_selected_similarities", linewidth=1.8)
    ax.set_title(f"Prompt Similarity Compare | epoch={epoch} step={step} prompt={prompt_id}")
    ax.set_xlabel("selection index")
    ax.set_ylabel("dot similarity")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()

    filename = f"epoch_{epoch:04d}__{safe_name(step)}__{safe_name(prompt_id)}.png"
    output_path = output_dir / filename
    fig.savefig(output_path)
    plt.close(fig)

    return {
        "epoch": epoch,
        "step": step,
        "prompt_id": prompt_id,
        "current_similarity_count": int(len(current_sims)),
        "optim_similarity_count": int(len(optim_sims)),
        "output_path": str(output_path),
    }


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    merged_entry_plot_dir = args.output_root / "draft_len_plots"
    prompt_plot_dir = args.output_root / "similarity_plots"
    merged_entry_plot_dir.mkdir(parents=True, exist_ok=True)
    prompt_plot_dir.mkdir(parents=True, exist_ok=True)

    shared_prompt_items = build_shared_prompt_items(args.current_root, args.optim_root)

    prompt_items_to_plot = shared_prompt_items[args.plot_num_offset : args.plot_num_offset + max(args.max_plot_num, 0)]

    merged_entry_outputs: list[dict[str, Any]] = []
    for item_key, current_path, optim_path in prompt_items_to_plot:
        merged_entry_outputs.append(
            plot_merged_prompt_entry_item(
                item_key=item_key,
                current_path=current_path,
                optim_path=optim_path,
                output_dir=merged_entry_plot_dir,
                dpi=args.dpi,
            )
        )

    prompt_outputs: list[dict[str, Any]] = []
    for item_key, current_path, optim_path in prompt_items_to_plot:
        prompt_outputs.append(
            plot_prompt_item(
                item_key=item_key,
                current_path=current_path,
                optim_path=optim_path,
                output_dir=prompt_plot_dir,
                dpi=args.dpi,
            )
        )

    summary = {
        "max_plot_num": int(args.max_plot_num),
        "plot_num_offset": int(args.plot_num_offset),
        "shared_prompt_count": int(len(shared_prompt_items)),
        "plotted_prompt_count": int(len(prompt_outputs)),
        "plotted_merged_entry_prompt_count": int(len(merged_entry_outputs)),
        "similarity_plots": prompt_outputs,
        "draft_len_plots": merged_entry_outputs,
    }
    with open(args.output_root / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
