"""
Offline simulator for the current HSpec strategy.

This script replays the current HSpec table-query policy offline using
trainer-side HSPEC_DUMP artifacts. For each epoch/prompt/rollout:

1. The query key is the projected hidden state at the current pointer.
2. The selected entry is the table entry with maximum dot similarity.
3. The draft length is the prompt-local table wnd_size.
4. Acceptance is the exact prefix match length between the chosen draft and
   the known rollout continuation.

The script records the same decision fields as the optim simulator so their
outputs can be compared directly later.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np


@dataclass(frozen=True)
class TableView:
    prompt_id: str
    prompt_token_ids: np.ndarray
    epoch: int
    active_table_version: int
    mean: np.ndarray
    components: np.ndarray
    keys: np.ndarray
    rollout_seqs: list[np.ndarray]
    entry_rollout_idx: np.ndarray
    entry_offset: np.ndarray
    n_entries: int
    wnd_size: int
    max_wnd: int
    min_wnd: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Offline simulation of the current HSpec strategy.",
    )
    parser.add_argument(
        "--dump-root",
        type=Path,
        required=True,
        help="Root directory produced by HSPEC_DUMP.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="Directory to write offline simulation outputs.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output files.",
    )
    parser.add_argument(
        "--epoch-start",
        type=int,
        default=1,
        help="First epoch to simulate (default: 1, since epoch 0 has no active table).",
    )
    return parser.parse_args()


def load_npz(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=True) as data:
        return {k: data[k] for k in data.files}


def decode_scalar(value: Any) -> Any:
    if isinstance(value, np.ndarray) and value.ndim == 0:
        return value.item()
    return value


def ensure_output_dir(path: Path, overwrite: bool) -> None:
    if path.exists() and any(path.iterdir()) and not overwrite:
        raise FileExistsError(
            f"Output directory {path} already exists and is non-empty. "
            "Pass --overwrite to reuse it.",
        )
    path.mkdir(parents=True, exist_ok=True)


def iter_epoch_dirs(dump_root: Path, epoch_start: int) -> Iterable[tuple[int, Path]]:
    for epoch_dir in sorted(dump_root.glob("epoch_*")):
        if not epoch_dir.is_dir():
            continue
        try:
            epoch = int(epoch_dir.name.split("_")[-1])
        except ValueError:
            continue
        if epoch < epoch_start:
            continue
        yield epoch, epoch_dir


def load_table(path: Path) -> TableView:
    data = load_npz(path)
    rollout_seqs_obj = data["rollout_seqs"]
    rollout_seqs = [
        np.ascontiguousarray(np.asarray(seq, dtype=np.int32))
        for seq in rollout_seqs_obj.tolist()
    ]
    return TableView(
        prompt_id=str(decode_scalar(data["prompt_id"])),
        prompt_token_ids=np.ascontiguousarray(data["prompt_token_ids"], dtype=np.int32),
        epoch=int(decode_scalar(data["epoch"])),
        active_table_version=int(decode_scalar(data["active_table_version"])),
        mean=np.ascontiguousarray(data["mean"], dtype=np.float32),
        components=np.ascontiguousarray(data["components"], dtype=np.float32),
        keys=np.ascontiguousarray(data["keys"], dtype=np.float32),
        rollout_seqs=rollout_seqs,
        entry_rollout_idx=np.ascontiguousarray(data["entry_rollout_idx"], dtype=np.int32),
        entry_offset=np.ascontiguousarray(data["entry_offset"], dtype=np.int32),
        n_entries=int(decode_scalar(data["n_entries"])),
        wnd_size=int(decode_scalar(data["wnd_size"])),
        max_wnd=int(decode_scalar(data["max_wnd"])),
        min_wnd=int(decode_scalar(data["min_wnd"])),
    )


def longest_exact_match(draft: np.ndarray, continuation: np.ndarray) -> int:
    max_len = min(len(draft), len(continuation))
    if max_len <= 0:
        return 0
    eq = (draft[:max_len] == continuation[:max_len])
    mismatches = np.flatnonzero(~eq)
    if mismatches.size == 0:
        return max_len
    return int(mismatches[0])


def build_entry_drafts(table: TableView) -> list[np.ndarray]:
    drafts: list[np.ndarray] = []
    for entry_id in range(table.n_entries):
        rollout_idx = int(table.entry_rollout_idx[entry_id])
        offset = int(table.entry_offset[entry_id])
        drafts.append(table.rollout_seqs[rollout_idx][offset:])
    return drafts


def update_window(wnd_size: int, accept_length: int, max_wnd: int, min_wnd: int) -> int:
    """Mirror hspec_proposer._CachedPromptTable.update_window()."""
    if accept_length >= wnd_size:
        return min(wnd_size + 1, max_wnd)
    if accept_length <= 1:
        return max(wnd_size // 2, min_wnd)
    return wnd_size


def simulate_rollout_current(
    response_tokens: np.ndarray,
    projected_hidden_states: np.ndarray,
    table: TableView,
    entry_drafts: list[np.ndarray],
) -> dict[str, Any]:
    seq = np.ascontiguousarray(response_tokens, dtype=np.int32)
    proj = np.ascontiguousarray(projected_hidden_states, dtype=np.float32)
    seq_len = len(seq)

    if seq_len <= 1:
        empty = np.empty((0,), dtype=np.int32)
        empty_f = np.empty((0,), dtype=np.float32)
        return {
            "num_match_steps": 0,
            "total_propose_steps": 0,
            "match_rate": 0.0,
            "pointer_positions": empty,
            "selected_entry_ids": empty,
            "selected_entry_rollout_idx": empty,
            "selected_entry_offset": empty,
            "selected_accept_lengths": empty,
            "selected_draft_lengths": empty,
            "selected_similarities": empty_f,
            "selected_keys": np.empty((0, table.keys.shape[1]), dtype=np.float32),
            "query_projected_hidden_states": np.empty((0, table.keys.shape[1]), dtype=np.float32),
            "selected_wnd_sizes": empty,
            "tie_counts": empty,
            "pointer_advance_to": empty,
            "skip_positions": empty,
            "total_optim_accept_length": 0,
            "avg_optim_accept_length": 0.0,
        }

    pointer_positions: list[int] = []
    selected_entry_ids: list[int] = []
    selected_entry_rollout_idx: list[int] = []
    selected_entry_offset: list[int] = []
    selected_accept_lengths: list[int] = []
    selected_draft_lengths: list[int] = []
    selected_similarities: list[float] = []
    selected_keys: list[np.ndarray] = []
    query_projected_hidden_states: list[np.ndarray] = []
    selected_wnd_sizes: list[int] = []
    tie_counts: list[int] = []
    pointer_advance_to: list[int] = []
    skip_positions: list[int] = []

    current_wnd_size = int(table.wnd_size)
    prev_accept_len = 1
    pointer = 0
    while pointer < seq_len - 1:
        continuation = seq[pointer + 1:]
        query_key = proj[pointer]
        current_wnd_size = update_window(
            current_wnd_size,
            prev_accept_len,
            table.max_wnd,
            table.min_wnd,
        )
        sims = table.keys @ query_key

        if sims.shape[0] == 0:
            skip_positions.append(pointer)
            prev_accept_len = 1
            pointer += 1
            continue

        best_entry_id = int(np.argmax(sims))
        best_similarity = float(sims[best_entry_id])
        tie_count = int(np.sum(sims == sims[best_entry_id]))

        full_draft = entry_drafts[best_entry_id]
        draft_len = min(len(full_draft), max(current_wnd_size, 0))
        if draft_len <= 0:
            skip_positions.append(pointer)
            prev_accept_len = 1
            pointer += 1
            continue

        draft = full_draft[:draft_len]
        accept_len = longest_exact_match(draft, continuation)

        if accept_len <= 0:
            skip_positions.append(pointer)
            prev_accept_len = 0
            pointer += 1
            continue

        pointer_positions.append(pointer)
        selected_entry_ids.append(best_entry_id)
        selected_entry_rollout_idx.append(int(table.entry_rollout_idx[best_entry_id]))
        selected_entry_offset.append(int(table.entry_offset[best_entry_id]))
        selected_accept_lengths.append(accept_len)
        selected_draft_lengths.append(draft_len)
        selected_similarities.append(best_similarity)
        selected_keys.append(np.ascontiguousarray(table.keys[best_entry_id], dtype=np.float32))
        query_projected_hidden_states.append(np.ascontiguousarray(query_key, dtype=np.float32))
        selected_wnd_sizes.append(current_wnd_size)
        tie_counts.append(tie_count)
        prev_accept_len = accept_len
        pointer = pointer + accept_len
        pointer_advance_to.append(pointer)

    total_accept = int(sum(selected_accept_lengths))
    avg_accept = (
        float(total_accept) / float(len(selected_accept_lengths))
        if selected_accept_lengths else 0.0
    )
    total_propose_steps = int(len(selected_entry_ids) + len(skip_positions))
    match_rate = (
        float(len(selected_entry_ids)) / float(total_propose_steps)
        if total_propose_steps > 0 else 0.0
    )

    key_matrix = (
        np.stack(selected_keys, axis=0)
        if selected_keys
        else np.empty((0, table.keys.shape[1]), dtype=np.float32)
    )
    query_matrix = (
        np.stack(query_projected_hidden_states, axis=0)
        if query_projected_hidden_states
        else np.empty((0, table.keys.shape[1]), dtype=np.float32)
    )

    return {
        "num_match_steps": int(len(selected_entry_ids)),
        "total_propose_steps": total_propose_steps,
        "match_rate": match_rate,
        "pointer_positions": np.asarray(pointer_positions, dtype=np.int32),
        "selected_entry_ids": np.asarray(selected_entry_ids, dtype=np.int32),
        "selected_entry_rollout_idx": np.asarray(selected_entry_rollout_idx, dtype=np.int32),
        "selected_entry_offset": np.asarray(selected_entry_offset, dtype=np.int32),
        "selected_accept_lengths": np.asarray(selected_accept_lengths, dtype=np.int32),
        "selected_draft_lengths": np.asarray(selected_draft_lengths, dtype=np.int32),
        "selected_similarities": np.asarray(selected_similarities, dtype=np.float32),
        "selected_keys": key_matrix,
        "query_projected_hidden_states": query_matrix,
        "selected_wnd_sizes": np.asarray(selected_wnd_sizes, dtype=np.int32),
        "tie_counts": np.asarray(tie_counts, dtype=np.int32),
        "pointer_advance_to": np.asarray(pointer_advance_to, dtype=np.int32),
        "skip_positions": np.asarray(skip_positions, dtype=np.int32),
        # Keep field names aligned with the optim script for downstream
        # comparison, even though this is the current strategy.
        "total_optim_accept_length": int(total_accept),
        "avg_optim_accept_length": float(avg_accept),
    }


def write_rollout_result(
    out_path: Path,
    meta: dict[str, Any],
    rollout_results: list[dict[str, Any]],
    response_tokens_list: list[np.ndarray],
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    total_num_match_steps = sum(int(r["num_match_steps"]) for r in rollout_results)
    total_propose_steps = sum(int(r["total_propose_steps"]) for r in rollout_results)
    match_rate = (
        float(total_num_match_steps) / float(total_propose_steps)
        if total_propose_steps > 0 else 0.0
    )

    payload: dict[str, Any] = {
        "strategy": np.asarray("current_hspec"),
        "prompt_id": np.asarray(meta["prompt_id"]),
        "epoch": np.asarray(meta["epoch"], dtype=np.int32),
        "global_step": np.asarray(meta["global_step"], dtype=np.int32),
        "active_table_version": np.asarray(meta["active_table_version"], dtype=np.int32),
        "prompt_token_ids": np.asarray(meta["prompt_token_ids"], dtype=np.int32),
        "table_path": np.asarray(str(meta["table_path"])),
        "rollout_input_path": np.asarray(str(meta["rollout_input_path"])),
        "table_n_entries": np.asarray(meta["table_n_entries"], dtype=np.int32),
        "table_wnd_size": np.asarray(meta["table_wnd_size"], dtype=np.int32),
        "table_max_wnd": np.asarray(meta["table_max_wnd"], dtype=np.int32),
        "table_min_wnd": np.asarray(meta["table_min_wnd"], dtype=np.int32),
        "rollout_count": np.asarray(len(rollout_results), dtype=np.int32),
        "total_num_match_steps": np.asarray(total_num_match_steps, dtype=np.int32),
        "total_propose_steps": np.asarray(total_propose_steps, dtype=np.int32),
        "match_rate": np.asarray(match_rate, dtype=np.float32),
        "response_tokens": np.empty((len(response_tokens_list),), dtype=object),
        "num_match_steps": np.empty((len(rollout_results),), dtype=np.int32),
        "total_propose_steps_per_rollout": np.empty((len(rollout_results),), dtype=np.int32),
        "match_rate_per_rollout": np.empty((len(rollout_results),), dtype=np.float32),
        "pointer_positions": np.empty((len(rollout_results),), dtype=object),
        "selected_entry_ids": np.empty((len(rollout_results),), dtype=object),
        "selected_entry_rollout_idx": np.empty((len(rollout_results),), dtype=object),
        "selected_entry_offset": np.empty((len(rollout_results),), dtype=object),
        "selected_accept_lengths": np.empty((len(rollout_results),), dtype=object),
        "selected_draft_lengths": np.empty((len(rollout_results),), dtype=object),
        "selected_similarities": np.empty((len(rollout_results),), dtype=object),
        "selected_keys": np.empty((len(rollout_results),), dtype=object),
        "query_projected_hidden_states": np.empty((len(rollout_results),), dtype=object),
        "selected_wnd_sizes": np.empty((len(rollout_results),), dtype=object),
        "tie_counts": np.empty((len(rollout_results),), dtype=object),
        "pointer_advance_to": np.empty((len(rollout_results),), dtype=object),
        "skip_positions": np.empty((len(rollout_results),), dtype=object),
        "total_optim_accept_length": np.empty((len(rollout_results),), dtype=np.int32),
        "avg_optim_accept_length": np.empty((len(rollout_results),), dtype=np.float32),
    }

    payload["response_tokens"][:] = response_tokens_list
    for i, result in enumerate(rollout_results):
        payload["num_match_steps"][i] = result["num_match_steps"]
        payload["total_propose_steps_per_rollout"][i] = result["total_propose_steps"]
        payload["match_rate_per_rollout"][i] = result["match_rate"]
        payload["pointer_positions"][i] = result["pointer_positions"]
        payload["selected_entry_ids"][i] = result["selected_entry_ids"]
        payload["selected_entry_rollout_idx"][i] = result["selected_entry_rollout_idx"]
        payload["selected_entry_offset"][i] = result["selected_entry_offset"]
        payload["selected_accept_lengths"][i] = result["selected_accept_lengths"]
        payload["selected_draft_lengths"][i] = result["selected_draft_lengths"]
        payload["selected_similarities"][i] = result["selected_similarities"]
        payload["selected_keys"][i] = result["selected_keys"]
        payload["query_projected_hidden_states"][i] = result["query_projected_hidden_states"]
        payload["selected_wnd_sizes"][i] = result["selected_wnd_sizes"]
        payload["tie_counts"][i] = result["tie_counts"]
        payload["pointer_advance_to"][i] = result["pointer_advance_to"]
        payload["skip_positions"][i] = result["skip_positions"]
        payload["total_optim_accept_length"][i] = result["total_optim_accept_length"]
        payload["avg_optim_accept_length"][i] = result["avg_optim_accept_length"]

    np.savez_compressed(out_path, **payload)


def make_summary_dict(
    *,
    prompts_simulated: int,
    rollouts_simulated: int,
    total_match_steps: int,
    total_propose_steps: int,
    total_accept_len: int,
    prompts_without_active_table: list[str],
    epoch: int | None = None,
    step: str | None = None,
) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "prompts_simulated": int(prompts_simulated),
        "rollouts_simulated": int(rollouts_simulated),
        "total_match_steps": int(total_match_steps),
        "total_propose_steps": int(total_propose_steps),
        "match_rate": (
            float(total_match_steps) / float(total_propose_steps)
            if total_propose_steps > 0 else 0.0
        ),
        "total_optim_accept_length": int(total_accept_len),
        "avg_optim_accept_length_per_match_step": (
            float(total_accept_len) / float(total_match_steps)
            if total_match_steps > 0 else 0.0
        ),
        "prompts_without_active_table": sorted(set(prompts_without_active_table)),
    }
    if epoch is not None:
        summary["epoch"] = int(epoch)
    if step is not None:
        summary["step"] = step
    return summary


def simulate_epoch(
    epoch: int,
    epoch_dir: Path,
    output_root: Path,
) -> dict[str, Any]:
    tables_dir = epoch_dir / "tables"
    rollouts_dir = epoch_dir / "rollouts"

    table_map: dict[str, TableView] = {}
    entry_drafts_map: dict[str, list[np.ndarray]] = {}
    if tables_dir.exists():
        for table_path in sorted(tables_dir.glob("*.npz")):
            table = load_table(table_path)
            table_map[table.prompt_id] = table
            entry_drafts_map[table.prompt_id] = build_entry_drafts(table)

    epoch_out_dir = output_root / f"epoch_{epoch:04d}"
    epoch_out_dir.mkdir(parents=True, exist_ok=True)
    decisions_dir = epoch_out_dir / "current_decisions"
    decisions_dir.mkdir(parents=True, exist_ok=True)
    step_summaries_dir = epoch_out_dir / "step_summaries"
    step_summaries_dir.mkdir(parents=True, exist_ok=True)

    rollout_files = sorted(rollouts_dir.glob("step_*/*.npz")) if rollouts_dir.exists() else []
    prompts_simulated = 0
    rollouts_simulated = 0
    total_match_steps = 0
    total_propose_steps = 0
    total_accept_len = 0
    prompts_without_table: list[str] = []
    step_aggregates: dict[str, dict[str, Any]] = {}

    for rollout_path in rollout_files:
        step_name = rollout_path.parent.name
        if step_name not in step_aggregates:
            step_aggregates[step_name] = {
                "prompts_simulated": 0,
                "rollouts_simulated": 0,
                "total_match_steps": 0,
                "total_propose_steps": 0,
                "total_accept_len": 0,
                "prompts_without_active_table": [],
            }
        step_agg = step_aggregates[step_name]

        rollout_data = load_npz(rollout_path)
        prompt_id = str(decode_scalar(rollout_data["prompt_id"]))
        table_present = bool(decode_scalar(rollout_data["table_present"]))
        if not table_present or prompt_id not in table_map:
            prompts_without_table.append(prompt_id)
            step_agg["prompts_without_active_table"].append(prompt_id)
            continue

        table = table_map[prompt_id]
        entry_drafts = entry_drafts_map[prompt_id]

        response_tokens_obj = rollout_data["response_tokens"].tolist()
        projected_hidden_states_obj = rollout_data["projected_hidden_states"].tolist()
        projection_available = np.asarray(
            rollout_data["projection_available"], dtype=np.bool_,
        )

        rollout_results: list[dict[str, Any]] = []
        response_tokens_list: list[np.ndarray] = []
        for rollout_idx, response_tokens in enumerate(response_tokens_obj):
            if not bool(projection_available[rollout_idx]):
                continue
            projected_hs = projected_hidden_states_obj[rollout_idx]
            if projected_hs is None:
                continue

            response_np = np.ascontiguousarray(np.asarray(response_tokens, dtype=np.int32))
            projected_np = np.ascontiguousarray(np.asarray(projected_hs, dtype=np.float32))
            result = simulate_rollout_current(
                response_tokens=response_np,
                projected_hidden_states=projected_np,
                table=table,
                entry_drafts=entry_drafts,
            )
            rollout_results.append(result)
            response_tokens_list.append(response_np)
            rollouts_simulated += 1
            total_match_steps += int(decode_scalar(result["num_match_steps"]))
            total_propose_steps += int(decode_scalar(result["total_propose_steps"]))
            total_accept_len += int(decode_scalar(result["total_optim_accept_length"]))
            step_agg["rollouts_simulated"] += 1
            step_agg["total_match_steps"] += int(decode_scalar(result["num_match_steps"]))
            step_agg["total_propose_steps"] += int(decode_scalar(result["total_propose_steps"]))
            step_agg["total_accept_len"] += int(decode_scalar(result["total_optim_accept_length"]))

        if not rollout_results:
            continue

        prompts_simulated += 1
        step_agg["prompts_simulated"] += 1
        rel_parent = rollout_path.parent.name
        out_path = decisions_dir / rel_parent / rollout_path.name
        write_rollout_result(
            out_path=out_path,
            meta={
                "prompt_id": prompt_id,
                "epoch": epoch,
                "global_step": int(decode_scalar(rollout_data["global_step"])),
                "active_table_version": int(decode_scalar(rollout_data["active_table_version"])),
                "prompt_token_ids": np.asarray(
                    rollout_data["prompt_token_ids"], dtype=np.int32,
                ),
                "table_path": tables_dir / f"{prompt_id}.npz",
                "rollout_input_path": rollout_path,
                "table_n_entries": table.n_entries,
                "table_wnd_size": table.wnd_size,
                "table_max_wnd": table.max_wnd,
                "table_min_wnd": table.min_wnd,
            },
            rollout_results=rollout_results,
            response_tokens_list=response_tokens_list,
        )

    for step_name, agg in step_aggregates.items():
        step_summary = make_summary_dict(
            epoch=epoch,
            step=step_name,
            prompts_simulated=agg["prompts_simulated"],
            rollouts_simulated=agg["rollouts_simulated"],
            total_match_steps=agg["total_match_steps"],
            total_propose_steps=agg["total_propose_steps"],
            total_accept_len=agg["total_accept_len"],
            prompts_without_active_table=agg["prompts_without_active_table"],
        )
        with open(step_summaries_dir / f"{step_name}.json", "w", encoding="utf-8") as f:
            json.dump(step_summary, f, ensure_ascii=False, indent=2)

    summary = make_summary_dict(
        epoch=epoch,
        prompts_simulated=prompts_simulated,
        rollouts_simulated=rollouts_simulated,
        total_match_steps=total_match_steps,
        total_propose_steps=total_propose_steps,
        total_accept_len=total_accept_len,
        prompts_without_active_table=prompts_without_table,
    )

    with open(epoch_out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    return summary


def main() -> None:
    args = parse_args()
    dump_root = args.dump_root
    output_root = args.output_root

    if not dump_root.exists():
        raise FileNotFoundError(f"Dump root does not exist: {dump_root}")
    ensure_output_dir(output_root, overwrite=args.overwrite)

    all_summaries: list[dict[str, Any]] = []
    overall_prompts_simulated = 0
    overall_rollouts_simulated = 0
    overall_total_match_steps = 0
    overall_total_propose_steps = 0
    overall_total_accept_len = 0
    for epoch, epoch_dir in iter_epoch_dirs(dump_root, args.epoch_start):
        summary = simulate_epoch(epoch, epoch_dir, output_root)
        all_summaries.append(summary)
        overall_prompts_simulated += int(summary["prompts_simulated"])
        overall_rollouts_simulated += int(summary["rollouts_simulated"])
        overall_total_match_steps += int(summary["total_match_steps"])
        overall_total_propose_steps += int(summary["total_propose_steps"])
        overall_total_accept_len += int(summary["total_optim_accept_length"])
        print(
            f"[current] epoch={epoch} prompts={summary['prompts_simulated']} "
            f"rollouts={summary['rollouts_simulated']} "
            f"match_steps={summary['total_match_steps']} "
            f"match_rate={summary['match_rate']:.4f} "
            f"avg_accept={summary['avg_optim_accept_length_per_match_step']:.4f}",
        )

    overall_summary = {
        "prompts_simulated": overall_prompts_simulated,
        "rollouts_simulated": overall_rollouts_simulated,
        "total_match_steps": overall_total_match_steps,
        "total_propose_steps": overall_total_propose_steps,
        "match_rate": (
            float(overall_total_match_steps) / float(overall_total_propose_steps)
            if overall_total_propose_steps > 0 else 0.0
        ),
        "total_optim_accept_length": overall_total_accept_len,
        "avg_optim_accept_length_per_match_step": (
            float(overall_total_accept_len) / float(overall_total_match_steps)
            if overall_total_match_steps > 0 else 0.0
        ),
    }
    with open(output_root / "summary_all_epochs.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "overall": overall_summary,
                "epochs": all_summaries,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )


if __name__ == "__main__":
    main()
