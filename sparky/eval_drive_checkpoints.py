#!/usr/bin/env python3
"""Evaluate Synapse checkpoints from Google Drive and write graph-ready outputs."""

import argparse
import csv
import datetime as dt
import hashlib
import json
import math
import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import random
import re
import shutil
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from lm_eval import simple_evaluate

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from sparky_eval import TASK_PRESETS, SynapseEvalLM  # noqa: E402
from sparky_model import BLOCK_SIZE, VOCAB_SIZE  # noqa: E402


DEFAULT_WORK_DIR = "/media/alfonso/shared/synapse_drive_eval"
DEFAULT_REMOTE = "gdrive"
DEFAULT_DRIVE_PATH = "synapse"
DEFAULT_PRESET = "quick"
DEFAULT_BATCH_SIZE = 1
DEFAULT_EVAL_BATCHES = 64
DEFAULT_SOURCE_EVAL_BATCHES = 8
DEFAULT_EVAL_SEED = 1337
DEFAULT_MIN_CHECKPOINT_SIZE_GB = 5.0
EVAL_FRACTION_PER_SOURCE = 0.02
TOKENS_PER_STEP_ESTIMATE = 4 * 64 * BLOCK_SIZE
QUICK_PRIMARY_METRICS = {
    "anli_r1": "acc",
    "boolq": "acc",
    "piqa": "acc_norm",
    "sciq": "acc_norm",
    "openbookqa": "acc_norm",
}
DTYPE_MAP = {
    "uint8": np.uint8,
    "uint16": np.uint16,
    "uint32": np.uint32,
    "int8": np.int8,
    "int16": np.int16,
    "int32": np.int32,
    "int64": np.int64,
}


class ShardDataset(Dataset):
    def __init__(self, tokens, block_size=BLOCK_SIZE, stride=BLOCK_SIZE):
        self.tokens = tokens
        self.block_size = block_size
        self.stride = stride
        self.length = max(0, (len(tokens) - block_size - 1) // stride)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        start = idx * self.stride
        if start + self.block_size + 1 > len(self.tokens):
            start = len(self.tokens) - self.block_size - 1
        chunk = self.tokens[start : start + self.block_size + 1]
        return torch.from_numpy(chunk[:-1].copy()), torch.from_numpy(chunk[1:].copy())


def run(cmd, capture=True):
    result = subprocess.run(
        cmd,
        text=True,
        check=True,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.PIPE if capture else None,
    )
    return result.stdout if capture else ""


def rclone_remote(args):
    return f"{args.remote}:{args.drive_path}"


def rclone_copyto(src, dst):
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    run([
        "rclone",
        "copyto",
        src,
        str(dst),
        "--progress",
        "--drive-chunk-size",
        "64M",
        "--transfers",
        "4",
        "--retries",
        "5",
        "--low-level-retries",
        "20",
    ], capture=False)


def rclone_lsjson(remote_dir, include_glob):
    out = run([
        "rclone",
        "lsjson",
        remote_dir,
        "--include",
        include_glob,
        "--no-modtime",
        "--no-mimetype",
    ])
    return json.loads(out)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024 * 16), b""):
            h.update(chunk)
    return h.hexdigest()


def now_iso():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def parse_timestamp_from_name(name):
    match = re.search(r"_(20\d{6})_(\d{6})\.pth$", name)
    if not match:
        return ""
    date_part, time_part = match.groups()
    parsed = dt.datetime.strptime(date_part + time_part, "%Y%m%d%H%M%S")
    return parsed.isoformat()


def sanitize_checkpoint_id(name):
    return Path(name).stem.replace(" ", "_")


def get_source_name(shard_entry):
    if shard_entry.get("domain"):
        return shard_entry["domain"]
    source = shard_entry.get("source", "")
    for part in source.replace("\\", "/").split("/"):
        if part.startswith("data_"):
            return part
    return "other"


def load_json(path):
    with open(path) as f:
        return json.load(f)


def save_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)


def ensure_assets(args, dirs):
    assets = dirs["assets"]
    tokenizer = assets / "tokenizer.json"
    meta = assets / "meta.json"
    manifest = assets / "shard_manifest.json"
    remote_root = rclone_remote(args)
    downloads = [
        (f"{remote_root}/tokenizer_out/tokenizer.json", tokenizer),
        (f"{remote_root}/token_shards_merged/meta.json", meta),
        (f"{remote_root}/token_shards_merged/shard_manifest.json", manifest),
    ]
    for src, dst in downloads:
        if not dst.exists() or args.refresh_assets:
            print(f"[asset] downloading {src}")
            rclone_copyto(src, dst)
    return tokenizer, meta, manifest


def select_eval_shards(shard_manifest):
    shards_by_source = defaultdict(list)
    for shard in shard_manifest["shards"]:
        shards_by_source[get_source_name(shard)].append(shard)

    rng = random.Random(DEFAULT_EVAL_SEED)
    eval_shards = []
    eval_by_source = {}
    for source, shards in shards_by_source.items():
        pool = shards.copy()
        rng.shuffle(pool)
        n_eval = max(1, int(len(pool) * EVAL_FRACTION_PER_SOURCE)) if len(pool) > 1 else 0
        selected = pool[:n_eval]
        eval_shards.extend(selected)
        eval_by_source[source] = selected
    return eval_shards, eval_by_source


def ensure_shard(args, dirs, shard_name):
    local = dirs["shards"] / shard_name
    if local.exists():
        return local
    src = f"{rclone_remote(args)}/token_shards_merged/{shard_name}"
    print(f"[shard] downloading {shard_name}")
    rclone_copyto(src, local)
    return local


@torch.no_grad()
def eval_loss_for_shards(model, shard_entries, args, dirs, shard_dtype, max_batches):
    device = torch.device(args.device)
    losses = []
    total_loss_tokens = 0
    total_loss_sum = 0.0
    batches_seen = 0

    if max_batches <= 0 or not shard_entries:
        return {
            "loss": None,
            "ppl": None,
            "num_batches": 0,
            "num_tokens_eval": 0,
            "num_shards_used": 0,
        }

    used_shards = 0
    model.eval()
    for shard_info in shard_entries:
        shard_path = ensure_shard(args, dirs, shard_info["shard"])
        tokens = np.fromfile(shard_path, dtype=shard_dtype).astype(np.int64)
        dataset = ShardDataset(tokens)
        if len(dataset) == 0:
            continue
        used_shards += 1
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            autocast_enabled = device.type == "cuda"
            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=autocast_enabled):
                logits, _ = model(xb, kv_cache=None, pos_offset=0)
                loss = F.cross_entropy(logits.reshape(-1, VOCAB_SIZE), yb.reshape(-1))
            n_tokens = yb.numel()
            loss_value = float(loss.item())
            losses.append(loss_value)
            total_loss_sum += loss_value * n_tokens
            total_loss_tokens += n_tokens
            batches_seen += 1
            del xb, yb, logits, loss
            if device.type == "cuda":
                torch.cuda.empty_cache()
            if batches_seen >= max_batches:
                break
        del tokens, dataset, loader
        if batches_seen >= max_batches:
            break

    mean_loss = total_loss_sum / total_loss_tokens if total_loss_tokens else None
    return {
        "loss": mean_loss,
        "ppl": math.exp(mean_loss) if mean_loss is not None and mean_loss < 100 else None,
        "num_batches": batches_seen,
        "num_tokens_eval": total_loss_tokens,
        "num_shards_used": used_shards,
        "batch_losses": losses,
    }


def metric_value(task_result, metric):
    return task_result.get(f"{metric},none")


def metric_higher_is_better(lm_results, task, metric):
    hib = lm_results.get("higher_is_better", {})
    return hib.get(task, {}).get(metric)


def flatten_benchmark_metrics(lm_results, checkpoint_row, args):
    rows = []
    for task, task_result in sorted(lm_results.get("results", {}).items()):
        sample_len = task_result.get("sample_len")
        for key, value in task_result.items():
            if "," not in key or key.endswith("_stderr,none"):
                continue
            metric, _ = key.split(",", 1)
            if not isinstance(value, (int, float)):
                continue
            rows.append({
                "checkpoint_id": checkpoint_row["checkpoint_id"],
                "checkpoint_step": checkpoint_row["checkpoint_step"],
                "checkpoint_timestamp_from_name": checkpoint_row["checkpoint_timestamp_from_name"],
                "metric_group": "benchmark",
                "task": task,
                "metric": metric,
                "value": value,
                "higher_is_better": metric_higher_is_better(lm_results, task, metric),
                "sample_len": sample_len,
                "num_fewshot": args.num_fewshot,
                "preset": args.preset,
            })
    return rows


def build_wide_row(checkpoint_row, lm_results, global_loss):
    row = {
        "checkpoint_id": checkpoint_row["checkpoint_id"],
        "checkpoint_step": checkpoint_row["checkpoint_step"],
        "checkpoint_timestamp_from_name": checkpoint_row["checkpoint_timestamp_from_name"],
        "heldout_loss": global_loss.get("loss"),
        "heldout_ppl": global_loss.get("ppl"),
        "anli_r1_acc": None,
        "boolq_acc": None,
        "piqa_acc": None,
        "piqa_acc_norm": None,
        "sciq_acc": None,
        "sciq_acc_norm": None,
        "openbookqa_acc": None,
        "openbookqa_acc_norm": None,
        "quick_mean_primary": None,
        "quick_mean_raw_acc": None,
    }
    task_results = lm_results.get("results", {})
    for task in ["anli_r1", "boolq", "piqa", "sciq", "openbookqa"]:
        result = task_results.get(task, {})
        for metric in ["acc", "acc_norm"]:
            value = metric_value(result, metric)
            key = f"{task}_{metric}"
            if key in row and isinstance(value, (int, float)):
                row[key] = value

    primary = []
    for task, metric in QUICK_PRIMARY_METRICS.items():
        value = metric_value(task_results.get(task, {}), metric)
        if isinstance(value, (int, float)):
            primary.append(value)
    row["quick_mean_primary"] = sum(primary) / len(primary) if primary else None

    raw_acc = []
    for task in ["anli_r1", "boolq", "piqa", "sciq", "openbookqa"]:
        value = metric_value(task_results.get(task, {}), "acc")
        if isinstance(value, (int, float)):
            raw_acc.append(value)
    row["quick_mean_raw_acc"] = sum(raw_acc) / len(raw_acc) if raw_acc else None
    return row


def write_csv(path, rows, fieldnames):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def sort_key(row):
    step = row.get("checkpoint_step")
    if step not in (None, ""):
        try:
            return (0, int(step))
        except ValueError:
            pass
    ts = row.get("checkpoint_timestamp_from_name") or ""
    return (1, ts, row.get("checkpoint_id") or "")


def write_summary_md(path, wide_rows, checkpoint_rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = sorted(wide_rows, key=sort_key)
    ckpt_by_id = {r["checkpoint_id"]: r for r in checkpoint_rows}
    lines = [
        "# Synapse Drive Eval Summary",
        "",
        f"Generated: {now_iso()}",
        "",
        "| Checkpoint | Step | Heldout Loss | Delta Loss | Heldout PPL | Quick Mean | Delta Quick | Best Loss | Best Quick | Status |",
        "|---|---:|---:|---:|---:|---:|---:|---|---|---|",
    ]
    for row in rows:
        ckpt = ckpt_by_id.get(row["checkpoint_id"], {})
        lines.append(
            "| {checkpoint_id} | {step} | {loss} | {loss_delta} | {ppl} | {quick} | {quick_delta} | {best_loss} | {best_quick} | {status} |".format(
                checkpoint_id=row["checkpoint_id"],
                step=row.get("checkpoint_step") or "",
                loss=fmt_num(row.get("heldout_loss")),
                loss_delta=fmt_num(row.get("heldout_loss_delta_prev")),
                ppl=fmt_num(row.get("heldout_ppl")),
                quick=fmt_num(row.get("quick_mean_primary")),
                quick_delta=fmt_num(row.get("quick_mean_primary_delta_prev")),
                best_loss="yes" if str(row.get("is_best_heldout_loss")).lower() == "true" else "",
                best_quick="yes" if str(row.get("is_best_quick_mean")).lower() == "true" else "",
                status=ckpt.get("status", ""),
            )
        )
    path.write_text("\n".join(lines) + "\n")


def fmt_num(value):
    if value in (None, ""):
        return ""
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return str(value)


def read_existing_csv(path):
    path = Path(path)
    if not path.exists():
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def as_float(value):
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def processed_checkpoint_ids(results_dir, retry_failed=False):
    path = results_dir / "checkpoints.csv"
    processed = set()
    for row in read_existing_csv(path):
        checkpoint_id = row.get("checkpoint_id")
        status = row.get("status")
        if not checkpoint_id or not status:
            continue
        if status == "failed" and retry_failed:
            continue
        processed.add(checkpoint_id)
    return processed


def replace_rows(rows, key_name, key_value, new_rows):
    return [r for r in rows if r.get(key_name) != key_value] + new_rows


def copy_repo_summaries(results_dir):
    repo_dir = SCRIPT_DIR / "results" / "drive_eval"
    repo_dir.mkdir(parents=True, exist_ok=True)
    src_csv = results_dir / "metrics_wide.csv"
    src_md = results_dir / "summary.md"
    if src_csv.exists():
        shutil.copy2(src_csv, repo_dir / "summary.csv")
    if src_md.exists():
        shutil.copy2(src_md, repo_dir / "summary.md")


def write_all_outputs(results_dir, checkpoint_rows, metric_rows, wide_rows, source_rows, run_rows):
    checkpoint_fields = [
        "checkpoint_id",
        "checkpoint_name",
        "drive_path",
        "local_path",
        "file_size_bytes",
        "sha256",
        "checkpoint_step",
        "checkpoint_timestamp_from_name",
        "stored_last_eval_loss",
        "stored_eval_history_last_step",
        "stored_eval_history_last_loss",
        "tokens_seen_estimate",
        "status",
        "error",
        "download_time_s",
        "load_time_s",
        "quick_eval_time_s",
        "heldout_eval_time_s",
        "total_time_s",
        "gpu_name",
        "vram_mb",
    ]
    metric_fields = [
        "checkpoint_id",
        "checkpoint_step",
        "checkpoint_timestamp_from_name",
        "metric_group",
        "task",
        "metric",
        "value",
        "higher_is_better",
        "sample_len",
        "num_fewshot",
        "preset",
    ]
    wide_fields = [
        "checkpoint_id",
        "checkpoint_step",
        "checkpoint_timestamp_from_name",
        "heldout_loss",
        "heldout_ppl",
        "heldout_loss_delta_prev",
        "heldout_ppl_delta_prev",
        "anli_r1_acc",
        "boolq_acc",
        "piqa_acc",
        "piqa_acc_norm",
        "sciq_acc",
        "sciq_acc_norm",
        "openbookqa_acc",
        "openbookqa_acc_norm",
        "quick_mean_primary",
        "quick_mean_raw_acc",
        "quick_mean_primary_delta_prev",
        "quick_mean_raw_acc_delta_prev",
        "is_best_heldout_loss",
        "is_best_quick_mean",
    ]
    source_fields = [
        "checkpoint_id",
        "checkpoint_step",
        "source",
        "loss",
        "ppl",
        "num_batches",
        "num_tokens_eval",
    ]
    run_fields = [
        "timestamp",
        "checkpoint_id",
        "event",
        "message",
        "elapsed_s",
    ]
    checkpoint_rows = sorted(checkpoint_rows, key=sort_key)
    wide_rows = enrich_wide_rows(sorted(wide_rows, key=sort_key))
    metric_rows = sorted(metric_rows, key=lambda r: (sort_key(r), r.get("task") or "", r.get("metric") or ""))
    source_rows = sorted(source_rows, key=lambda r: (sort_key(r), r.get("source") or ""))
    write_csv(results_dir / "checkpoints.csv", checkpoint_rows, checkpoint_fields)
    write_csv(results_dir / "metrics_long.csv", metric_rows, metric_fields)
    write_csv(results_dir / "metrics_wide.csv", wide_rows, wide_fields)
    write_csv(results_dir / "eval_loss_by_source.csv", source_rows, source_fields)
    write_csv(results_dir / "run_log.csv", run_rows, run_fields)
    write_summary_md(results_dir / "summary.md", wide_rows, checkpoint_rows)
    copy_repo_summaries(results_dir)


def enrich_wide_rows(rows):
    prev_loss = None
    prev_ppl = None
    prev_quick = None
    prev_raw = None
    best_loss = None
    best_quick = None
    enriched = []
    for row in rows:
        row = dict(row)
        loss = as_float(row.get("heldout_loss"))
        ppl = as_float(row.get("heldout_ppl"))
        quick = as_float(row.get("quick_mean_primary"))
        raw = as_float(row.get("quick_mean_raw_acc"))

        row["heldout_loss_delta_prev"] = loss - prev_loss if loss is not None and prev_loss is not None else None
        row["heldout_ppl_delta_prev"] = ppl - prev_ppl if ppl is not None and prev_ppl is not None else None
        row["quick_mean_primary_delta_prev"] = quick - prev_quick if quick is not None and prev_quick is not None else None
        row["quick_mean_raw_acc_delta_prev"] = raw - prev_raw if raw is not None and prev_raw is not None else None

        row["is_best_heldout_loss"] = loss is not None and (best_loss is None or loss <= best_loss)
        row["is_best_quick_mean"] = quick is not None and (best_quick is None or quick >= best_quick)

        if loss is not None:
            best_loss = loss if best_loss is None else min(best_loss, loss)
            prev_loss = loss
        if ppl is not None:
            prev_ppl = ppl
        if quick is not None:
            best_quick = quick if best_quick is None else max(best_quick, quick)
            prev_quick = quick
        if raw is not None:
            prev_raw = raw

        enriched.append(row)
    return enriched


def checkpoint_info_from_lm(lm):
    info = lm._info or {}
    history = info.get("eval_history") or []
    last_history = history[-1] if history else {}
    step = info.get("step")
    return {
        "checkpoint_step": step,
        "stored_last_eval_loss": info.get("last_eval_loss"),
        "stored_eval_history_last_step": last_history.get("step"),
        "stored_eval_history_last_loss": last_history.get("loss"),
        "tokens_seen_estimate": step * TOKENS_PER_STEP_ESTIMATE if isinstance(step, int) else None,
    }


def load_existing_outputs(results_dir):
    return {
        "checkpoint_rows": read_existing_csv(results_dir / "checkpoints.csv"),
        "metric_rows": read_existing_csv(results_dir / "metrics_long.csv"),
        "wide_rows": read_existing_csv(results_dir / "metrics_wide.csv"),
        "source_rows": read_existing_csv(results_dir / "eval_loss_by_source.csv"),
        "run_rows": read_existing_csv(results_dir / "run_log.csv"),
    }


def log_event(run_rows, checkpoint_id, event, message, elapsed_s=None):
    run_rows.append({
        "timestamp": now_iso(),
        "checkpoint_id": checkpoint_id,
        "event": event,
        "message": message,
        "elapsed_s": elapsed_s,
    })


def discover_checkpoints(args):
    remote_dir = f"{rclone_remote(args)}/checkpoints"
    entries = rclone_lsjson(remote_dir, args.checkpoint_glob)
    checkpoints = []
    for entry in entries:
        if entry.get("IsDir"):
            continue
        name = entry["Name"]
        if not name.endswith(".pth"):
            continue
        checkpoints.append({
            "name": name,
            "path": f"{remote_dir}/{entry['Path']}",
            "size": entry.get("Size"),
            "checkpoint_id": sanitize_checkpoint_id(name),
            "timestamp": parse_timestamp_from_name(name),
        })
    return sorted(checkpoints, key=lambda c: (c["timestamp"] or "", c["name"]))


def min_checkpoint_bytes(args):
    if args.min_checkpoint_size_gb <= 0:
        return 0
    return int(args.min_checkpoint_size_gb * 1024**3)


def skipped_small_checkpoint_row(args, checkpoint):
    size = checkpoint.get("size") or 0
    return {
        "checkpoint_id": checkpoint["checkpoint_id"],
        "checkpoint_name": checkpoint["name"],
        "drive_path": checkpoint["path"],
        "local_path": "",
        "file_size_bytes": size,
        "sha256": "",
        "checkpoint_step": "",
        "checkpoint_timestamp_from_name": checkpoint["timestamp"],
        "stored_last_eval_loss": "",
        "stored_eval_history_last_step": "",
        "stored_eval_history_last_loss": "",
        "tokens_seen_estimate": "",
        "status": "skipped_small",
        "error": f"checkpoint is {size / 1024**3:.2f} GiB; below --min-checkpoint-size-gb {args.min_checkpoint_size_gb:g}",
        "download_time_s": "",
        "load_time_s": "",
        "quick_eval_time_s": "",
        "heldout_eval_time_s": "",
        "total_time_s": 0,
        "gpu_name": "",
        "vram_mb": "",
    }


def download_checkpoint(args, dirs, checkpoint):
    local = dirs["checkpoints"] / checkpoint["name"]
    if local.exists() and local.stat().st_size == checkpoint.get("size"):
        return local, 0.0
    started = time.time()
    print(f"[checkpoint] downloading {checkpoint['path']}")
    rclone_copyto(checkpoint["path"], local)
    return local, time.time() - started


def evaluate_checkpoint(args, dirs, checkpoint, tokenizer_path, meta, shard_manifest):
    checkpoint_id = checkpoint["checkpoint_id"]
    started_total = time.time()
    local_ckpt = None
    lm = None
    device = torch.device(args.device)

    try:
        local_ckpt, download_time = download_checkpoint(args, dirs, checkpoint)
        file_sha = sha256_file(local_ckpt)

        print(f"[eval] loading {checkpoint['name']}")
        load_started = time.time()
        lm = SynapseEvalLM(
            str(local_ckpt),
            str(tokenizer_path),
            device=args.device,
            no_compile=args.no_compile,
            max_batch_tokens=args.max_batch_tokens,
        )
        load_time = time.time() - load_started

        info = checkpoint_info_from_lm(lm)
        gpu_name = None
        vram_mb = None
        if device.type == "cuda":
            gpu_name = torch.cuda.get_device_name(0)
            vram_mb = torch.cuda.get_device_properties(0).total_memory / 1024**2

        tasks = TASK_PRESETS[args.preset]
        print(f"[eval] quick benchmark for {checkpoint_id}: {', '.join(tasks)}")
        quick_started = time.time()
        lm_results = simple_evaluate(
            model=lm,
            tasks=tasks,
            num_fewshot=args.num_fewshot,
            limit=args.limit,
            bootstrap_iters=0,
            log_samples=False,
            confirm_run_unsafe_code=True,
        )
        quick_time = time.time() - quick_started
        if lm_results is None:
            raise RuntimeError("lm-eval returned no results")

        eval_shards, eval_by_source = select_eval_shards(shard_manifest)
        rng = random.Random(DEFAULT_EVAL_SEED)
        global_pool = eval_shards.copy()
        rng.shuffle(global_pool)
        shard_dtype_name = meta.get("shard_dtype", "uint16")
        if shard_dtype_name not in DTYPE_MAP:
            raise RuntimeError(f"Unsupported shard dtype: {shard_dtype_name}")
        shard_dtype = DTYPE_MAP[shard_dtype_name]

        print(f"[eval] heldout loss for {checkpoint_id}: {args.eval_batches} batches")
        loss_started = time.time()
        global_loss = eval_loss_for_shards(lm.model, global_pool, args, dirs, shard_dtype, args.eval_batches)
        source_losses = {}
        for source, source_shards in sorted(eval_by_source.items()):
            source_pool = source_shards.copy()
            random.Random(DEFAULT_EVAL_SEED).shuffle(source_pool)
            source_losses[source] = eval_loss_for_shards(
                lm.model,
                source_pool,
                args,
                dirs,
                shard_dtype,
                args.source_eval_batches,
            )
        loss_time = time.time() - loss_started

        checkpoint_row = {
            "checkpoint_id": checkpoint_id,
            "checkpoint_name": checkpoint["name"],
            "drive_path": checkpoint["path"],
            "local_path": str(local_ckpt),
            "file_size_bytes": local_ckpt.stat().st_size,
            "sha256": file_sha,
            "checkpoint_timestamp_from_name": checkpoint["timestamp"],
            "status": "ok",
            "error": "",
            "download_time_s": round(download_time, 3),
            "load_time_s": round(load_time, 3),
            "quick_eval_time_s": round(quick_time, 3),
            "heldout_eval_time_s": round(loss_time, 3),
            "total_time_s": round(time.time() - started_total, 3),
            "gpu_name": gpu_name,
            "vram_mb": vram_mb,
            **info,
        }
        metric_rows = flatten_benchmark_metrics(lm_results, checkpoint_row, args)
        metric_rows.extend([
            {
                "checkpoint_id": checkpoint_id,
                "checkpoint_step": checkpoint_row["checkpoint_step"],
                "checkpoint_timestamp_from_name": checkpoint_row["checkpoint_timestamp_from_name"],
                "metric_group": "loss",
                "task": "heldout",
                "metric": "loss",
                "value": global_loss.get("loss"),
                "higher_is_better": False,
                "sample_len": global_loss.get("num_batches"),
                "num_fewshot": args.num_fewshot,
                "preset": "evalloss",
            },
            {
                "checkpoint_id": checkpoint_id,
                "checkpoint_step": checkpoint_row["checkpoint_step"],
                "checkpoint_timestamp_from_name": checkpoint_row["checkpoint_timestamp_from_name"],
                "metric_group": "loss",
                "task": "heldout",
                "metric": "ppl",
                "value": global_loss.get("ppl"),
                "higher_is_better": False,
                "sample_len": global_loss.get("num_batches"),
                "num_fewshot": args.num_fewshot,
                "preset": "evalloss",
            },
        ])
        wide_row = build_wide_row(checkpoint_row, lm_results, global_loss)
        source_rows = []
        for source, loss_info in source_losses.items():
            source_rows.append({
                "checkpoint_id": checkpoint_id,
                "checkpoint_step": checkpoint_row["checkpoint_step"],
                "source": source,
                "loss": loss_info.get("loss"),
                "ppl": loss_info.get("ppl"),
                "num_batches": loss_info.get("num_batches"),
                "num_tokens_eval": loss_info.get("num_tokens_eval"),
            })

        save_json(dirs["results"] / f"{checkpoint_id}_{args.preset}.json", {
            "config": {
                "checkpoint": checkpoint_row,
                "preset": args.preset,
                "tasks": tasks,
                "num_fewshot": args.num_fewshot,
                "limit": args.limit,
                "max_batch_tokens": args.max_batch_tokens,
            },
            "results": lm_results,
        })
        save_json(dirs["results"] / f"{checkpoint_id}_evalloss.json", {
            "config": {
                "eval_batches": args.eval_batches,
                "source_eval_batches": args.source_eval_batches,
                "batch_size": args.batch_size,
                "eval_seed": DEFAULT_EVAL_SEED,
                "eval_fraction_per_source": EVAL_FRACTION_PER_SOURCE,
            },
            "global": global_loss,
            "by_source": source_losses,
        })
        return checkpoint_row, metric_rows, wide_row, source_rows
    except Exception as exc:
        checkpoint_row = {
            "checkpoint_id": checkpoint_id,
            "checkpoint_name": checkpoint["name"],
            "drive_path": checkpoint["path"],
            "local_path": str(local_ckpt) if local_ckpt else "",
            "file_size_bytes": checkpoint.get("size"),
            "sha256": "",
            "checkpoint_step": "",
            "checkpoint_timestamp_from_name": checkpoint["timestamp"],
            "stored_last_eval_loss": "",
            "stored_eval_history_last_step": "",
            "stored_eval_history_last_loss": "",
            "tokens_seen_estimate": "",
            "status": "failed",
            "error": repr(exc),
            "download_time_s": "",
            "load_time_s": "",
            "quick_eval_time_s": "",
            "heldout_eval_time_s": "",
            "total_time_s": round(time.time() - started_total, 3),
            "gpu_name": "",
            "vram_mb": "",
        }
        return checkpoint_row, [], None, []
    finally:
        del lm
        if device.type == "cuda":
            torch.cuda.empty_cache()
        if local_ckpt and local_ckpt.exists() and not args.keep_checkpoints:
            try:
                local_ckpt.unlink()
            except OSError as exc:
                print(f"[warn] could not delete {local_ckpt}: {exc}")


def build_dirs(work_dir):
    root = Path(work_dir).expanduser().resolve()
    dirs = {
        "root": root,
        "checkpoints": root / "checkpoints",
        "assets": root / "assets",
        "tmp": root / "tmp",
        "results": root / "results",
        "shards": root / "assets" / "eval_shards",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--remote", default=DEFAULT_REMOTE)
    parser.add_argument("--drive-path", default=DEFAULT_DRIVE_PATH)
    parser.add_argument("--work-dir", default=DEFAULT_WORK_DIR)
    parser.add_argument("--checkpoint-glob", default="*.pth")
    parser.add_argument("--preset", choices=sorted(TASK_PRESETS), default=DEFAULT_PRESET)
    parser.add_argument("--eval-batches", type=int, default=DEFAULT_EVAL_BATCHES)
    parser.add_argument("--source-eval-batches", type=int, default=DEFAULT_SOURCE_EVAL_BATCHES)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num_fewshot", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-batch-tokens", type=int, default=4096)
    parser.add_argument("--min-checkpoint-size-gb", type=float, default=DEFAULT_MIN_CHECKPOINT_SIZE_GB,
                        help="Skip likely incomplete checkpoint files smaller than this size; set 0 to disable.")
    parser.add_argument("--limit-checkpoints", "--limit-checkpoint-count", dest="limit_checkpoints", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--retry-failed", action="store_true",
                        help="Retry checkpoints whose previous row has status=failed. By default, any recorded checkpoint is skipped.")
    parser.add_argument("--keep-checkpoints", action="store_true")
    parser.add_argument("--compile", dest="no_compile", action="store_false",
                        help="Enable torch.compile; faster sometimes, but more likely to OOM on 16 GB GPUs.")
    parser.add_argument("--no-compile", dest="no_compile", action="store_true",
                        help="Disable torch.compile; default for this batch runner.")
    parser.set_defaults(no_compile=True)
    parser.add_argument("--refresh-assets", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    dirs = build_dirs(args.work_dir)
    print(f"[work] {dirs['root']}")

    all_checkpoints = discover_checkpoints(args)
    outputs = load_existing_outputs(dirs["results"])
    checkpoint_rows = outputs["checkpoint_rows"]
    metric_rows = outputs["metric_rows"]
    wide_rows = outputs["wide_rows"]
    source_rows = outputs["source_rows"]
    run_rows = outputs["run_rows"]

    processed = processed_checkpoint_ids(dirs["results"], retry_failed=args.retry_failed)
    if args.force:
        pending_checkpoints = all_checkpoints
    else:
        pending_checkpoints = [
            checkpoint for checkpoint in all_checkpoints
            if checkpoint["checkpoint_id"] not in processed
        ]

    min_bytes = min_checkpoint_bytes(args)
    pending_too_small = [
        checkpoint for checkpoint in pending_checkpoints
        if (checkpoint.get("size") or 0) < min_bytes
    ]
    runnable_checkpoints = [
        checkpoint for checkpoint in pending_checkpoints
        if (checkpoint.get("size") or 0) >= min_bytes
    ]

    checkpoints = runnable_checkpoints
    if args.limit_checkpoints:
        checkpoints = checkpoints[:args.limit_checkpoints]

    skipped_existing = len(all_checkpoints) - len(pending_checkpoints) if not args.force else 0
    print(f"[drive] found {len(all_checkpoints)} checkpoint(s)")
    if skipped_existing:
        print(f"[resume] skipping {skipped_existing} already recorded checkpoint(s)")
    if pending_too_small:
        print(f"[resume] skipping {len(pending_too_small)} pending checkpoint(s) below --min-checkpoint-size-gb {args.min_checkpoint_size_gb:g}")
        for checkpoint in pending_too_small:
            size_gb = (checkpoint.get("size") or 0) / 1e9
            print(f"  [small] {checkpoint['name']} ({size_gb:.2f} GB)")
    if args.limit_checkpoints:
        print(f"[limit] selected up to {args.limit_checkpoints} runnable checkpoint(s)")
    print(f"[queue] {len(checkpoints)} checkpoint(s) selected for this run")
    for checkpoint in checkpoints:
        print(f"  {checkpoint['name']} ({(checkpoint.get('size') or 0) / 1e9:.2f} GB)")
    if args.dry_run:
        return

    for checkpoint in pending_too_small:
        checkpoint_id = checkpoint["checkpoint_id"]
        checkpoint_row = skipped_small_checkpoint_row(args, checkpoint)
        checkpoint_rows = replace_rows(checkpoint_rows, "checkpoint_id", checkpoint_id, [checkpoint_row])
        metric_rows = replace_rows(metric_rows, "checkpoint_id", checkpoint_id, [])
        source_rows = replace_rows(source_rows, "checkpoint_id", checkpoint_id, [])
        wide_rows = replace_rows(wide_rows, "checkpoint_id", checkpoint_id, [])
        log_event(run_rows, checkpoint_id, "skipped_small", checkpoint_row["error"], 0)
        processed.add(checkpoint_id)
        print(f"[skip] {checkpoint_id}: {checkpoint_row['error']}")
    if pending_too_small:
        write_all_outputs(dirs["results"], checkpoint_rows, metric_rows, wide_rows, source_rows, run_rows)

    if not checkpoints:
        print("[done] no new runnable checkpoints to evaluate")
        return

    tokenizer_path, meta_path, manifest_path = ensure_assets(args, dirs)
    meta = load_json(meta_path)
    shard_manifest = load_json(manifest_path)

    for checkpoint in checkpoints:
        checkpoint_id = checkpoint["checkpoint_id"]
        print(f"[start] {checkpoint_id}")
        started = time.time()
        log_event(run_rows, checkpoint_id, "start", "evaluation started")
        checkpoint_row, new_metric_rows, wide_row, new_source_rows = evaluate_checkpoint(
            args,
            dirs,
            checkpoint,
            tokenizer_path,
            meta,
            shard_manifest,
        )
        elapsed = time.time() - started
        checkpoint_rows = replace_rows(checkpoint_rows, "checkpoint_id", checkpoint_id, [checkpoint_row])
        metric_rows = replace_rows(metric_rows, "checkpoint_id", checkpoint_id, new_metric_rows)
        source_rows = replace_rows(source_rows, "checkpoint_id", checkpoint_id, new_source_rows)
        wide_rows = replace_rows(wide_rows, "checkpoint_id", checkpoint_id, [wide_row] if wide_row else [])
        status = checkpoint_row.get("status")
        log_event(run_rows, checkpoint_id, status, checkpoint_row.get("error") or "evaluation complete", round(elapsed, 3))
        write_all_outputs(dirs["results"], checkpoint_rows, metric_rows, wide_rows, source_rows, run_rows)
        if status != "failed" or not args.retry_failed:
            processed.add(checkpoint_id)
        print(f"[done] {checkpoint_id}: {status} ({elapsed / 60:.1f} min)")

    write_all_outputs(dirs["results"], checkpoint_rows, metric_rows, wide_rows, source_rows, run_rows)
    print(f"[results] {dirs['results']}")


if __name__ == "__main__":
    main()
