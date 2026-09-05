#!/usr/bin/env python3
"""Build a source .txt -> training shard map from Synapse manifests.

The pipeline already records provenance in shard manifests. This script turns
that into an easy report:

    source .txt -> tokenizer shard -> merged training shard -> selected/eval status

Examples:
  python pretrain/map_training_sources.py --synapse-dir ./synapse --query all_sciences
  python pretrain/map_training_sources.py --remote gdrive:synapse --query all_sciences

Outputs default to out/source_map/, which is gitignored.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


CSV_FIELDS = [
    "source_txt",
    "source_file",
    "source_domain",
    "source_tokens",
    "source_shard",
    "merged_shard",
    "merged_domain",
    "merged_tokens",
    "merged_mb",
    "selected_for_training",
    "eval_heldout",
    "training_weight",
    "selected_domain_passes",
    "selected_domain_tokens",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Map original .txt sources to merged training shards."
    )
    parser.add_argument(
        "--synapse-dir",
        default=os.environ.get("SYNAPSE_DIR", "./synapse"),
        help="Local data root containing token_shards_merged/ and manifests/.",
    )
    parser.add_argument(
        "--remote",
        default="",
        help="Optional rclone root, e.g. gdrive:synapse. If set, manifests are read with rclone cat.",
    )
    parser.add_argument(
        "--merged-manifest",
        default="",
        help="Override local path to token_shards_merged/shard_manifest.json.",
    )
    parser.add_argument(
        "--training-manifest",
        default="",
        help="Override local path to manifests/training_latest.json.",
    )
    parser.add_argument(
        "--eval-pin",
        default="",
        help="Override local path to manifests/eval_shards.json.",
    )
    parser.add_argument(
        "--out-dir",
        default="out/source_map",
        help="Directory for source_training_map.csv and source_training_map.md.",
    )
    parser.add_argument(
        "--query",
        default="",
        help="Only show matching rows in the Markdown detail section, e.g. all_sciences.",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=30,
        help="Rows to show in Markdown when no --query is given.",
    )
    return parser.parse_args()


def rclone_cat(remote_root: str, relpath: str) -> str:
    remote = f"{remote_root.rstrip('/')}/{relpath.lstrip('/')}"
    result = subprocess.run(
        ["rclone", "cat", remote],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise FileNotFoundError(
            f"Could not read {remote!r} with rclone: {result.stderr.strip()}"
        )
    return result.stdout


def load_json_local(path: str | Path, *, required: bool) -> dict[str, Any] | None:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        if required:
            raise
        return None


def load_json(
    args: argparse.Namespace,
    relpath: str,
    override: str,
    *,
    required: bool,
) -> dict[str, Any] | None:
    if args.remote:
        try:
            return json.loads(rclone_cat(args.remote, relpath))
        except FileNotFoundError:
            if required:
                raise
            return None

    path = override or os.path.join(args.synapse_dir, relpath)
    return load_json_local(path, required=required)


def source_domain(path: str, fallback: str = "") -> str:
    parts = path.replace("\\", "/").split("/")
    for part in parts:
        if part.startswith("data_"):
            return part
    return fallback or "other"


def build_training_indexes(
    training_manifest: dict[str, Any] | None,
) -> tuple[set[str], dict[str, dict[str, Any]], dict[str, Any]]:
    if not training_manifest:
        return set(), {}, {}

    data_selection = training_manifest.get("data_selection", {})
    sources = data_selection.get("sources", {}) or {}
    selected_shards: set[str] = set()
    for info in sources.values():
        selected_shards.update(info.get("unique", []) or [])

    return selected_shards, sources, data_selection.get("data_mix", {}) or {}


def training_weight(data_mix: dict[str, Any], domain: str) -> Any:
    spec = data_mix.get(domain, "")
    if isinstance(spec, dict):
        return spec.get("weight", "")
    return spec


def build_rows(
    merged_manifest: dict[str, Any],
    training_manifest: dict[str, Any] | None,
    eval_pin: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    selected_shards, selected_by_domain, data_mix = build_training_indexes(training_manifest)
    eval_shards = set((eval_pin or {}).get("shards", []) or [])

    rows: list[dict[str, Any]] = []
    for merged in merged_manifest.get("shards", []):
        merged_shard = merged.get("shard", "")
        merged_domain = merged.get("domain") or source_domain(merged.get("source", ""))
        merged_tokens = int(merged.get("tokens") or 0)
        merged_mb = merged.get("shard_mb", "")

        sources = merged.get("merged_from") or [
            {
                "shard": merged_shard,
                "source": merged.get("source", ""),
                "tokens": merged_tokens,
            }
        ]
        selected_info = selected_by_domain.get(merged_domain, {})

        for src in sources:
            src_path = src.get("source", "")
            row = {
                "source_txt": src_path,
                "source_file": os.path.basename(src_path),
                "source_domain": source_domain(src_path, merged_domain),
                "source_tokens": int(src.get("tokens") or 0),
                "source_shard": src.get("shard", ""),
                "merged_shard": merged_shard,
                "merged_domain": merged_domain,
                "merged_tokens": merged_tokens,
                "merged_mb": merged_mb,
                "selected_for_training": "yes" if merged_shard in selected_shards else "no",
                "eval_heldout": "yes" if merged_shard in eval_shards else "no",
                "training_weight": training_weight(data_mix, merged_domain),
                "selected_domain_passes": selected_info.get("passes", ""),
                "selected_domain_tokens": selected_info.get("tokens", ""),
            }
            rows.append(row)
    return rows


def summarize_domains(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    stats: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "txt_files": 0,
            "source_tokens": 0,
            "merged_shards": set(),
            "selected_merged_shards": set(),
            "eval_merged_shards": set(),
        }
    )
    for row in rows:
        domain = row["source_domain"]
        bucket = stats[domain]
        bucket["txt_files"] += 1
        bucket["source_tokens"] += int(row["source_tokens"] or 0)
        bucket["merged_shards"].add(row["merged_shard"])
        if row["selected_for_training"] == "yes":
            bucket["selected_merged_shards"].add(row["merged_shard"])
        if row["eval_heldout"] == "yes":
            bucket["eval_merged_shards"].add(row["merged_shard"])

    summary = []
    for domain, bucket in stats.items():
        summary.append(
            {
                "domain": domain,
                "txt_files": bucket["txt_files"],
                "source_tokens": bucket["source_tokens"],
                "merged_shards": len(bucket["merged_shards"]),
                "selected_merged_shards": len(bucket["selected_merged_shards"]),
                "eval_merged_shards": len(bucket["eval_merged_shards"]),
            }
        )
    summary.sort(key=lambda row: (-row["source_tokens"], row["domain"]))
    return summary


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in CSV_FIELDS})


def md_table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(lines)


def row_matches(row: dict[str, Any], query: str) -> bool:
    q = query.lower()
    fields = ("source_txt", "source_file", "source_domain", "merged_shard")
    return any(q in str(row.get(field, "")).lower() for field in fields)


def write_markdown(
    rows: list[dict[str, Any]],
    domain_summary: list[dict[str, Any]],
    path: Path,
    args: argparse.Namespace,
    merged_manifest: dict[str, Any],
    training_manifest: dict[str, Any] | None,
) -> None:
    query = args.query.strip()
    if query:
        detail_rows = [row for row in rows if row_matches(row, query)]
    else:
        detail_rows = sorted(
            rows, key=lambda row: int(row["source_tokens"] or 0), reverse=True
        )[: args.top]

    selected_merged = {row["merged_shard"] for row in rows if row["selected_for_training"] == "yes"}
    eval_merged = {row["merged_shard"] for row in rows if row["eval_heldout"] == "yes"}
    tokenization_id = (
        merged_manifest.get("tokenization_id")
        or (training_manifest or {}).get("tokenization_id")
        or ""
    )

    lines = [
        "# Source To Training Map",
        "",
        f"Generated: {dt.datetime.now().isoformat(timespec='seconds')}",
        f"Source: {args.remote or os.path.abspath(args.synapse_dir)}",
        f"Tokenization ID: `{tokenization_id}`",
        "",
        "## What Each Row Means",
        "",
        "`source_txt` is the original text file. `source_shard` is the tokenizer shard "
        "made from that file. `merged_shard` is the larger `.bin` file that "
        "`train.py` reads. `selected_for_training=yes` means the merged shard "
        "appears in `training_latest.json`.",
        "",
        "## Totals",
        "",
        md_table(
            ["txt files", "source tokens", "merged shards", "selected merged shards", "eval heldout shards"],
            [[
                f"{len(rows):,}",
                f"{sum(int(row['source_tokens'] or 0) for row in rows):,}",
                f"{len({row['merged_shard'] for row in rows}):,}",
                f"{len(selected_merged):,}",
                f"{len(eval_merged):,}",
            ]],
        ),
        "",
    ]

    if training_manifest:
        data_selection = training_manifest.get("data_selection", {})
        lines.extend(
            [
                "## Latest Training Manifest",
                "",
                md_table(
                    ["status", "checkpoint step", "selected shard-passes", "selected tokens"],
                    [[
                        training_manifest.get("status", ""),
                        training_manifest.get("checkpoint_step", ""),
                        data_selection.get("selected_shards", ""),
                        data_selection.get("selected_tokens", ""),
                    ]],
                ),
                "",
            ]
        )

    lines.extend(
        [
            "## Domains",
            "",
            md_table(
                ["domain", "txt files", "source tokens", "merged", "selected", "eval"],
                [
                    [
                        row["domain"],
                        f"{row['txt_files']:,}",
                        f"{row['source_tokens']:,}",
                        f"{row['merged_shards']:,}",
                        f"{row['selected_merged_shards']:,}",
                        f"{row['eval_merged_shards']:,}",
                    ]
                    for row in domain_summary
                ],
            ),
            "",
        ]
    )

    title = f"Matches For `{query}`" if query else f"Largest {len(detail_rows)} Source Files"
    lines.extend(
        [
            f"## {title}",
            "",
            md_table(
                ["source file", "domain", "source tokens", "source shard", "merged shard", "selected", "eval"],
                [
                    [
                        row["source_file"],
                        row["source_domain"],
                        f"{int(row['source_tokens']):,}",
                        row["source_shard"],
                        row["merged_shard"],
                        row["selected_for_training"],
                        row["eval_heldout"],
                    ]
                    for row in detail_rows
                ]
                or [["(none)", "", "", "", "", "", ""]],
            ),
            "",
        ]
    )

    if query and detail_rows:
        matched_merged = sorted({row["merged_shard"] for row in detail_rows})
        sibling_rows = [row for row in rows if row["merged_shard"] in matched_merged]
        sibling_rows.sort(key=lambda row: (row["merged_shard"], row["source_file"]))
        lines.extend(
            [
                "## Same Merged Shard Contents",
                "",
                "Training reads the whole merged shard, so these files travel with the query match.",
                "",
                md_table(
                    ["merged shard", "source file", "tokens"],
                    [
                        [row["merged_shard"], row["source_file"], f"{int(row['source_tokens']):,}"]
                        for row in sibling_rows
                    ],
                ),
                "",
            ]
        )

    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    merged_manifest = load_json(
        args,
        "token_shards_merged/shard_manifest.json",
        args.merged_manifest,
        required=True,
    )
    training_manifest = load_json(
        args,
        "manifests/training_latest.json",
        args.training_manifest,
        required=False,
    )
    eval_pin = load_json(
        args,
        "manifests/eval_shards.json",
        args.eval_pin,
        required=False,
    )

    rows = build_rows(merged_manifest, training_manifest, eval_pin)
    if not rows:
        raise RuntimeError("No source rows found in merged manifest.")

    domain_summary = summarize_domains(rows)
    csv_path = out_dir / "source_training_map.csv"
    md_path = out_dir / "source_training_map.md"
    write_csv(rows, csv_path)
    write_markdown(rows, domain_summary, md_path, args, merged_manifest, training_manifest)

    print(f"Wrote {len(rows):,} source rows")
    print(f"  CSV:      {csv_path}")
    print(f"  Markdown: {md_path}")
    if args.query:
        matches = sum(1 for row in rows if row_matches(row, args.query))
        print(f"  Query {args.query!r}: {matches:,} matching row(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
