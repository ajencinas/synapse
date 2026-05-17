#!/usr/bin/env python3
"""Push SFT data + manifests from local $SYNAPSE_DIR to a Drive remote.

Usage:
  SYNAPSE_DIR=/workspace/synapse SYNAPSE_PUSH_REMOTE=gdrive:synapse \
      python sft/push_to_drive.py --datasets alpaca

  # Or pass explicitly:
  python sft/push_to_drive.py --datasets alpaca \
      --synapse-dir /workspace/synapse --remote gdrive:synapse

Pushes:
  $SYNAPSE_DIR/sft_tokenized/<name>/         → <remote>/sft_tokenized/<name>/
  $SYNAPSE_DIR/sft_tokenized/tokenization_id.txt
                                             → <remote>/sft_tokenized/
  $SYNAPSE_DIR/manifests/sft_tokenization_latest.json
                                             → <remote>/manifests/
  $SYNAPSE_DIR/datasets_sft/<name>/          → <remote>/datasets_sft/<name>/   (unless --no-raw)

Uses `rclone copy --checksum` (idempotent — skips files already matching at
the remote). Synchronous and fail-loud: any rclone non-zero exit aborts.
Skips dirs that don't exist locally so partial runs (e.g. tokenize only)
still work.
"""
import argparse
import os
import subprocess
import sys


def default_synapse_dir():
    if os.path.isdir("/content/drive/MyDrive"):
        return "/content/drive/MyDrive/synapse"
    return os.path.abspath("./synapse")


def rclone_copy(src, dst, *, is_dir):
    if not os.path.exists(src):
        print(f"  skip (missing locally): {src}")
        return
    label = "dir " if is_dir else "file"
    print(f"  push {label}: {src}  →  {dst}")
    cmd = [
        "rclone", "copy" if is_dir else "copyto",
        src, dst,
        "--checksum", "--drive-chunk-size=64M",
        "--stats", "10s", "--stats-one-line",
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        sys.stderr.write(
            f"rclone failed (exit {r.returncode}):\n"
            f"  cmd: {' '.join(cmd)}\n"
            f"  stderr: {r.stderr.strip()[:500]}\n"
        )
        raise SystemExit(r.returncode)
    if r.stdout.strip():
        for line in r.stdout.strip().splitlines()[-3:]:
            print(f"    {line}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", required=True, help="comma-separated names, or 'all'")
    ap.add_argument("--synapse-dir",
                    default=os.environ.get("SYNAPSE_DIR") or default_synapse_dir())
    ap.add_argument("--remote",
                    default=os.environ.get("SYNAPSE_PUSH_REMOTE"),
                    help="rclone remote, e.g. gdrive:synapse")
    ap.add_argument("--no-raw", action="store_true",
                    help="skip pushing datasets_sft/<name>/ (raw download)")
    args = ap.parse_args()

    if not args.remote:
        raise SystemExit("--remote (or $SYNAPSE_PUSH_REMOTE) is required")
    remote = args.remote.rstrip("/")
    syn = args.synapse_dir

    if args.datasets == "all":
        tok_base = os.path.join(syn, "sft_tokenized")
        if not os.path.isdir(tok_base):
            raise SystemExit(f"--datasets all: {tok_base} missing")
        names = sorted(
            n for n in os.listdir(tok_base)
            if os.path.isdir(os.path.join(tok_base, n))
        )
    else:
        names = args.datasets.split(",")

    print(f"synapse_dir: {syn}")
    print(f"remote:      {remote}")
    print(f"datasets:    {names}")
    print(f"include raw: {not args.no_raw}")
    print()

    for name in names:
        print(f"[{name}]")
        rclone_copy(
            os.path.join(syn, "sft_tokenized", name),
            f"{remote}/sft_tokenized/{name}",
            is_dir=True,
        )
        if not args.no_raw:
            rclone_copy(
                os.path.join(syn, "datasets_sft", name),
                f"{remote}/datasets_sft/{name}",
                is_dir=True,
            )

    print("[shared]")
    rclone_copy(
        os.path.join(syn, "sft_tokenized", "tokenization_id.txt"),
        f"{remote}/sft_tokenized/tokenization_id.txt",
        is_dir=False,
    )
    rclone_copy(
        os.path.join(syn, "manifests", "sft_tokenization_latest.json"),
        f"{remote}/manifests/sft_tokenization_latest.json",
        is_dir=False,
    )
    print("\nDone.")


if __name__ == "__main__":
    main()
