#!/usr/bin/env python3
"""Mini Streamlit dashboard for Synapse Drive checkpoint eval results."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
import streamlit as st


DEFAULT_RESULTS_DIR = Path("/media/alfonso/shared/synapse_drive_eval/results")


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def numeric(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in columns:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def parse_oom(message: object) -> str:
    text = "" if pd.isna(message) else str(message)
    if "OutOfMemoryError" not in text and "CUDA out of memory" not in text:
        return "other"
    match = re.search(r"Tried to allocate ([^.]+?)(?:\\. GPU| GiB| MiB)", text)
    if match:
        return f"cuda oom: tried {match.group(1).strip()}"
    return "cuda oom"


def add_order(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    if "checkpoint_timestamp_from_name" in out.columns:
        out["checkpoint_time"] = pd.to_datetime(out["checkpoint_timestamp_from_name"], errors="coerce")
    else:
        out["checkpoint_time"] = pd.NaT
    if "checkpoint_step" in out.columns:
        out["checkpoint_step_num"] = pd.to_numeric(out["checkpoint_step"], errors="coerce")
    else:
        out["checkpoint_step_num"] = pd.NA
    out["checkpoint_label"] = out.get("checkpoint_id", pd.Series(index=out.index, dtype="object")).fillna("")
    out = out.sort_values(["checkpoint_step_num", "checkpoint_time", "checkpoint_label"], na_position="last")
    out["order"] = range(1, len(out) + 1)
    return out


def load_results(results_dir: Path) -> dict[str, pd.DataFrame]:
    return {
        "checkpoints": add_order(read_csv(results_dir / "checkpoints.csv")),
        "wide": add_order(read_csv(results_dir / "metrics_wide.csv")),
        "long": add_order(read_csv(results_dir / "metrics_long.csv")),
        "source": add_order(read_csv(results_dir / "eval_loss_by_source.csv")),
        "run_log": read_csv(results_dir / "run_log.csv"),
    }


def metric_card(label: str, value: object, help_text: str | None = None) -> None:
    st.metric(label, value if value not in (None, "") else "n/a", help=help_text)


def render_overview(data: dict[str, pd.DataFrame]) -> None:
    checkpoints = data["checkpoints"]
    wide = data["wide"]
    run_log = data["run_log"]

    st.subheader("Run Overview")
    if checkpoints.empty:
        st.warning("No checkpoint rows found.")
        return

    status_counts = checkpoints["status"].fillna("unknown").value_counts()
    ok_count = int(status_counts.get("ok", 0))
    failed_count = int(status_counts.get("failed", 0))
    total_count = len(checkpoints)
    total_runtime = pd.to_numeric(checkpoints.get("total_time_s"), errors="coerce").sum()

    cols = st.columns(4)
    with cols[0]:
        metric_card("Checkpoints", total_count)
    with cols[1]:
        metric_card("Succeeded", ok_count)
    with cols[2]:
        metric_card("Failed", failed_count)
    with cols[3]:
        metric_card("Total Runtime", f"{total_runtime / 3600:.2f} h")

    if ok_count == 0 and failed_count:
        st.error("All checkpoints failed. The current run produced failure diagnostics but no benchmark/loss trend metrics.")
        st.code(
            "python3 sparky/eval_drive_checkpoints.py \\\n"
            "  --work-dir /media/alfonso/shared/synapse_drive_eval \\\n"
            "  --preset quick \\\n"
            "  --eval-batches 64 \\\n"
            "  --source-eval-batches 8 \\\n"
            "  --batch-size 1 \\\n"
            "  --max-batch-tokens 1024 \\\n"
            "  --no-compile \\\n"
            "  --force \\\n"
            "  --device cuda",
            language="bash",
        )

    st.write("Status counts")
    st.bar_chart(status_counts.rename_axis("status").reset_index(name="count"), x="status", y="count")

    runtime = numeric(checkpoints, ["total_time_s", "download_time_s", "load_time_s", "quick_eval_time_s", "heldout_eval_time_s"])
    runtime_cols = [c for c in ["download_time_s", "load_time_s", "quick_eval_time_s", "heldout_eval_time_s", "total_time_s"] if c in runtime]
    if runtime_cols:
        chart = runtime[["checkpoint_label", *runtime_cols]].set_index("checkpoint_label")
        st.write("Runtime by checkpoint")
        st.bar_chart(chart)

    if not wide.empty:
        st.write("Best available summary")
        shown = [c for c in ["checkpoint_id", "checkpoint_step", "heldout_loss", "heldout_ppl", "quick_mean_primary", "is_best_heldout_loss", "is_best_quick_mean"] if c in wide]
        st.dataframe(wide[shown], use_container_width=True, hide_index=True)

    if not run_log.empty:
        st.write("Recent run log")
        st.dataframe(run_log.tail(20), use_container_width=True, hide_index=True)


def render_metrics(data: dict[str, pd.DataFrame]) -> None:
    wide = numeric(
        data["wide"],
        [
            "heldout_loss",
            "heldout_ppl",
            "quick_mean_primary",
            "quick_mean_raw_acc",
            "anli_r1_acc",
            "boolq_acc",
            "piqa_acc_norm",
            "sciq_acc_norm",
            "openbookqa_acc_norm",
        ],
    )
    long = numeric(data["long"], ["value"])

    st.subheader("Benchmark And Loss Trends")
    if wide.empty and long.empty:
        st.warning("No metric rows were produced. Check the Failures tab for the reason.")
        return

    trend_cols = [c for c in ["heldout_loss", "heldout_ppl", "quick_mean_primary", "quick_mean_raw_acc"] if c in wide and wide[c].notna().any()]
    if trend_cols:
        st.write("Primary trends")
        st.line_chart(wide[["checkpoint_label", *trend_cols]].set_index("checkpoint_label"))

    task_cols = [c for c in ["anli_r1_acc", "boolq_acc", "piqa_acc_norm", "sciq_acc_norm", "openbookqa_acc_norm"] if c in wide and wide[c].notna().any()]
    if task_cols:
        st.write("Quick benchmark tasks")
        st.line_chart(wide[["checkpoint_label", *task_cols]].set_index("checkpoint_label"))

    if not long.empty and {"task", "metric", "value", "checkpoint_label"}.issubset(long.columns):
        choices = sorted((long["task"].fillna("") + " / " + long["metric"].fillna("")).unique())
        selected = st.multiselect("Metric series", choices, default=choices[: min(8, len(choices))])
        if selected:
            tmp = long.copy()
            tmp["series"] = tmp["task"].fillna("") + " / " + tmp["metric"].fillna("")
            pivot = tmp[tmp["series"].isin(selected)].pivot_table(index="checkpoint_label", columns="series", values="value", aggfunc="first")
            st.line_chart(pivot)

    st.write("Wide metrics table")
    st.dataframe(wide, use_container_width=True, hide_index=True)


def render_source_loss(data: dict[str, pd.DataFrame]) -> None:
    source = numeric(data["source"], ["loss", "ppl", "num_batches", "num_tokens_eval"])
    st.subheader("Eval Loss By Source")
    if source.empty:
        st.warning("No per-source loss rows were produced.")
        return

    metric = st.radio("Metric", ["loss", "ppl"], horizontal=True)
    pivot = source.pivot_table(index="checkpoint_label", columns="source", values=metric, aggfunc="first")
    st.line_chart(pivot)
    st.write("Per-source table")
    st.dataframe(source, use_container_width=True, hide_index=True)


def render_failures(data: dict[str, pd.DataFrame]) -> None:
    checkpoints = data["checkpoints"].copy()
    st.subheader("Failures And Diagnostics")
    if checkpoints.empty:
        st.warning("No checkpoint rows found.")
        return

    failed = checkpoints[checkpoints["status"].fillna("") != "ok"].copy()
    if failed.empty:
        st.success("No failed checkpoints.")
        return

    failed["error_type"] = failed["error"].map(parse_oom)
    st.write("Failure types")
    counts = failed["error_type"].value_counts().rename_axis("error_type").reset_index(name="count")
    st.bar_chart(counts, x="error_type", y="count")

    display_cols = [c for c in ["checkpoint_id", "checkpoint_timestamp_from_name", "file_size_bytes", "total_time_s", "error_type", "error"] if c in failed]
    st.dataframe(failed[display_cols], use_container_width=True, hide_index=True)


def render_data(data: dict[str, pd.DataFrame]) -> None:
    st.subheader("Raw Tables")
    for name, df in data.items():
        with st.expander(f"{name} ({len(df)} rows)", expanded=name == "checkpoints"):
            st.dataframe(df, use_container_width=True, hide_index=True)


def main() -> None:
    st.set_page_config(page_title="Synapse Eval Dashboard", layout="wide")
    st.title("Synapse Checkpoint Eval Dashboard")

    with st.sidebar:
        st.header("Data")
        results_dir = Path(st.text_input("Results directory", str(DEFAULT_RESULTS_DIR))).expanduser()
        st.caption("Reads CSVs generated by `sparky/eval_drive_checkpoints.py`.")
        refresh = st.button("Refresh")

    if refresh:
        st.cache_data.clear()

    data = load_results(results_dir)
    st.caption(f"Results: `{results_dir}`")

    tabs = st.tabs(["Overview", "Metrics", "Source Loss", "Failures", "Raw Data"])
    with tabs[0]:
        render_overview(data)
    with tabs[1]:
        render_metrics(data)
    with tabs[2]:
        render_source_loss(data)
    with tabs[3]:
        render_failures(data)
    with tabs[4]:
        render_data(data)


if __name__ == "__main__":
    main()
