#!/usr/bin/env python3
"""CLI to build the interactive GPU perf HTML dashboard from ``perf_samples_*.csv``.

This script is the supported entrypoint for dashboard generation. Chart rendering is
delegated to ``generate_perf_charts_multi`` in the project helper module.
"""

from __future__ import annotations

import argparse
import html as html_mod
import json
import logging
import re
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from srt_upload_helper import generate_perf_charts_multi, _PERF_METRIC_PANELS

logger = logging.getLogger(__name__)


def _slurm_job_id_for_log_dir(log_dir: Path) -> str:
    """Infer Slurm job id from ``.../<jobid>/logs`` or ``.../<jobid>`` style paths."""
    if log_dir.name == "logs" and log_dir.parent.name.isdigit():
        return log_dir.parent.name
    if log_dir.name.isdigit():
        return log_dir.name
    if log_dir.parent.name.isdigit():
        return log_dir.parent.name
    return log_dir.name


def _resolve_log_dir(root: Path) -> Path:
    """Use *root* if it contains perf CSVs; otherwise ``root / 'logs'`` if that does."""
    root = root.resolve()
    if list(root.glob("perf_samples_*.csv")):
        return root
    logs = root / "logs"
    if list(logs.glob("perf_samples_*.csv")):
        logger.info("Using %s (perf CSVs under logs/)", logs)
        return logs
    return root


def _normalize_slurm_ids(n_runs: int, ids: list[str] | None) -> list[str | None] | None:
    """Return ``None`` for auto-inference, or one id per run after validation."""
    if not ids:
        return None
    if len(ids) == n_runs:
        return [s.strip() or None for s in ids]
    if len(ids) == 1 and n_runs == 1:
        return [ids[0].strip() or None]
    sys.stderr.write(
        f"error: got {len(ids)} --slurm-id value(s) for {n_runs} run(s); "
        "pass one per run_dir or omit for automatic ids.\n"
    )
    sys.exit(2)


def _document_title(
    n_runs: int,
    *,
    title: str | None,
    job_id: str | None,
    resolved_logs: list[Path],
    slurm_ids: list[str | None] | None,
) -> str | None:
    """HTML ``<title>`` / header string; ``None`` lets the chart layer pick defaults."""
    if title is not None:
        return title
    if job_id is not None:
        return job_id
    if n_runs == 1 and slurm_ids and slurm_ids[0]:
        return slurm_ids[0]
    if n_runs == 1:
        return _slurm_job_id_for_log_dir(resolved_logs[0])
    return None


def _find_total_power_panel_idx() -> int | None:
    for i, p in enumerate(_PERF_METRIC_PANELS):
        if p.get("agg") == "sum" and p["col"] == "power_w":
            return i
    return None


def _extract_js_var(html: str, var_name: str) -> tuple[str, int, int] | None:
    """Find ``var <var_name> = <JSON>;`` and return (json_str, start, end_after_semicolon)."""
    marker = f"var {var_name} = "
    start = html.find(marker)
    if start == -1:
        return None
    json_start = start + len(marker)
    depth = 0
    i = json_start
    while i < len(html):
        ch = html[i]
        if ch == '{' or ch == '[':
            depth += 1
        elif ch == '}' or ch == ']':
            depth -= 1
            if depth == 0:
                json_end = i + 1
                semi = html.find(';', json_end)
                if semi == -1:
                    semi = json_end
                return html[json_start:json_end], start, semi + 1
        elif ch == '"':
            i += 1
            while i < len(html) and html[i] != '"':
                if html[i] == '\\':
                    i += 1
                i += 1
        i += 1
    return None


_RACK_RE = re.compile(r"node-([a-z]+\d+)-")
_RACK_RE_ALT = re.compile(r"gb\d+-nvl-(\d+)-")
_RACK_COLORS = ["#FF5722", "#E91E63", "#9C27B0", "#3F51B5", "#009688", "#795548", "#607D8B", "#FF9800"]


def _extract_rack_id(host_key: str) -> str:
    """Extract rack id from composite host key like '86361::node-b11-c01' -> 'b11'."""
    m = _RACK_RE.search(host_key)
    if m:
        return m.group(1)
    m = _RACK_RE_ALT.search(host_key)
    return m.group(1) if m else "unknown"


def _inject_whole_rack_trace(html_path: Path) -> None:
    """Post-process the dashboard HTML to add per-rack 'Whole Rack' traces to Total Power Draw.

    Groups per-host total traces by rack (from node naming like node-b11-*) and
    creates one summed trace per rack.
    """
    panel_idx = _find_total_power_panel_idx()
    if panel_idx is None:
        return

    html = html_path.read_text(encoding="utf-8")

    traces_parsed = _extract_js_var(html, "PANEL_TRACES")
    meta_parsed = _extract_js_var(html, "PANEL_META")
    if not traces_parsed or not meta_parsed:
        logger.warning("Could not parse PANEL_TRACES/PANEL_META from HTML; skipping Whole Rack injection")
        return

    traces_json, traces_start, traces_end = traces_parsed
    all_traces = json.loads(traces_json)

    meta_json, _, _ = meta_parsed
    all_meta = json.loads(meta_json)

    panel_key = str(panel_idx)
    traces = all_traces.get(panel_key, [])
    meta = all_meta.get(panel_key, [])

    # Collect per-host total traces and group by rack
    rack_host_traces: dict[str, list[dict]] = {}
    rack_hosts: dict[str, set[str]] = {}
    for i, m in enumerate(meta):
        t = traces[i]
        host = m.get("host", "")
        if host == "__agg" or host == "":
            # For role-level aggregates, group by rack from source hosts
            if not m.get("is_worker_agg") and "Total" in t.get("name", ""):
                src_hosts = m.get("agg_source_hosts") or []
                for sh in src_hosts:
                    rid = _extract_rack_id(sh)
                    rack_hosts.setdefault(rid, set()).add(sh)
                if src_hosts:
                    rid = _extract_rack_id(src_hosts[0])
                    rack_host_traces.setdefault(rid, []).append(t)
            continue
        rid = _extract_rack_id(host)
        rack_hosts.setdefault(rid, set()).add(host)

    # Also collect per-host (non-agg) total traces by rack
    per_host_by_rack: dict[str, list[dict]] = {}
    for i, m in enumerate(meta):
        t = traces[i]
        host = m.get("host", "")
        if host and host != "__agg" and "Total" in t.get("name", ""):
            rid = _extract_rack_id(host)
            per_host_by_rack.setdefault(rid, []).append(t)

    # Build per-rack sum traces from per-host totals
    all_source_hosts = set()
    for m in meta:
        for h in m.get("agg_source_hosts") or []:
            all_source_hosts.add(h)
        if m.get("host") and m["host"] != "__agg":
            all_source_hosts.add(m["host"])

    racks = sorted(set(list(rack_host_traces.keys()) + list(per_host_by_rack.keys())))
    if not racks:
        logger.info("No rack data found; skipping Whole Rack")
        return

    for rack_idx, rid in enumerate(racks):
        # Use per-host totals if available, else role totals
        source_traces = per_host_by_rack.get(rid) or rack_host_traces.get(rid, [])
        if not source_traces:
            continue

        dfs = []
        for t in source_traces:
            df = pd.DataFrame({"ts": t["x"], "power": t["y"]})
            df["ts"] = pd.to_datetime(df["ts"], format="ISO8601", utc=True)
            dfs.append(df.set_index("ts"))

        combined = pd.concat(dfs).groupby(level=0).sum().sort_index()
        x = [ts.isoformat() for ts in combined.index]
        y = combined["power"].tolist()

        color = _RACK_COLORS[rack_idx % len(_RACK_COLORS)]
        label = f"Rack {rid} Total"
        rack_trace = {
            "x": x, "y": y,
            "type": "scatter", "mode": "lines",
            "name": label, "legendgroup": f"__rack_{rid}",
            "showlegend": True,
            "line": {"color": color, "width": 4, "dash": "solid"},
            "hovertemplate": (
                f"<b>{label}</b><br>"
                "Time (UTC): %{x|%Y-%m-%d %H:%M:%S}<br>"
                f"Total Power Draw: %{{y:.1f}} W"
                "<extra></extra>"
            ),
        }
        rack_source = sorted(rack_hosts.get(rid, set()) | all_source_hosts)
        rack_meta_entry = {
            "host": "__agg", "gpu": -1, "agg_role": "rack",
            "agg_source_hosts": sorted(rack_hosts.get(rid, all_source_hosts)),
        }

        traces.append(rack_trace)
        meta.append(rack_meta_entry)

    all_traces[panel_key] = traces
    all_meta[panel_key] = meta

    new_traces_line = f"var PANEL_TRACES = {json.dumps(all_traces, default=str)};\n"
    html = html[:traces_start] + new_traces_line + html[traces_end:]

    meta_parsed2 = _extract_js_var(html, "PANEL_META")
    if meta_parsed2:
        _, meta_start2, meta_end2 = meta_parsed2
        new_meta_line = f"var PANEL_META = {json.dumps(all_meta)};\n"
        html = html[:meta_start2] + new_meta_line + html[meta_end2:]

    html_path.write_text(html, encoding="utf-8")
    logger.info("Injected Whole Rack trace into panel %d", panel_idx)


_BARE_TS_RE = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})$", re.MULTILINE)


def _parse_benchmark_phases(log_dir: Path) -> list[dict[str, Any]]:
    """Parse benchmark.out for warmup and main benchmark result blocks.

    Returns a list of dicts with keys: phase, concurrency, duration_s,
    num_requests, total_token_tps, median_tpot_ms, start_time, isl, osl,
    and optionally prep_s (seconds spent on request generation / tokenization
    before the actual benchmark measurement began).
    """
    bench_path = log_dir / "benchmark.out"
    if not bench_path.exists():
        return []

    text = bench_path.read_text(encoding="utf-8", errors="replace")

    # Parse ISL/OSL from SA-Bench Config line
    isl, osl = None, None
    cfg_match = re.search(r"SA-Bench Config:.*?isl=(\d+).*?osl=(\d+)", text)
    if cfg_match:
        isl, osl = int(cfg_match.group(1)), int(cfg_match.group(2))
    if isl is None:
        for candidate in [log_dir / "config.yaml", log_dir.parent / "config.yaml"]:
            if candidate.is_file():
                try:
                    with open(candidate, "r") as f:
                        cfg = yaml.safe_load(f)
                    bench_cfg = cfg.get("benchmark") or {}
                    isl = bench_cfg.get("isl")
                    osl = bench_cfg.get("osl")
                    if isl is not None:
                        break
                except Exception:
                    pass

    blocks = text.split("============ Serving Benchmark Result ============")
    if len(blocks) < 2:
        return []

    def _parse_block(block: str) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for line in block.splitlines():
            line = line.strip()
            if line.startswith("Successful requests:"):
                result["num_requests"] = int(line.split(":")[1].strip())
            elif line.startswith("Benchmark duration (s):"):
                result["duration_s"] = float(line.split(":")[1].strip())
            elif line.startswith("Total Token throughput (tok/s):"):
                result["total_token_tps"] = float(line.split(":")[1].strip())
            elif line.startswith("Median TPOT (ms):"):
                result["median_tpot_ms"] = float(line.split(":")[1].strip())
            elif line.startswith("Maximum request concurrency:"):
                result["concurrency"] = int(line.split(":")[1].strip())
            elif line.startswith("Total input tokens:"):
                result["total_input_tokens"] = int(line.split(":")[1].strip())
            elif line.startswith("Total generated tokens:"):
                result["total_generated_tokens"] = int(line.split(":")[1].strip())
        return result

    def _find_first_timestamp(block: str) -> str | None:
        m = _BARE_TS_RE.search(block)
        return m.group(1) if m else None

    from datetime import datetime, timedelta

    phases = []

    # Warmup block: text between first and second separator
    warmup_data = _parse_block(blocks[1])
    if warmup_data:
        warmup_data["phase"] = "Warmup"
        warmup_data["isl"] = isl
        warmup_data["osl"] = osl
        if "concurrency" not in warmup_data:
            conc_match = re.search(r"Maximum request concurrency:\s+(\d+)", blocks[0])
            if conc_match:
                warmup_data["concurrency"] = int(conc_match.group(1))
        main_ts_str = _find_first_timestamp(blocks[1])
        dur = warmup_data.get("duration_s")
        if main_ts_str and dur:
            main_ts = datetime.strptime(main_ts_str, "%Y-%m-%d %H:%M:%S")
            warmup_start = main_ts - timedelta(seconds=dur)
            warmup_data["start_time"] = warmup_start.strftime("%Y-%m-%d %H:%M:%S")
        phases.append(warmup_data)

    # Main block: after "Running benchmark with concurrency:"
    if len(blocks) >= 3:
        main_data = _parse_block(blocks[2])
        if main_data:
            main_data["phase"] = "Benchmark"
            main_data["isl"] = isl
            main_data["osl"] = osl
            if "concurrency" not in main_data:
                conc_match = re.search(
                    r"Running benchmark with concurrency:\s+(\d+)", blocks[1]
                )
                if conc_match:
                    main_data["concurrency"] = int(conc_match.group(1))
            start_ts_str = _find_first_timestamp(blocks[1])
            if start_ts_str:
                main_data["start_time"] = start_ts_str
            end_ts_str = _find_first_timestamp(blocks[2])
            bench_dur = main_data.get("duration_s")
            if start_ts_str and end_ts_str and bench_dur:
                start_dt = datetime.strptime(start_ts_str, "%Y-%m-%d %H:%M:%S")
                end_dt = datetime.strptime(end_ts_str, "%Y-%m-%d %H:%M:%S")
                wall_s = (end_dt - start_dt).total_seconds()
                main_data["prep_s"] = max(0.0, wall_s - bench_dur)
            phases.append(main_data)

    return phases


def _gpu_counts_from_config(log_dir: Path) -> dict[str, int]:
    """Return {"total": N, "prefill": N, "decode": N} from config.yaml."""
    for candidate in [log_dir / "config.yaml", log_dir.parent / "config.yaml"]:
        if candidate.is_file():
            try:
                with open(candidate, "r") as f:
                    cfg = yaml.safe_load(f)
                res = cfg.get("resources") or {}
                pn = res.get("prefill_nodes", 0)
                dn = res.get("decode_nodes", 0)
                gpn = res.get("gpus_per_node", 1)
                return {
                    "total": (pn + dn) * gpn,
                    "prefill": pn * gpn,
                    "decode": dn * gpn,
                }
            except Exception:
                pass
    return {"total": 0, "prefill": 0, "decode": 0}


_PREP_FILL = "rgba(33,150,243,0.20)"
_PREP_LINE = "#1565c0"
_WARMUP_FILL = "rgba(255,193,7,0.25)"
_WARMUP_LINE = "#f9a825"
_MAIN_REGION_COLORS = [
    "rgba(118,185,0,0.25)", "rgba(26,115,232,0.25)",
    "rgba(211,47,47,0.25)", "rgba(156,39,176,0.25)",
]
_MAIN_LINE_COLORS = ["#76b900", "#1a73e8", "#d32f2f", "#9c27b0"]


def _split_prep_from_main(
    bench_intervals: list[dict[str, Any]],
    resolved_logs: list[Path],
) -> list[dict[str, Any]]:
    """Split "main" bench intervals into "prep" + "main" using prep_s from benchmark.out.

    Modifies the main intervals in-place (shortening their start) and returns
    new prep intervals to be appended to the list.
    """
    from datetime import timedelta

    jid_prep: dict[str, list[float]] = {}
    for ld in resolved_logs:
        jid = _slurm_job_id_for_log_dir(ld)
        phases = _parse_benchmark_phases(ld)
        preps = [p.get("prep_s", 0) or 0 for p in phases if p.get("phase") == "Benchmark"]
        if preps:
            jid_prep.setdefault(jid, []).extend(preps)

    prep_intervals: list[dict[str, Any]] = []
    jid_counters: dict[str, int] = {}
    for iv in bench_intervals:
        if iv.get("phase") != "main":
            continue
        jid = iv.get("slurm_job_id", "")
        idx = jid_counters.get(jid, 0)
        jid_counters[jid] = idx + 1
        preps = jid_prep.get(jid, [])
        if idx >= len(preps):
            continue
        prep_s = preps[idx]
        if prep_s <= 0:
            continue
        prep_end = pd.Timestamp(iv["start_wall"]) + timedelta(seconds=prep_s)
        prep_intervals.append({
            "phase": "prep",
            "start_wall": iv["start_wall"],
            "end_wall": prep_end.isoformat(),
            "slurm_job_id": jid,
            "concurrency": iv.get("concurrency"),
        })
        iv["start_wall"] = prep_end.isoformat()

    return prep_intervals


def _build_phase_timeline_html(resolved_logs: list[Path]) -> str:
    """Build a Plotly timeline chart showing benchmark phases as colored boxes."""
    from srt_upload_helper import (
        _parse_benchmark_log, _sync_perf_timestamps, _read_perf_csv,
        _host_from_csv_name, _parse_parallelization_tag,
        _COMPOSITE_HOST_SEP,
    )
    from datetime import timedelta

    _REGISTERED_RE = re.compile(
        r"\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\] Successfully registered LLM"
    )

    all_intervals: list[dict[str, Any]] = []

    for ld in resolved_logs:
        jid = _slurm_job_id_for_log_dir(ld)
        par_tag = _parse_parallelization_tag(ld)

        csv_files = sorted(ld.glob("perf_samples_*.csv"))
        if not csv_files:
            continue
        hd: dict[str, pd.DataFrame] = {}
        for csv_path in csv_files:
            try:
                hostname = _host_from_csv_name(csv_path.name)
                df = _read_perf_csv(csv_path)
                if not df.empty:
                    hd[f"{jid}{_COMPOSITE_HOST_SEP}{hostname}"] = df
            except Exception:
                pass
        if not hd:
            continue

        tz_offset = _compute_tz_offset(ld)
        csv_t0 = min(d["timestamp"].min() for d in hd.values())

        # Parse earliest worker log timestamp and last "Successfully registered LLM"
        _FIRST_TS_RE = re.compile(r"\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\]")
        earliest_worker_ts: pd.Timestamp | None = None
        last_registered: pd.Timestamp | None = None
        for f in sorted(ld.iterdir()):
            if not _WORKER_FILE_RE.match(f.name):
                continue
            try:
                text = f.read_text(encoding="utf-8", errors="replace")
            except Exception:
                continue
            first_m = _FIRST_TS_RE.search(text)
            if first_m:
                ts = pd.Timestamp(first_m.group(1))
                ts_utc = (ts + tz_offset).tz_localize("UTC")
                if earliest_worker_ts is None or ts_utc < earliest_worker_ts:
                    earliest_worker_ts = ts_utc
            for m in _REGISTERED_RE.finditer(text):
                ts = pd.Timestamp(m.group(1))
                ts_utc = (ts + tz_offset).tz_localize("UTC")
                if last_registered is None or ts_utc > last_registered:
                    last_registered = ts_utc

        startup_start = earliest_worker_ts or csv_t0
        if last_registered is not None:
            all_intervals.append({
                "phase": "startup",
                "start_wall": startup_start.isoformat(),
                "end_wall": last_registered.isoformat(),
                "slurm_job_id": jid,
                "par_tag": par_tag,
            })

        raw_intervals = _parse_benchmark_log(ld)
        _, bench = _sync_perf_timestamps(hd, raw_intervals or None)
        for iv in bench:
            iv["slurm_job_id"] = jid
            iv["par_tag"] = par_tag

        phases = _parse_benchmark_phases(ld)
        main_phases = [p for p in phases if p.get("phase") == "Benchmark"]
        main_bench = [iv for iv in bench if iv.get("phase") == "main"]
        for mp, iv in zip(main_phases, main_bench):
            prep_s = mp.get("prep_s")
            if prep_s and prep_s > 0:
                prep_end_utc = pd.Timestamp(iv["start_wall"]) + timedelta(seconds=prep_s)
                all_intervals.append({
                    "phase": "prep",
                    "start_wall": iv["start_wall"],
                    "end_wall": prep_end_utc.isoformat(),
                    "slurm_job_id": jid,
                    "par_tag": par_tag,
                })
                iv["start_wall"] = prep_end_utc.isoformat()

        all_intervals.extend(bench)

    if not all_intervals:
        return ""

    # Group intervals by job (preserving order)
    seen_jids: list[str] = []
    jid_intervals: dict[str, list[dict[str, Any]]] = {}
    for iv in all_intervals:
        jid = iv.get("slurm_job_id", "")
        if jid not in jid_intervals:
            seen_jids.append(jid)
            jid_intervals[jid] = []
        jid_intervals[jid].append(iv)

    n_rows = len(seen_jids)
    row_height = 60
    chart_height = n_rows * row_height + 60

    # x-axis range from all intervals
    all_starts = [pd.Timestamp(iv["start_wall"]) for iv in all_intervals]
    all_ends = [pd.Timestamp(iv["end_wall"]) for iv in all_intervals]
    x_min = min(all_starts)
    x_max = max(all_ends)
    pad = (x_max - x_min) * 0.02
    x_range = [(x_min - pad).isoformat(), (x_max + pad).isoformat()]

    traces = []
    shapes = []
    annotations = []

    for row_idx, jid in enumerate(seen_jids):
        row_num = row_idx + 1
        xaxis_key = "x" if row_num == 1 else f"x{row_num}"
        yaxis_key = "y" if row_num == 1 else f"y{row_num}"
        yref = "y" if row_num == 1 else f"y{row_num}"
        xref = "x" if row_num == 1 else f"x{row_num}"

        # Invisible anchor trace per subplot
        traces.append({
            "x": [x_range[0], x_range[1]],
            "y": [0.5, 0.5],
            "xaxis": xaxis_key,
            "yaxis": yaxis_key,
            "type": "scatter", "mode": "lines",
            "line": {"color": "rgba(0,0,0,0)", "width": 0},
            "showlegend": False,
            "hoverinfo": "skip",
        })

        _STARTUP_FILL = "rgba(156,39,176,0.15)"
        _STARTUP_LINE = "#9c27b0"
        _PREP_FILL = "rgba(33,150,243,0.20)"
        _PREP_LINE = "#1565c0"

        main_idx = 0
        for iv in jid_intervals[jid]:
            phase = iv.get("phase", "")
            x0, x1 = iv["start_wall"], iv["end_wall"]

            if phase == "startup":
                fill = _STARTUP_FILL
                lc = _STARTUP_LINE
            elif phase == "warmup":
                fill = _WARMUP_FILL
                lc = _WARMUP_LINE
            elif phase == "prep":
                fill = _PREP_FILL
                lc = _PREP_LINE
            else:
                fill = _MAIN_REGION_COLORS[main_idx % len(_MAIN_REGION_COLORS)]
                lc = _MAIN_LINE_COLORS[main_idx % len(_MAIN_LINE_COLORS)]
                main_idx += 1

            shapes.append({
                "type": "rect",
                "xref": xref, "yref": yref,
                "x0": x0, "x1": x1,
                "y0": 0, "y1": 1,
                "fillcolor": fill, "line": {"color": lc, "width": 1.5},
                "layer": "below",
            })

            ts0 = pd.Timestamp(x0)
            ts1 = pd.Timestamp(x1)
            mid_x = (ts0 + (ts1 - ts0) / 2).isoformat()
            if phase == "startup":
                label = "Server Startup"
            elif phase == "warmup":
                label = "Warmup"
            elif phase == "prep":
                label = "Prep"
            else:
                label = "<b>Benchmark</b>"

            annotations.append({
                "x": mid_x, "y": 0.5, "yref": yref, "xref": xref,
                "text": label,
                "showarrow": False,
                "font": {"size": 11, "color": lc},
                "bgcolor": "rgba(255,255,255,0.85)",
                "borderpad": 4,
            })

    # Build layout with subplots sharing x-axis
    gap_px = 10
    domain_gap = gap_px / chart_height if chart_height > 0 else 0.02
    usable = 1.0 - domain_gap * (n_rows - 1) if n_rows > 1 else 1.0
    row_size = usable / n_rows

    layout: dict[str, Any] = {
        "height": chart_height,
        "margin": {"l": 180, "r": 20, "t": 10, "b": 35},
        "shapes": shapes,
        "annotations": annotations,
        "hovermode": "x",
        "showlegend": False,
    }

    for row_idx, jid in enumerate(seen_jids):
        row_num = row_idx + 1
        par_tag = jid_intervals[jid][0].get("par_tag", "")
        y_label = f"{jid}<br>{par_tag}" if par_tag else jid

        # Domains: top row first (reversed so row 0 is at top)
        y_start = 1.0 - (row_idx + 1) * row_size - row_idx * domain_gap
        y_end = 1.0 - row_idx * row_size - row_idx * domain_gap

        x_suffix = "" if row_num == 1 else str(row_num)
        y_suffix = "" if row_num == 1 else str(row_num)

        layout[f"xaxis{x_suffix}"] = {
            "type": "date",
            "showgrid": True,
            "gridcolor": "#eee",
            "range": x_range,
            "anchor": f"y{y_suffix}",
            "showticklabels": row_idx == n_rows - 1,
            "title": "Time (UTC)" if row_idx == n_rows - 1 else None,
            "showspikes": True,
            "spikemode": "across",
            "spikethickness": 1,
            "spikecolor": "#999",
            "spikedash": "dot",
        }
        if row_num > 1:
            layout[f"xaxis{x_suffix}"]["matches"] = "x"

        layout[f"yaxis{y_suffix}"] = {
            "domain": [max(0, y_start), y_end],
            "visible": True,
            "showticklabels": False,
            "showgrid": False,
            "fixedrange": True,
            "range": [0, 1],
            "anchor": f"x{x_suffix}",
        }

        # Horizontal label annotation to the left of each row
        y_mid = (y_start + y_end) / 2
        annotations.append({
            "x": 0, "xref": "paper", "xanchor": "right",
            "y": y_mid, "yref": "paper", "yanchor": "middle",
            "text": y_label,
            "showarrow": False,
            "font": {"size": 12, "color": "#333"},
            "align": "right",
        })

    traces_json = json.dumps(traces, default=str)
    layout_json = json.dumps(layout, default=str)
    return f"""
    <div id="phase-timeline" style="margin-top:16px"></div>
    <script>
    Plotly.newPlot("phase-timeline", {traces_json}, {layout_json}, {{responsive: true, displaylogo: false}});
    </script>"""


def _parse_framework(log_dir: Path) -> str:
    """Derive framework string like 'Dynamo/SGLang/Disagg' from config.yaml."""
    for candidate in [log_dir / "config.yaml", log_dir.parent / "config.yaml"]:
        if candidate.is_file():
            try:
                with open(candidate, "r") as f:
                    cfg = yaml.safe_load(f)
            except Exception:
                continue
            if not isinstance(cfg, dict):
                continue

            router = "Dynamo" if cfg.get("dynamo") else "—"

            container = ((cfg.get("model") or {}).get("container") or "").lower()
            if "sglang" in container:
                backend = "SGLang"
            elif "vllm" in container:
                backend = "vLLM"
            elif "trtllm" in container or "tensorrt" in container:
                backend = "TRTLLM"
            else:
                backend = container or "—"

            sglang_cfg = (cfg.get("backend") or {}).get("sglang_config") or {}
            has_disagg = False
            for role_cfg in [sglang_cfg.get("prefill") or {}, sglang_cfg.get("decode") or {}]:
                if role_cfg.get("disaggregation-mode"):
                    has_disagg = True
                    break
            mode = "Disagg" if has_disagg else "Agg"

            return f"{router}/{backend}/{mode}"
    return "—"


def _render_merged_table_rows(
    data_rows: list[list[str]],
    merge_cols: int | None = None,
) -> str:
    """Render table rows with rowspan merging on leading columns.

    *merge_cols* controls how many columns (from the left) are eligible for
    merging.  ``None`` means all columns.
    """
    if not data_rows:
        return ""
    n_cols = len(data_rows[0])
    if merge_cols is None:
        merge_cols = n_cols

    # For each column (left to right, up to merge_cols), compute rowspan
    # A cell is merged with the one above if all columns to its left are also
    # the same (i.e., we only merge within the same "group").
    spans: list[list[int]] = [[1] * n_cols for _ in data_rows]
    for col in range(min(merge_cols, n_cols)):
        for row in range(len(data_rows) - 1, 0, -1):
            left_same = all(
                data_rows[row][c] == data_rows[row - 1][c] for c in range(col)
            ) if col > 0 else True
            if left_same and data_rows[row][col] == data_rows[row - 1][col]:
                spans[row][col] = 0
                spans[row - 1][col] += spans[row][col] if spans[row][col] > 0 else 1

    # Re-accumulate: walk top-down so the topmost cell gets the full span
    for col in range(min(merge_cols, n_cols)):
        r = 0
        while r < len(data_rows):
            if spans[r][col] == 0:
                r += 1
                continue
            count = 1
            for r2 in range(r + 1, len(data_rows)):
                left_same = all(
                    data_rows[r2][c] == data_rows[r][c] for c in range(col)
                ) if col > 0 else True
                if left_same and data_rows[r2][col] == data_rows[r][col]:
                    count += 1
                else:
                    break
            spans[r][col] = count
            for r2 in range(r + 1, r + count):
                spans[r2][col] = 0
            r += count

    html_rows = []
    td_style = 'style="padding:6px 12px;vertical-align:top"'
    for r, row_data in enumerate(data_rows):
        cells = []
        for c, val in enumerate(row_data):
            if c < merge_cols and spans[r][c] == 0:
                continue
            rs = spans[r][c] if c < merge_cols else 1
            rs_attr = f' rowspan="{rs}"' if rs > 1 else ""
            cells.append(f"<td {td_style}{rs_attr}>{val}</td>")
        html_rows.append(f'<tr style="border-bottom:1px solid #eee">{"".join(cells)}</tr>')

    return "\n      ".join(html_rows)


def _compute_phase_power_stats(
    log_dir: Path,
    phases: list[dict[str, Any]],
) -> list[dict[str, float | None]]:
    """Return per-GPU peak and avg power (W) for each benchmark phase.

    Each element corresponds to the phase at the same index and contains
    ``peak_power_per_gpu`` and ``avg_power_per_gpu`` (or ``None`` when data
    is unavailable).
    """
    from srt_upload_helper import _read_perf_csv
    from datetime import timedelta

    csv_files = sorted(log_dir.glob("perf_samples_*.csv"))
    if not csv_files or not phases:
        return [{"peak_power_per_gpu": None, "avg_power_per_gpu": None}] * len(phases)

    dfs = []
    for p in csv_files:
        try:
            dfs.append(_read_perf_csv(p)[["timestamp", "power_w"]])
        except Exception:
            continue
    if not dfs:
        return [{"peak_power_per_gpu": None, "avg_power_per_gpu": None}] * len(phases)
    all_power = pd.concat(dfs, ignore_index=True)
    if all_power.empty:
        return [{"peak_power_per_gpu": None, "avg_power_per_gpu": None}] * len(phases)

    tz_offset = _compute_tz_offset(log_dir)

    results: list[dict[str, float | None]] = []
    for p in phases:
        st_str = p.get("start_time")
        dur = p.get("duration_s")
        if not st_str or not dur:
            results.append({"peak_power_per_gpu": None, "avg_power_per_gpu": None})
            continue
        start_naive = pd.Timestamp(st_str)
        start_utc = (start_naive + tz_offset).tz_localize("UTC")
        prep_s = p.get("prep_s", 0.0) or 0.0
        start_utc = start_utc + timedelta(seconds=prep_s)
        end_utc = start_utc + timedelta(seconds=dur)
        mask = (all_power["timestamp"] >= start_utc) & (all_power["timestamp"] <= end_utc)
        window = all_power.loc[mask, "power_w"]
        if window.empty:
            results.append({"peak_power_per_gpu": None, "avg_power_per_gpu": None})
        else:
            results.append({
                "peak_power_per_gpu": float(window.max()),
                "avg_power_per_gpu": float(window.mean()),
            })
    return results


def _build_perf_table_html(resolved_logs: list[Path]) -> str:
    """Build the Perf section HTML table for the Overview tab."""
    from srt_upload_helper import _parse_parallelization_tag

    _MODEL_PATH_MAP = {
        "dsfp4": "nvidia/DeepSeek-R1-0528-NVFP4-v2",
        "dsfp8": "deepseek-ai/DeepSeek-R1-0528",
    }

    def _parse_model_name(log_dir: Path) -> str:
        for candidate in [log_dir / "config.yaml", log_dir.parent / "config.yaml"]:
            if candidate.is_file():
                try:
                    with open(candidate, "r") as f:
                        cfg = yaml.safe_load(f)
                    path = (cfg.get("model") or {}).get("path", "")
                    if path:
                        return _MODEL_PATH_MAP.get(str(path), str(path))
                except Exception:
                    pass
        return "—"

    def _detect_rack(log_dir: Path) -> str:
        for csv_path in sorted(log_dir.glob("perf_samples_*.csv"))[:1]:
            m = _RACK_RE.search(csv_path.name)
            if m:
                return m.group(1)
            m = _RACK_RE_ALT.search(csv_path.name)
            if m:
                return m.group(1)
        return "—"

    data_rows: list[list[str]] = []
    for ld in resolved_logs:
        jid = _slurm_job_id_for_log_dir(ld)
        rack = _detect_rack(ld)
        par_tag = _parse_parallelization_tag(ld)
        framework = _parse_framework(ld)
        model_name = _parse_model_name(ld)
        gpu_counts = _gpu_counts_from_config(ld)
        total_gpus = gpu_counts["total"]
        prefill_gpus = gpu_counts["prefill"]
        decode_gpus = gpu_counts["decode"]
        phases = _parse_benchmark_phases(ld)
        power_stats = _compute_phase_power_stats(ld, phases)

        for p, ps in zip(phases, power_stats):
            conc = str(p.get("concurrency", "—"))
            dur = p.get("duration_s")
            dur_str = f"{dur / 60:.1f} min" if dur else "—"
            n_req = str(p.get("num_requests", "—"))
            tps_total = p.get("total_token_tps")
            tps_gpu = f"{tps_total / total_gpus:.1f}" if tps_total and total_gpus else "—"
            median_tpot = p.get("median_tpot_ms")
            tps_user = f"{1000.0 / median_tpot:.1f}" if median_tpot else "—"
            phase_label = p.get("phase", "—")
            start_time = str(p.get("start_time", "—"))
            isl_val = str(p.get("isl", "—"))
            osl_val = str(p.get("osl", "—"))

            input_tokens = p.get("total_input_tokens")
            gen_tokens = p.get("total_generated_tokens")
            input_tps_gpu = (
                f"{input_tokens / dur / prefill_gpus:.1f}"
                if input_tokens and dur and prefill_gpus else "—"
            )
            output_tps_gpu = (
                f"{gen_tokens / dur / decode_gpus:.1f}"
                if gen_tokens and dur and decode_gpus else "—"
            )

            peak_pw = ps["peak_power_per_gpu"]
            avg_pw = ps["avg_power_per_gpu"]
            peak_pw_str = f"{peak_pw:.0f}" if peak_pw is not None else "—"
            avg_pw_str = f"{avg_pw:.0f}" if avg_pw is not None else "—"

            prep = p.get("prep_s")
            prep_str = f"{prep / 60:.1f} min" if prep is not None else "—"

            data_rows.append([
                html_mod.escape(str(jid)),
                html_mod.escape(rack),
                html_mod.escape(framework),
                html_mod.escape(model_name),
                html_mod.escape(par_tag),
                html_mod.escape(phase_label),
                html_mod.escape(start_time),
                dur_str, prep_str, conc, n_req, isl_val, osl_val,
                tps_user, tps_gpu, input_tps_gpu, output_tps_gpu,
                peak_pw_str, avg_pw_str,
            ])

    if not data_rows:
        return ""

    body = _render_merged_table_rows(data_rows, merge_cols=5)
    return f"""
  <div style="background:#fff;border-radius:6px;box-shadow:0 1px 4px rgba(0,0,0,.08);padding:16px 24px;margin-top:16px">
    <h3 style="margin:0 0 12px;font-size:16px;font-weight:600;color:#444">Benchmark Overview</h3>
    <table style="width:100%;border-collapse:collapse;font-size:13px;color:#555">
      <thead>
        <tr style="border-bottom:2px solid #76b900;text-align:left">
          <th style="padding:8px 12px;font-weight:600;color:#333">Slurm Job ID</th>
          <th style="padding:8px 12px;font-weight:600;color:#333">Rack</th>
          <th style="padding:8px 12px;font-weight:600;color:#333">Framework</th>
          <th style="padding:8px 12px;font-weight:600;color:#333">Model</th>
          <th style="padding:8px 12px;font-weight:600;color:#333">Parallelization</th>
          <th style="padding:8px 12px;font-weight:600;color:#333">Phase</th>
          <th style="padding:8px 12px;font-weight:600;color:#333">Start Time</th>
          <th style="padding:8px 12px;font-weight:600;color:#333">Duration</th>
          <th style="padding:8px 12px;font-weight:600;color:#333">Prep Time</th>
          <th style="padding:8px 12px;font-weight:600;color:#333">Concurrency</th>
          <th style="padding:8px 12px;font-weight:600;color:#333"># Requests</th>
          <th style="padding:8px 12px;font-weight:600;color:#333">ISL</th>
          <th style="padding:8px 12px;font-weight:600;color:#333">OSL</th>
          <th style="padding:8px 12px;font-weight:600;color:#333">TPS/User</th>
          <th style="padding:8px 12px;font-weight:600;color:#333">TPS/GPU</th>
          <th style="padding:8px 12px;font-weight:600;color:#333">Input TPS/GPU</th>
          <th style="padding:8px 12px;font-weight:600;color:#333">Output TPS/GPU</th>
          <th style="padding:8px 12px;font-weight:600;color:#333">Peak Power/GPU (W)</th>
          <th style="padding:8px 12px;font-weight:600;color:#333">Avg Power/GPU (W)</th>
        </tr>
      </thead>
      <tbody>
      {body}
      </tbody>
    </table>
    <div style="margin-top:8px;font-size:11px;color:#999">
      TPS/User = 1 / Median TPOT<br>
      TPS/GPU = Total Token Throughput / Total GPUs<br>
      Input TPS/GPU = Input Token Throughput / Prefill GPUs<br>
      Output TPS/GPU = Output Token Throughput / Decode GPUs<br>
      Prep Time = Wall time minus benchmark duration (request generation + tokenization overhead)<br>
      Peak Power/GPU = Max power_w across all GPUs during the measurement window (excludes prep)<br>
      Avg Power/GPU = Mean power_w across all GPUs during the measurement window (excludes prep)
    </div>
    {_build_phase_timeline_html(resolved_logs)}
  </div>"""


def _cleanup_summary_bar(html_path: Path, resolved_logs: list[Path]) -> None:
    """Post-process the summary bar:
    - Move Hosts list into an Overview tab
    - Convert Duration from seconds to minutes
    - Clarify Avg Util label
    """
    html = html_path.read_text(encoding="utf-8")

    # Extract the Hosts value from the summary bar
    hosts_match = re.search(
        r'<div class="stat">Hosts: <span class="stat-value">(.*?)</span></div>',
        html,
    )
    hosts_value = hosts_match.group(1) if hosts_match else ""

    # Remove the Hosts stat from the summary bar
    if hosts_match:
        html = html[:hosts_match.start()] + html[hosts_match.end():]

    # Convert Duration from seconds to minutes
    dur_match = re.search(
        r'<div class="stat">Duration: <span class="stat-value">([\d.]+)s</span></div>',
        html,
    )
    if dur_match:
        dur_s = float(dur_match.group(1))
        dur_min = dur_s / 60
        new_dur = f'<div class="stat">Duration: <span class="stat-value">{dur_min:.1f} min</span></div>'
        html = html[:dur_match.start()] + new_dur + html[dur_match.end():]

    # Rename "Avg Util" to "Avg GPU Util (all GPUs)"
    html = html.replace(
        'Avg Util: <span class="stat-value">',
        'Avg GPU Util: <span class="stat-value">',
    )

    # Replace the entire tab bar with reordered and renamed tabs
    tab_bar_match = re.search(r'<div class="tab-bar">\n.*?\n</div>', html, re.DOTALL)
    if tab_bar_match:
        new_tab_bar = """<div class="tab-bar">
<button class="tab active" data-tab="overview" onclick="switchTab(this)">Overview</button>
<button class="tab" data-tab="performance" onclick="switchTab(this)">Performance</button>
<button class="tab" data-tab="all" onclick="switchTab(this)">GPU Power (All)</button>
<button class="tab" data-tab="role:prefill" onclick="switchTab(this)">GPU Power (Prefill)</button>
<button class="tab" data-tab="role:decode" onclick="switchTab(this)">GPU Power (Decode)</button>
<button class="tab" data-tab="per-node" onclick="switchTab(this)">GPU Power (Per-Node)</button>
</div>"""
        html = html[:tab_bar_match.start()] + new_tab_bar + html[tab_bar_match.end():]

    # Build the hosts table with worker grouping matching chart legends
    from srt_upload_helper import (
        _split_composite_host_key, _parse_parallelization_tag,
        _parse_worker_counts,
    )
    hosts_parsed = _extract_js_var(html, "ALL_HOSTS")
    roles_parsed = _extract_js_var(html, "HOST_ROLES")
    all_hosts_list: list[str] = json.loads(hosts_parsed[0]) if hosts_parsed else []
    host_roles_map: dict[str, list[str]] = json.loads(roles_parsed[0]) if roles_parsed else {}

    # Group hosts by jid and role
    jid_role_hosts: dict[str, dict[str, list[str]]] = {}
    for hk in all_hosts_list:
        jid, bare = _split_composite_host_key(hk)
        jid = jid or "—"
        for role in host_roles_map.get(hk, []):
            if role in ("prefill", "decode"):
                jid_role_hosts.setdefault(jid, {}).setdefault(role, []).append(bare)

    setup_data_rows: list[list[str]] = []
    for jid in sorted(jid_role_hosts.keys()):
        par_tag = ""
        wc: dict[str, int] = {}
        gpus_per_node = 1
        role_data = jid_role_hosts[jid]
        for ld in resolved_logs:
            if _slurm_job_id_for_log_dir(ld) == jid:
                par_tag = _parse_parallelization_tag(ld)
                wc = _parse_worker_counts(ld)
                gc = _gpu_counts_from_config(ld)
                total_hosts = len(role_data.get("prefill", [])) + len(role_data.get("decode", []))
                if total_hosts > 0 and gc["total"] > 0:
                    gpus_per_node = gc["total"] // total_hosts
                break
        for role in ["prefill", "decode"]:
            hosts = sorted(role_data.get(role, []))
            if not hosts:
                continue
            n_workers = wc.get(role, len(hosts))
            n_workers = min(n_workers, len(hosts))
            if n_workers <= 0:
                n_workers = len(hosts)
            chunk = max(1, len(hosts) // n_workers)
            role_cap = role.capitalize()
            for wi in range(n_workers):
                start = wi * chunk
                end = start + chunk if wi < n_workers - 1 else len(hosts)
                worker_hosts = hosts[start:end]
                worker_label = f"{role_cap}{wi}"
                num_gpus = str(len(worker_hosts) * gpus_per_node)
                hosts_str = ", ".join(worker_hosts)
                setup_data_rows.append([jid, par_tag, worker_label, num_gpus, hosts_str])
    table_body = _render_merged_table_rows(setup_data_rows, merge_cols=2)

    perf_table_html = _build_perf_table_html(resolved_logs)

    overview_panel = f"""<div id="overview-panel" style="display:block;padding:16px 24px">
  <div style="background:#fff;border-radius:6px;box-shadow:0 1px 4px rgba(0,0,0,.08);padding:16px 24px">
    <h3 style="margin:0 0 12px;font-size:16px;font-weight:600;color:#444">System Setup</h3>
    <table style="width:100%;border-collapse:collapse;font-size:13px;color:#555">
      <thead>
        <tr style="border-bottom:2px solid #76b900;text-align:left">
          <th style="padding:8px 12px;font-weight:600;color:#333">Slurm Job ID</th>
          <th style="padding:8px 12px;font-weight:600;color:#333">Parallelization</th>
          <th style="padding:8px 12px;font-weight:600;color:#333">Worker</th>
          <th style="padding:8px 12px;font-weight:600;color:#333">Num GPU</th>
          <th style="padding:8px 12px;font-weight:600;color:#333">Hosts</th>
        </tr>
      </thead>
      <tbody>
      {table_body}
      </tbody>
    </table>
  </div>
  {perf_table_html}
</div>

"""
    html = html.replace(
        '<div class="panels">',
        overview_panel + '<div class="panels" style="display:none">',
    )

    # Set initial tab mode to overview
    html = html.replace(
        'var currentTabMode = "all";',
        'var currentTabMode = "overview";',
    )

    # Add JS to show/hide overview panel in switchTab
    html = html.replace(
        'var selectorBar = document.getElementById("node-selector-bar");',
        'var selectorBar = document.getElementById("node-selector-bar");\n'
        '  var overviewPanel = document.getElementById("overview-panel");\n'
        '  var chartsArea = document.querySelector(".panels");\n'
        '  var toolbar = document.querySelector(".toolbar");',
    )
    html = html.replace(
        'if (tab === "per-node") {',
        'if (tab === "overview") {\n'
        '    overviewPanel.style.display = "block";\n'
        '    chartsArea.style.display = "none";\n'
        '    toolbar.style.display = "none";\n'
        '    selectorBar.classList.remove("visible");\n'
        '    return;\n'
        '  }\n'
        '  overviewPanel.style.display = "none";\n'
        '  chartsArea.style.display = "flex";\n'
        '  toolbar.style.display = "flex";\n\n'
        '  if (tab === "per-node") {',
    )

    # Also hide overview on resetAll
    html = html.replace(
        'document.getElementById("node-selector-bar").classList.remove("visible");',
        'document.getElementById("node-selector-bar").classList.remove("visible");\n'
        '  document.getElementById("overview-panel").style.display = "none";\n'
        '  document.querySelector(".panels").style.display = "flex";',
    )

    # Hide toolbar by default (Overview is the initial tab)
    html = html.replace('<div class="toolbar">', '<div class="toolbar" style="display:none">')

    # Hide Hosts, GPUs, Metrics filter dropdowns (keep in DOM for JS filtering)
    for filter_id in ["host-filter", "gpu-filter", "metric-filter"]:
        html = html.replace(
            f'<details class="filter-group" id="{filter_id}">',
            f'<details class="filter-group" id="{filter_id}" style="display:none">',
        )

    # Replace cross-panel sync with a debounced version to prevent browser crashes
    sync_start = html.find("// Link x-axis zoom and legend clicks across panels")
    sync_end_marker = "\nfunction getChecked"
    if sync_start > 0:
        sync_end = html.find(sync_end_marker, sync_start)
        if sync_end > sync_start:
            debounced_sync = """// Debounced cross-panel x-axis sync
var _syncTimer = null;
for (var i = 0; i < PANEL_COUNT; i++) {
  (function(idx) {
    var el = document.getElementById("panel-" + idx);
    el.on("plotly_relayout", function(ed) {
      if (_syncTimer) clearTimeout(_syncTimer);
      _syncTimer = setTimeout(function() {
        var update = {};
        if (ed["xaxis.range[0]"] !== undefined) {
          update["xaxis.range[0]"] = ed["xaxis.range[0]"];
          update["xaxis.range[1]"] = ed["xaxis.range[1]"];
        }
        if (ed["xaxis.autorange"] !== undefined) {
          update["xaxis.autorange"] = ed["xaxis.autorange"];
        }
        if (Object.keys(update).length > 0) {
          for (var j = 0; j < PANEL_COUNT; j++) {
            if (j !== idx) Plotly.relayout("panel-" + j, update);
          }
        }
      }, 300);
    });
  })(i);
}
"""
            html = html[:sync_start] + debounced_sync + html[sync_end:]

    html_path.write_text(html, encoding="utf-8")
    logger.info("Cleaned up summary bar")


_PREFILL_THROUGHPUT_RE = re.compile(
    r"\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\s+"
    r"([^\]]+)\].*?input throughput \(token/s\):\s+([\d.]+)"
)
_DECODE_THROUGHPUT_RE = re.compile(
    r"\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\s+"
    r"([^\]]+)\].*?gen throughput \(token/s\):\s+([\d.]+)"
)
_WORKER_FILE_RE = re.compile(r"^(.+?)_(prefill|decode)_\w+\.out$")


def _compute_tz_offset(log_dir: Path) -> pd.Timedelta:
    """Compute the timezone offset between naive log times and UTC perf CSV times."""
    from srt_upload_helper import _read_perf_csv, _host_from_csv_name, _parse_benchmark_log
    from datetime import timedelta

    csv_files = sorted(log_dir.glob("perf_samples_*.csv"))
    if not csv_files:
        return pd.Timedelta(0)
    try:
        df = _read_perf_csv(csv_files[0])
        csv_t0 = df["timestamp"].min()
    except Exception:
        return pd.Timedelta(0)

    raw_intervals = _parse_benchmark_log(log_dir)
    if raw_intervals:
        bench_t0_naive = raw_intervals[0]["start"]
        delta_s = (csv_t0.tz_localize(None) - bench_t0_naive).total_seconds()
        hours = round(delta_s / 3600)
        return pd.Timedelta(hours=hours)
    return pd.Timedelta(0)


def _parse_worker_throughput(
    log_dir: Path, tz_offset: pd.Timedelta,
) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    """Parse throughput from worker logs, returning (prefill_data, decode_data).

    Keys are ``<hostname>_<worker_suffix>/<rank>`` (e.g.
    ``node-b15-c03_prefill_w2/DP0 TP0 EP0``).  Each rank within a worker
    log file is a separate trace.  Multiple lines with the same rank and
    timestamp are averaged.

    DataFrames have columns ``timestamp`` (UTC) and ``throughput``.
    """
    prefill_data: dict[str, pd.DataFrame] = {}
    decode_data: dict[str, pd.DataFrame] = {}

    for f in sorted(log_dir.iterdir()):
        m = _WORKER_FILE_RE.match(f.name)
        if not m:
            continue
        hostname, role = m.group(1), m.group(2)
        worker_key = f.stem

        pattern = _PREFILL_THROUGHPUT_RE if role == "prefill" else _DECODE_THROUGHPUT_RE

        try:
            text = f.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue

        rank_records: dict[str, list[tuple]] = {}
        for match in pattern.finditer(text):
            ts_naive = pd.Timestamp(match.group(1))
            ts_utc = (ts_naive + tz_offset).tz_localize("UTC")
            rank = match.group(2).strip()
            val = float(match.group(3))
            rank_records.setdefault(rank, []).append((ts_utc, val))

        target = prefill_data if role == "prefill" else decode_data
        for rank, records in sorted(rank_records.items()):
            df = pd.DataFrame(records, columns=["timestamp", "throughput"])
            df = df.groupby("timestamp", as_index=False).agg({"throughput": "max"})
            trace_key = f"{worker_key}/{rank}"
            target[trace_key] = df

    return prefill_data, decode_data


def _build_throughput_chart(
    data: dict[str, pd.DataFrame],
    title: str,
    chart_id: str,
    jid_label: str,
    color: str,
    bench_intervals: list[dict[str, Any]],
) -> str:
    """Build a Plotly throughput chart HTML snippet."""
    if not data:
        return ""

    traces = []
    for worker_key in sorted(data.keys()):
        df = data[worker_key].sort_values("timestamp")
        x = [t.isoformat() for t in df["timestamp"]]
        y = df["throughput"].tolist()
        traces.append({
            "x": x, "y": y,
            "type": "scatter", "mode": "lines",
            "name": f"{jid_label} {worker_key}",
            "line": {"width": 1.5},
            "hovertemplate": (
                f"<b>{jid_label} {worker_key}</b><br>"
                "Time (UTC): %{x|%Y-%m-%d %H:%M:%S}<br>"
                f"{title}: %{{y:.1f}} tok/s"
                "<extra></extra>"
            ),
        })

    # Avg trace across all workers
    all_dfs = list(data.values())
    if len(all_dfs) > 1:
        combined = pd.concat(all_dfs, ignore_index=True).sort_values("timestamp")
        agg = (
            combined.groupby(pd.Grouper(key="timestamp", freq="1s"))
            .agg({"throughput": "mean"})
            .dropna()
            .sort_index()
        )
        if not agg.empty:
            x = [t.isoformat() for t in agg.index]
            y = agg["throughput"].tolist()
            traces.append({
                "x": x, "y": y,
                "type": "scatter", "mode": "lines",
                "name": f"{jid_label} Avg",
                "line": {"color": color, "width": 3},
                "hovertemplate": (
                    f"<b>{jid_label} Avg</b><br>"
                    "Time (UTC): %{x|%Y-%m-%d %H:%M:%S}<br>"
                    f"{title}: %{{y:.1f}} tok/s"
                    "<extra></extra>"
                ),
            })

    # Build benchmark phase shapes
    shapes = []
    annotations = []
    main_idx = 0
    for iv in bench_intervals:
        is_warmup = iv.get("phase") == "warmup"
        x0, x1 = iv["start_wall"], iv["end_wall"]
        if is_warmup:
            fill = _WARMUP_FILL
            lc = _WARMUP_LINE
        else:
            fill = _MAIN_REGION_COLORS[main_idx % len(_MAIN_REGION_COLORS)]
            lc = _MAIN_LINE_COLORS[main_idx % len(_MAIN_LINE_COLORS)]
            main_idx += 1
        shapes.append({
            "type": "rect", "x0": x0, "x1": x1,
            "y0": 0, "y1": 1, "yref": "paper",
            "fillcolor": fill, "line": {"width": 0}, "layer": "below",
        })

    layout = {
        "height": 300,
        "margin": {"l": 60, "r": 180, "t": 36, "b": 40},
        "showlegend": True,
        "hovermode": "x unified",
        "legend": {
            "orientation": "v", "x": 1.02, "xanchor": "left",
            "y": 1, "yanchor": "top",
            "bgcolor": "rgba(255,255,255,0.9)",
            "bordercolor": "#76b900", "borderwidth": 1,
            "font": {"size": 11},
        },
        "xaxis": {"type": "date", "showgrid": True, "gridcolor": "#eee"},
        "yaxis": {
            "title": f"{title} (tok/s)",
            "showgrid": True, "gridcolor": "#eee",
        },
        "shapes": shapes,
    }

    traces_json = json.dumps(traces, default=str)
    layout_json = json.dumps(layout, default=str)
    return f"""
    <div style="background:#fff;border-radius:6px;box-shadow:0 1px 4px rgba(0,0,0,.08);
      overflow:hidden;margin-bottom:12px">
      <div style="padding:8px 16px;font-size:14px;font-weight:600;color:#444;
        border-bottom:1px solid #f0f0f0;background:#fafafa">{title} (tok/s)</div>
      <div id="{chart_id}"></div>
    </div>
    <script>
    Plotly.newPlot("{chart_id}", {traces_json}, {layout_json},
      {{responsive: true, displaylogo: false}});
    </script>"""


def _inject_performance_tab(html_path: Path, resolved_logs: list[Path]) -> None:
    """Add a Performance tab with merged prefill/decode throughput charts across all runs."""
    from srt_upload_helper import (
        _parse_benchmark_log, _sync_perf_timestamps, _read_perf_csv,
        _host_from_csv_name, _parse_parallelization_tag,
        _COMPOSITE_HOST_SEP,
    )

    all_prefill_traces: list[dict] = []
    all_decode_traces: list[dict] = []
    all_prefill_rank_traces: list[dict] = []
    all_decode_rank_traces: list[dict] = []
    all_bench_intervals: list[dict[str, Any]] = []

    for ld in resolved_logs:
        jid = _slurm_job_id_for_log_dir(ld)
        par_tag = _parse_parallelization_tag(ld)
        jid_label = f"{jid} {par_tag}" if par_tag else jid

        tz_offset = _compute_tz_offset(ld)
        prefill_data, decode_data = _parse_worker_throughput(ld, tz_offset)

        # Get benchmark intervals for overlay
        csv_files = sorted(ld.glob("perf_samples_*.csv"))
        hd: dict[str, pd.DataFrame] = {}
        for csv_path in csv_files:
            try:
                hostname = _host_from_csv_name(csv_path.name)
                df = _read_perf_csv(csv_path)
                if not df.empty:
                    hd[f"{jid}{_COMPOSITE_HOST_SEP}{hostname}"] = df
            except Exception:
                pass

        raw_intervals = _parse_benchmark_log(ld)
        _, bench = _sync_perf_timestamps(hd, raw_intervals or None)
        for iv in bench:
            iv["slurm_job_id"] = jid
        all_bench_intervals.extend(bench)

        # Determine aggregation method per role from parallelization config
        from srt_upload_helper import _sglang_parallel_label, _parse_worker_counts
        role_agg_method: dict[str, str] = {}
        for candidate in [ld / "config.yaml", ld.parent / "config.yaml"]:
            if candidate.is_file():
                try:
                    with open(candidate, "r") as _f:
                        _cfg = yaml.safe_load(_f)
                    sglang = (_cfg.get("backend") or {}).get("sglang_config") or {}
                    for role_name in ["prefill", "decode"]:
                        plabel = _sglang_parallel_label(sglang.get(role_name) or {})
                        role_agg_method[role_name] = "sum" if plabel.startswith("DEP") else "mean"
                    break
                except Exception:
                    pass

        wc = _parse_worker_counts(ld)

        def _build_worker_traces(
            rank_data: dict[str, pd.DataFrame],
            role: str,
            title_prefix: str,
            target_traces: list[dict],
            target_rank_traces: list[dict],
        ) -> None:
            """Group per-rank traces into worker groups and build aggregated + per-rank traces."""
            agg_method = role_agg_method.get(role, "mean")

            host_ranks: dict[str, list[str]] = {}
            for key in rank_data:
                host_part = key.split("/")[0]
                host_ranks.setdefault(host_part, []).append(key)

            sorted_hosts = sorted(host_ranks.keys())
            n_workers = wc.get(role, len(sorted_hosts))
            n_workers = min(n_workers, len(sorted_hosts))
            if n_workers <= 0:
                n_workers = len(sorted_hosts)
            chunk = max(1, len(sorted_hosts) // n_workers)

            role_cap = role.capitalize()
            for wi in range(n_workers):
                start_i = wi * chunk
                end_i = start_i + chunk if wi < n_workers - 1 else len(sorted_hosts)
                worker_hosts = sorted_hosts[start_i:end_i]

                rank_dfs = []
                for h in worker_hosts:
                    for rk in sorted(host_ranks.get(h, [])):
                        rank_dfs.append(rank_data[rk])
                        # Per-rank trace (hidden by default)
                        rdf = rank_data[rk].sort_values("timestamp")
                        x = [t.isoformat() for t in rdf["timestamp"]]
                        y = rdf["throughput"].tolist()
                        rlabel = f"{jid_label} {rk}"
                        target_rank_traces.append({
                            "x": x, "y": y,
                            "type": "scatter", "mode": "lines",
                            "name": rlabel,
                            "visible": False,
                            "line": {"width": 1, "dash": "dot"},
                            "hovertemplate": (
                                f"<b>{rlabel}</b><br>"
                                "Time (UTC): %{x|%Y-%m-%d %H:%M:%S}<br>"
                                f"{title_prefix} Throughput: %{{y:.1f}} tok/s"
                                "<extra></extra>"
                            ),
                        })

                if not rank_dfs:
                    continue

                combined = pd.concat(rank_dfs, ignore_index=True)
                agg_df = (
                    combined.groupby("timestamp", as_index=False)
                    .agg({"throughput": agg_method})
                    .sort_values("timestamp")
                )

                wlabel = f"{jid_label} {role_cap}{wi}"
                x = [t.isoformat() for t in agg_df["timestamp"]]
                y = agg_df["throughput"].tolist()
                target_traces.append({
                    "x": x, "y": y,
                    "type": "scatter", "mode": "lines",
                    "name": wlabel,
                    "line": {"width": 2},
                    "hovertemplate": (
                        f"<b>{wlabel}</b> ({agg_method})<br>"
                        "Time (UTC): %{x|%Y-%m-%d %H:%M:%S}<br>"
                        f"{title_prefix} Throughput: %{{y:.1f}} tok/s"
                        "<extra></extra>"
                    ),
                })

        _build_worker_traces(prefill_data, "prefill", "Prefill", all_prefill_traces, all_prefill_rank_traces)
        _build_worker_traces(decode_data, "decode", "Decode", all_decode_traces, all_decode_rank_traces)

    if not all_prefill_traces and not all_decode_traces:
        return

    # Split main intervals into prep + main
    prep_ivs = _split_prep_from_main(all_bench_intervals, resolved_logs)
    all_bench_intervals.extend(prep_ivs)

    # Build benchmark phase shapes and annotations for overlay
    shapes = []
    bench_annotations = []
    main_idx = 0
    for iv in all_bench_intervals:
        phase = iv.get("phase", "main")
        x0, x1 = iv["start_wall"], iv["end_wall"]
        jid = iv.get("slurm_job_id", "")
        par_tag_iv = ""
        for ld2 in resolved_logs:
            if _slurm_job_id_for_log_dir(ld2) == jid:
                par_tag_iv = _parse_parallelization_tag(ld2)
                break
        jid_display = f"{jid} {par_tag_iv}" if par_tag_iv else jid

        if phase == "warmup":
            fill = _WARMUP_FILL
            lc = _WARMUP_LINE
        elif phase == "prep":
            fill = _PREP_FILL
            lc = _PREP_LINE
        else:
            fill = _MAIN_REGION_COLORS[main_idx % len(_MAIN_REGION_COLORS)]
            lc = _MAIN_LINE_COLORS[main_idx % len(_MAIN_LINE_COLORS)]
            main_idx += 1
        shapes.append({
            "type": "rect", "x0": x0, "x1": x1,
            "y0": 0, "y1": 1, "yref": "paper",
            "fillcolor": fill, "line": {"width": 0}, "layer": "below",
        })

        ts0 = pd.Timestamp(x0)
        ts1 = pd.Timestamp(x1)
        mid_x = (ts0 + (ts1 - ts0) / 2).isoformat()
        if phase == "warmup":
            label = f"Warmup {jid_display}"
        elif phase == "prep":
            label = f"Prep {jid_display}"
        else:
            label = f"<b>{jid_display}</b>"
        bench_annotations.append({
            "x": mid_x, "y": 1.06, "yref": "paper", "xref": "x",
            "text": label,
            "showarrow": False,
            "font": {"size": 10, "color": lc},
            "bgcolor": "rgba(255,255,255,0.85)",
            "borderpad": 3,
        })

    # Compute global x range from perf CSV data to align with GPU power charts
    global_x_min = None
    global_x_max = None
    for ld in resolved_logs:
        for csv_path in sorted(ld.glob("perf_samples_*.csv")):
            try:
                df = _read_perf_csv(csv_path)
                if df.empty:
                    continue
                t_min = df["timestamp"].min()
                t_max = df["timestamp"].max()
                if global_x_min is None or t_min < global_x_min:
                    global_x_min = t_min
                if global_x_max is None or t_max > global_x_max:
                    global_x_max = t_max
            except Exception:
                pass
    x_range_setting = {}
    if global_x_min is not None and global_x_max is not None:
        x_range_setting = {"range": [global_x_min.isoformat(), global_x_max.isoformat()]}

    def _make_chart(traces: list[dict], title: str, chart_id: str) -> str:
        xaxis_cfg = {"type": "date", "showgrid": True, "gridcolor": "#eee"}
        xaxis_cfg.update(x_range_setting)
        layout = {
            "height": 300,
            "margin": {"l": 60, "r": 180, "t": 36, "b": 40},
            "showlegend": True,
            "hovermode": "x unified",
            "legend": {
                "orientation": "v", "x": 1.02, "xanchor": "left",
                "y": 1, "yanchor": "top",
                "bgcolor": "rgba(255,255,255,0.9)",
                "bordercolor": "#76b900", "borderwidth": 1,
                "font": {"size": 11},
            },
            "xaxis": xaxis_cfg,
            "yaxis": {"title": f"{title} (tok/s)", "showgrid": True, "gridcolor": "#eee"},
            "shapes": shapes,
            "annotations": bench_annotations,
        }
        if bench_annotations:
            layout["margin"]["t"] = max(layout["margin"]["t"], 48)
        t_json = json.dumps(traces, default=str)
        l_json = json.dumps(layout, default=str)
        return (
            f'<div style="background:#fff;border-radius:6px;box-shadow:0 1px 4px rgba(0,0,0,.08);'
            f'overflow:hidden;margin-bottom:12px">'
            f'<div style="padding:8px 16px;font-size:14px;font-weight:600;color:#444;'
            f'border-bottom:1px solid #f0f0f0;background:#fafafa">{title} (tok/s)</div>'
            f'<div id="{chart_id}"></div></div>'
            f'<script>Plotly.newPlot("{chart_id}", {t_json}, {l_json},'
            f'{{responsive: true, displaylogo: false}});</script>'
        )

    n_prefill_worker = len(all_prefill_traces)
    n_decode_worker = len(all_decode_traces)

    charts_parts = []
    if all_prefill_traces or all_prefill_rank_traces:
        charts_parts.append(_make_chart(
            all_prefill_traces + all_prefill_rank_traces,
            "Prefill Throughput", "prefill-tp",
        ))
    if all_decode_traces or all_decode_rank_traces:
        charts_parts.append(_make_chart(
            all_decode_traces + all_decode_rank_traces,
            "Decode Throughput", "decode-tp",
        ))

    html = html_path.read_text(encoding="utf-8")

    toggle_html = (
        '<div style="padding:12px 0;display:flex;align-items:center;gap:8px">'
        '<label style="display:flex;align-items:center;gap:6px;font-size:13px;cursor:pointer">'
        '<input type="checkbox" id="show-per-rank-cb" onchange="toggleRankTraces()" '
        'style="accent-color:#76b900"> Show per-rank traces</label></div>'
    )

    toggle_js = f"""
    <script>
    var PREFILL_WORKER_COUNT = {n_prefill_worker};
    var PREFILL_RANK_COUNT = {len(all_prefill_rank_traces)};
    var DECODE_WORKER_COUNT = {n_decode_worker};
    var DECODE_RANK_COUNT = {len(all_decode_rank_traces)};
    function toggleRankTraces() {{
      var show = document.getElementById("show-per-rank-cb").checked;
      var vis = show ? true : false;
      var pEl = document.getElementById("prefill-tp");
      if (pEl && PREFILL_RANK_COUNT > 0) {{
        var indices = [];
        var vals = [];
        for (var i = PREFILL_WORKER_COUNT; i < PREFILL_WORKER_COUNT + PREFILL_RANK_COUNT; i++) {{
          indices.push(i);
          vals.push(vis);
        }}
        Plotly.restyle("prefill-tp", {{visible: vals}}, indices);
      }}
      var dEl = document.getElementById("decode-tp");
      if (dEl && DECODE_RANK_COUNT > 0) {{
        var indices = [];
        var vals = [];
        for (var i = DECODE_WORKER_COUNT; i < DECODE_WORKER_COUNT + DECODE_RANK_COUNT; i++) {{
          indices.push(i);
          vals.push(vis);
        }}
        Plotly.restyle("decode-tp", {{visible: vals}}, indices);
      }}
    }}
    </script>"""

    perf_chart_ids = []
    if all_prefill_traces or all_prefill_rank_traces:
        perf_chart_ids.append("prefill-tp")
    if all_decode_traces or all_decode_rank_traces:
        perf_chart_ids.append("decode-tp")

    perf_sync_js = ""
    if len(perf_chart_ids) > 1:
        ids_json = json.dumps(perf_chart_ids)
        perf_sync_js = f"""
    <script>
    (function() {{
      var perfChartIds = {ids_json};
      var _perfSyncTimer = null;
      perfChartIds.forEach(function(srcId) {{
        var el = document.getElementById(srcId);
        if (!el) return;
        el.on("plotly_relayout", function(ed) {{
          if (_perfSyncTimer) clearTimeout(_perfSyncTimer);
          _perfSyncTimer = setTimeout(function() {{
            var update = {{}};
            if (ed["xaxis.range[0]"] !== undefined) {{
              update["xaxis.range[0]"] = ed["xaxis.range[0]"];
              update["xaxis.range[1]"] = ed["xaxis.range[1]"];
            }}
            if (ed["xaxis.autorange"] !== undefined) {{
              update["xaxis.autorange"] = ed["xaxis.autorange"];
            }}
            if (Object.keys(update).length > 0) {{
              perfChartIds.forEach(function(dstId) {{
                if (dstId !== srcId) Plotly.relayout(dstId, update);
              }});
            }}
          }}, 300);
        }});
      }});
    }})();
    </script>"""

    perf_panel = (
        '<div id="performance-panel" style="display:none;padding:16px 24px">\n'
        + toggle_html + "\n"
        + "\n".join(charts_parts)
        + toggle_js
        + perf_sync_js
        + "\n</div>\n\n"
    )

    html = html.replace(
        '<div id="overview-panel"',
        perf_panel + '<div id="overview-panel"',
    )

    # Add show/hide for performance panel in switchTab
    html = html.replace(
        'if (tab === "overview") {',
        'if (tab === "performance") {\n'
        '    document.getElementById("performance-panel").style.display = "block";\n'
        '    overviewPanel.style.display = "none";\n'
        '    chartsArea.style.display = "none";\n'
        '    toolbar.style.display = "none";\n'
        '    selectorBar.classList.remove("visible");\n'
        '    document.getElementById("performance-panel").querySelectorAll("[id^=prefill-tp],[id^=decode-tp]").forEach(function(el) { Plotly.Plots.resize(el); });\n'
        '    return;\n'
        '  }\n'
        '  document.getElementById("performance-panel").style.display = "none";\n\n'
        '  if (tab === "overview") {',
    )

    # Also hide performance panel on resetAll
    html = html.replace(
        '  document.getElementById("overview-panel").style.display = "none";',
        '  document.getElementById("overview-panel").style.display = "none";\n'
        '  document.getElementById("performance-panel").style.display = "none";',
    )

    html_path.write_text(html, encoding="utf-8")
    logger.info("Injected Performance tab with throughput charts")


def _downsample_traces(html_path: Path, max_points_per_trace: int = 500) -> None:
    """Downsample per-node GPU traces using max-of-every-N-bucket to preserve spikes."""
    html = html_path.read_text(encoding="utf-8")

    traces_parsed = _extract_js_var(html, "PANEL_TRACES")
    if not traces_parsed:
        return

    traces_json, traces_start, traces_end = traces_parsed
    all_traces = json.loads(traces_json)

    changed = False
    for panel_key, panel_traces in all_traces.items():
        for trace in panel_traces:
            x = trace.get("x")
            y = trace.get("y")
            if not x or not isinstance(x, list) or not y or not isinstance(y, list):
                continue
            n = len(x)
            if n <= max_points_per_trace:
                continue
            step = max(1, n // max_points_per_trace)
            new_x, new_y = [], []
            for bucket_start in range(0, n, step):
                bucket_end = min(bucket_start + step, n)
                bucket_y = y[bucket_start:bucket_end]
                max_idx = bucket_start + max(range(len(bucket_y)), key=lambda i: bucket_y[i])
                new_x.append(x[max_idx])
                new_y.append(y[max_idx])
            trace["x"] = new_x
            trace["y"] = new_y
            if "customdata" in trace and isinstance(trace["customdata"], list):
                cd = trace["customdata"]
                new_cd = []
                for bucket_start in range(0, n, step):
                    bucket_end = min(bucket_start + step, n)
                    bucket_y = y[bucket_start:bucket_end] if bucket_start < len(y) else [0]
                    max_idx = bucket_start + max(range(len(bucket_y)), key=lambda i: bucket_y[i])
                    if max_idx < len(cd):
                        new_cd.append(cd[max_idx])
                trace["customdata"] = new_cd
            changed = True

    if changed:
        new_line = f"var PANEL_TRACES = {json.dumps(all_traces, default=str)};\n"
        html = html[:traces_start] + new_line + html[traces_end:]
        html_path.write_text(html, encoding="utf-8")
        logger.info("Downsampled traces to max %d points per trace (max-of-bucket)", max_points_per_trace)


def _inject_perf_overlay(html_path: Path, resolved_logs: list[Path]) -> None:
    """Add a 'Show perf trace' checkbox to GPU power tabs that overlays throughput on a 2nd y-axis."""
    from srt_upload_helper import (
        _parse_benchmark_log, _sync_perf_timestamps, _read_perf_csv,
        _host_from_csv_name, _parse_parallelization_tag, _COMPOSITE_HOST_SEP,
        _sglang_parallel_label, _parse_worker_counts,
    )

    overlay_traces: list[dict] = []

    for ld in resolved_logs:
        jid = _slurm_job_id_for_log_dir(ld)
        par_tag = _parse_parallelization_tag(ld)
        jid_label = f"{jid} {par_tag}" if par_tag else jid

        tz_offset = _compute_tz_offset(ld)
        prefill_data, decode_data = _parse_worker_throughput(ld, tz_offset)

        role_agg: dict[str, str] = {}
        for candidate in [ld / "config.yaml", ld.parent / "config.yaml"]:
            if candidate.is_file():
                try:
                    with open(candidate, "r") as _f:
                        _cfg = yaml.safe_load(_f)
                    sglang = (_cfg.get("backend") or {}).get("sglang_config") or {}
                    for rn in ["prefill", "decode"]:
                        pl = _sglang_parallel_label(sglang.get(rn) or {})
                        role_agg[rn] = "sum" if pl.startswith("DEP") else "mean"
                    break
                except Exception:
                    pass

        wc = _parse_worker_counts(ld)

        for role, data, color in [
            ("prefill", prefill_data, "rgba(118,185,0,0.6)"),
            ("decode", decode_data, "rgba(26,115,232,0.6)"),
        ]:
            agg_method = role_agg.get(role, "mean")
            host_ranks: dict[str, list[str]] = {}
            for key in data:
                host_ranks.setdefault(key.split("/")[0], []).append(key)

            sorted_hosts = sorted(host_ranks.keys())
            nw = min(wc.get(role, len(sorted_hosts)), len(sorted_hosts))
            if nw <= 0:
                nw = len(sorted_hosts)
            chunk = max(1, len(sorted_hosts) // nw)
            role_cap = role.capitalize()

            for wi in range(nw):
                si = wi * chunk
                ei = si + chunk if wi < nw - 1 else len(sorted_hosts)
                wh = sorted_hosts[si:ei]
                rank_dfs = [data[rk] for h in wh for rk in sorted(host_ranks.get(h, []))]
                if not rank_dfs:
                    continue
                combined = pd.concat(rank_dfs, ignore_index=True)
                agg_df = (
                    combined.groupby("timestamp", as_index=False)
                    .agg({"throughput": agg_method})
                    .sort_values("timestamp")
                )
                x = [t.isoformat() for t in agg_df["timestamp"]]
                y = agg_df["throughput"].tolist()
                label = f"{jid_label} {role_cap}{wi} TPS"
                overlay_traces.append({
                    "x": x, "y": y,
                    "type": "scatter", "mode": "lines",
                    "name": label,
                    "yaxis": "y2",
                    "line": {"color": color, "width": 2, "dash": "dashdot"},
                    "hovertemplate": (
                        f"<b>{label}</b><br>"
                        "Time (UTC): %{x|%Y-%m-%d %H:%M:%S}<br>"
                        "Throughput: %{y:.1f} tok/s"
                        "<extra></extra>"
                    ),
                    "_perf_role": role,
                })

    if not overlay_traces:
        return

    html = html_path.read_text(encoding="utf-8")

    # Rename per-node checkbox and add perf overlay checkbox after it
    html = html.replace(
        'Show per-node traces',
        'Show per-node GPU traces',
    )
    html = html.replace(
        'Show per-node GPU traces</label>',
        'Show per-node GPU traces</label>\n'
        '  <label style="display:flex;align-items:center;gap:6px;font-size:13px;cursor:pointer;align-self:center">\n'
        '    <input type="checkbox" id="show-perf-overlay-cb" '
        'onchange="togglePerfOverlay()" style="accent-color:#76b900">\n'
        '    Show perf trace</label>',
    )

    # Build role index arrays for filtering by tab
    prefill_overlay_indices = [i for i, t in enumerate(overlay_traces) if t.get("_perf_role") == "prefill"]
    decode_overlay_indices = [i for i, t in enumerate(overlay_traces) if t.get("_perf_role") == "decode"]

    # Remove the internal _perf_role key before serializing
    for t in overlay_traces:
        t.pop("_perf_role", None)

    # Inject overlay data and toggle JS before </body>
    overlay_json = json.dumps(overlay_traces, default=str)

    # Get panel count from HTML
    panel_count_match = re.search(r"var PANEL_COUNT = (\d+);", html)
    n_panels = int(panel_count_match.group(1)) if panel_count_match else 5

    overlay_js = f"""
<script>
var PERF_OVERLAY_TRACES = {overlay_json};
var perfOverlayAdded = false;
var perfOverlayCount = PERF_OVERLAY_TRACES.length;
var PREFILL_OVERLAY_IDX = {json.dumps(prefill_overlay_indices)};
var DECODE_OVERLAY_IDX = {json.dumps(decode_overlay_indices)};

function togglePerfOverlay() {{
  var show = document.getElementById("show-perf-overlay-cb").checked;
  if (show && !perfOverlayAdded) {{
    for (var p = 0; p < {n_panels}; p++) {{
      var el = document.getElementById("panel-" + p);
      if (!el) continue;
      Plotly.addTraces("panel-" + p, PERF_OVERLAY_TRACES);
      Plotly.relayout("panel-" + p, {{
        "yaxis2": {{
          "title": "Throughput (tok/s)",
          "overlaying": "y",
          "side": "right",
          "showgrid": false,
          "titlefont": {{"color": "#888"}},
          "tickfont": {{"color": "#888"}}
        }}
      }});
    }}
    perfOverlayAdded = true;
    _applyPerfOverlayFilter();
  }} else if (show && perfOverlayAdded) {{
    _applyPerfOverlayFilter();
  }} else if (!show && perfOverlayAdded) {{
    for (var p = 0; p < {n_panels}; p++) {{
      var el = document.getElementById("panel-" + p);
      if (!el) continue;
      var n = el.data.length;
      var indices = [];
      var vals = [];
      for (var i = n - perfOverlayCount; i < n; i++) {{
        indices.push(i);
        vals.push(false);
      }}
      Plotly.restyle("panel-" + p, {{visible: vals}}, indices);
    }}
  }}
}}

function _applyPerfOverlayFilter() {{
  if (!perfOverlayAdded || !document.getElementById("show-perf-overlay-cb").checked) return;
  var tab = currentTabMode || "all";
  var showPrefill = (tab === "all" || tab === "role:prefill");
  var showDecode = (tab === "all" || tab === "role:decode");

  for (var p = 0; p < {n_panels}; p++) {{
    var el = document.getElementById("panel-" + p);
    if (!el) continue;
    var n = el.data.length;
    var indices = [];
    var vals = [];
    for (var i = 0; i < perfOverlayCount; i++) {{
      var traceIdx = n - perfOverlayCount + i;
      indices.push(traceIdx);
      if (PREFILL_OVERLAY_IDX.indexOf(i) >= 0) {{
        vals.push(showPrefill ? true : false);
      }} else if (DECODE_OVERLAY_IDX.indexOf(i) >= 0) {{
        vals.push(showDecode ? true : false);
      }} else {{
        vals.push(true);
      }}
    }}
    Plotly.restyle("panel-" + p, {{visible: vals}}, indices);
  }}
}}

// Hook into applyFilters to update perf overlay on tab switch
var _origApplyFilters = applyFilters;
applyFilters = function() {{
  _origApplyFilters();
  if (typeof _applyPerfOverlayFilter === "function") _applyPerfOverlayFilter();
}};
</script>
"""
    html = html.replace("</body>", overlay_js + "</body>")

    html_path.write_text(html, encoding="utf-8")
    logger.info("Injected perf overlay toggle for GPU power charts")


def main() -> None:
    epilog = """examples:
  %(prog)s run1/86283
  %(prog)s run1/86283 run1/86284 -o combined.html
  %(prog)s run1/86283 run1/86284 --title "86283 vs 86284" -o out.html
  %(prog)s run1/86283 --slurm-id 86283 --job-id "nightly A"
"""
    p = argparse.ArgumentParser(
        description=(
            "Write perf_monitor_dashboard.html from perf_samples_*.csv. "
            "Multiple run directories are merged into one figure; legends are prefixed "
            "with each run's Slurm job id (inferred from paths unless --slurm-id is set)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=epilog,
    )
    p.add_argument(
        "run_dirs",
        type=Path,
        nargs="+",
        metavar="RUN_DIR",
        help="One or more run roots (e.g. run1/86283) or a logs/ directory that contains perf CSVs",
    )
    p.add_argument(
        "--slurm-id",
        dest="slurm_ids",
        metavar="ID",
        action="append",
        default=None,
        help=(
            "Slurm job id for legend prefixes, one per RUN_DIR in the same order "
            "(optional; paths are used by default). Example: --slurm-id 111 --slurm-id 222"
        ),
    )
    p.add_argument(
        "--job-id",
        type=str,
        default=None,
        help="Single-run: sets the HTML title. Multi-run: same as --title if --title omitted.",
    )
    p.add_argument(
        "--title",
        type=str,
        default=None,
        help="Explicit HTML document / header title (overrides --job-id when set).",
    )
    p.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        metavar="PATH",
        help="Output .html path (default: beside the first run's perf CSVs)",
    )
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    resolved_logs: list[Path] = []
    for rd in args.run_dirs:
        log_dir = _resolve_log_dir(rd)
        if not list(log_dir.glob("perf_samples_*.csv")):
            logger.error("No perf_samples_*.csv under %s (also checked %s/logs)", rd, rd)
            sys.exit(1)
        resolved_logs.append(log_dir)

    n = len(resolved_logs)
    slurm_ids = _normalize_slurm_ids(n, args.slurm_ids)

    doc_title = _document_title(
        n,
        title=args.title,
        job_id=args.job_id,
        resolved_logs=resolved_logs,
        slurm_ids=slurm_ids,
    )

    default_out = resolved_logs[0] / "perf_monitor_dashboard.html"
    out_path = args.output or default_out

    def _prep_transform(bench: list[dict[str, Any]]) -> list[dict[str, Any]]:
        prep_ivs = _split_prep_from_main(bench, resolved_logs)
        bench.extend(prep_ivs)
        return bench

    out = generate_perf_charts_multi(
        resolved_logs,
        title=doc_title,
        slurm_ids=slurm_ids,
        output_html_path=out_path,
        interval_transform=_prep_transform,
    )

    if not out:
        logger.error("Dashboard not generated (no valid perf CSV data).")
        sys.exit(1)

    for path in out:
        _downsample_traces(path)
        _inject_whole_rack_trace(path)
        _cleanup_summary_bar(path, resolved_logs)
        _inject_performance_tab(path, resolved_logs)
        _inject_perf_overlay(path, resolved_logs)
        logger.info("Wrote %s", path)


if __name__ == "__main__":
    main()
