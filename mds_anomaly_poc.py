#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CephFS MDS anomaly POC (EWMA + z-score + optional IF + alerts)

- EWMA residuals + baseline z-score (σ floor to avoid inf)
- Optional IsolationForest score (used only in severity)
- Warm-up suppression at phase boundaries
- Subevents: short anomalies (row groups)
- Alerts: longer "hot periods" grouping subevents
- Alert lifecycle logging to alerts_output.json (JSON lines)
- Event severity score + label
- Paginated all-signals plots + per-event zooms
- Static HTML dashboard (index.html)

Baselines:
- Either explicit baseline CSVs via --base
- Or the first --base-first-seconds of the trace (if no base files supplied)
"""

import argparse
from pathlib import Path
from dataclasses import dataclass, asdict
import json
import numpy as np
import pandas as pd

# -----------------------------
# Columns & defaults
# -----------------------------
TIMECOL = "timestamp"

COUNTER_COLS = [
    "msgr_recv_bytes", "msgr_send_bytes",
    "msgr_recv_messages", "msgr_send_messages",
    "mds_request", "mds_reply",
    "objecter_send_bytes", "objecter_ops",
]

LATENCY_COLS = [
    "mds_reply_latency_s", "getattr_latency_s", "mkdir_latency_s",
    "mds_journal_latency_s", "objecter_op_latency_s",
]

GAUGE_COLS = ["mds_rss_bytes", "mds_heap_bytes", "sessions_open"]

ALL_EXPECTED = [TIMECOL] + COUNTER_COLS + LATENCY_COLS + GAUGE_COLS

DEFAULT_BASE_FILES = []
DEFAULT_STRESS_FILES = []


# -----------------------------
# Loading & preprocessing
# -----------------------------
def _parse_timestamp_col(series: pd.Series) -> pd.Series:
    """Robustly parse timestamps that may be UNIX (s/ms) or ISO8601 strings."""
    s = series.copy()

    if np.issubdtype(s.dtype, np.datetime64):
        return pd.to_datetime(s, utc=True, errors="coerce")

    # Try numeric epoch
    try:
        s_num = pd.to_numeric(s, errors="coerce")
        if s_num.notna().any():
            mx = s_num.max()
            if mx > 1e14:  # probably µs/ns, treat as invalid here
                raise ValueError
            unit = "ms" if mx > 1e11 else "s"
            return pd.to_datetime(s_num, unit=unit, utc=True, errors="coerce")
    except Exception:
        pass

    # Fallback: ISO8601
    return pd.to_datetime(s, utc=True, errors="coerce")


def load_csv(path: Path, label: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip().strip('"') for c in df.columns]
    if TIMECOL not in df.columns:
        raise ValueError(f"{path} has no '{TIMECOL}' column. Columns={df.columns.tolist()}")
    df[TIMECOL] = _parse_timestamp_col(df[TIMECOL])
    df = df.dropna(subset=[TIMECOL]).sort_values(TIMECOL).reset_index(drop=True)
    df["phase"] = label
    return df


def to_rates(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    dt = df[TIMECOL].diff().dt.total_seconds().astype("float64")
    if len(dt) > 0:
        dt.iloc[0] = np.nan
    for col in COUNTER_COLS:
        if col in df.columns:
            diff = df[col].diff()
            diff = diff.where(diff >= 0, np.nan)  # guard against counter reset
            df[col + "_rate"] = diff / dt
    return df


def ewma(df: pd.DataFrame, cols, span: int) -> pd.DataFrame:
    out = {}
    for c in cols:
        if c in df.columns:
            ewm = df[c].ewm(span=span, adjust=False, min_periods=span).mean()
            out[c + "_ewm"] = ewm
            out[c + "_resid"] = df[c] - ewm
    return pd.DataFrame(out)


def prepare_features(df: pd.DataFrame, ewma_span: int) -> pd.DataFrame:
    df = to_rates(df)
    rate_cols = [c + "_rate" for c in COUNTER_COLS if c in df.columns]
    keep_cols = rate_cols + [c for c in LATENCY_COLS + GAUGE_COLS if c in df.columns]
    ewm_resid = ewma(df, keep_cols, span=ewma_span)
    feat = pd.concat([df[[TIMECOL, "phase"] + keep_cols], ewm_resid], axis=1)
    return feat


# -----------------------------
# Scoring
# -----------------------------
def zscore_against_baseline(all_df: pd.DataFrame, baseline_mask: pd.Series, use_residuals=True):
    if use_residuals:
        metric_cols = [c for c in all_df.columns if c.endswith("_resid")]
    else:
        metric_cols = [c for c in all_df.columns if c not in (TIMECOL, "phase") and not c.endswith("_ewm")]

    base = all_df.loc[baseline_mask, metric_cols]
    mu = base.mean(skipna=True)
    sigma = base.std(skipna=True)
    sigma = sigma.mask((sigma < 1e-12) | (sigma.isna()))

    z = (all_df[metric_cols] - mu) / sigma
    z = z.replace([np.inf, -np.inf], np.nan)
    return metric_cols, z, mu, sigma


def anomaly_summary(phased_df: pd.DataFrame, metric_cols, z: pd.DataFrame, thresh: float):
    anom = z.abs() > thresh
    phased_df = phased_df.copy()
    phased_df["any_anomaly"] = anom.any(axis=1)
    per_metric = anom.sum(axis=0).sort_values(ascending=False).rename("anomaly_points")
    per_phase = phased_df.groupby("phase")["any_anomaly"].sum().rename("phase_anomaly_points")
    return per_metric, per_phase, anom


def try_isolation_forest(train_df: pd.DataFrame, test_df: pd.DataFrame, metric_cols):
    try:
        from sklearn.ensemble import IsolationForest
    except Exception as e:
        print(f"[IForest] Skipping (sklearn not available: {e})")
        return None

    try:
        mu = train_df[metric_cols].mean()
        X_train = train_df[metric_cols].fillna(mu)
        X_test = test_df[metric_cols].fillna(mu)
        if len(X_train) == 0 or len(X_test) == 0 or len(metric_cols) == 0:
            print("[IForest] Skipping (no data/features).")
            return None
        model = IsolationForest(
            n_estimators=200,
            contamination="auto",
            random_state=42,
            n_jobs=-1,
        )
        model.fit(X_train)
        score = -model.decision_function(X_test)  # higher = more anomalous
        return score
    except Exception as e:
        print(f"[IForest] Skipping IsolationForest (reason: {e})")
        return None


# -----------------------------
# Subevents (row-anomaly groups)
# -----------------------------
@dataclass
class AnomalyEvent:
    event_id: int
    start_ts: pd.Timestamp
    end_ts: pd.Timestamp
    duration_s: float
    phase_mix: dict
    n_points: int
    metrics_top: list      # [(metric, max_abs_z, direction), ...]
    max_abs_z: float
    metrics_count: int
    reasoning: str
    severity: float
    label: str


def _metric_kind(name: str) -> str:
    base = name.replace("_resid", "")
    if base.endswith("_latency_s"):
        return "latency"
    if base.endswith("_rate"):
        return "rate"
    if base in {"mds_rss_bytes", "mds_heap_bytes", "sessions_open"}:
        return "resource"
    return "other"


def _direction(zval: float) -> str:
    if pd.isna(zval):
        return "unknown"
    return "up" if zval > 0 else "down"


def build_anomaly_events(
    feat: pd.DataFrame,
    z: pd.DataFrame,
    metric_cols,
    zth: float = 3.0,
    min_metrics: int = 2,
    max_gap_sec: float = 30.0,
    suppress_mask: pd.Series | None = None,
    min_duration_sec: float = 0.0,
):
    """
    Build "subevents" from row-level anomalies.
    Row-level anomaly condition (z-only):
      z_anom_count >= min_metrics  where z_anom_count = #metrics with |z| > zth
    Rows are grouped into subevents if gap <= max_gap_sec.
    """
    ts = feat[TIMECOL].reset_index(drop=True)
    phases = feat["phase"].reset_index(drop=True)

    # Count how many metrics exceed threshold per row
    z_anom_count = (z[metric_cols].abs() > zth).sum(axis=1)
    feat["z_anom_count"] = z_anom_count

    # Optional suppression (warmup)
    anom_row_mask = z_anom_count >= min_metrics
    if suppress_mask is not None:
        anom_row_mask = anom_row_mask & (~suppress_mask)

    anom_idx = np.flatnonzero(anom_row_mask.to_numpy())
    events = []
    if len(anom_idx) == 0:
        return events

    def flush(group, evt_id):
        idxs = np.array(group)
        start_ts = ts.iloc[idxs].min()
        end_ts = ts.iloc[idxs].max()
        duration = (end_ts - start_ts).total_seconds()
        if duration < 0:
            start_ts, end_ts = end_ts, start_ts
            duration = abs(duration)
        if duration < min_duration_sec:
            return

        n_points = len(idxs)
        zwin = z.iloc[idxs][metric_cols]
        max_abs = zwin.abs().max().sort_values(ascending=False)

        top = []
        for m in max_abs.index.tolist():
            mvals = zwin[m].values
            if not np.isfinite(mvals).any():
                continue
            j = int(np.nanargmax(np.abs(mvals)))
            zv = float(mvals[j])
            if not np.isfinite(zv):
                continue
            top.append((m, abs(zv), _direction(zv)))
            if len(top) >= 8:
                break

        ph = phases.iloc[idxs].value_counts().to_dict()
        kinds = [_metric_kind(m) for m, _z, _d in top]
        kcnt = {k: kinds.count(k) for k in set(kinds)}
        reason_bits = []
        if kcnt.get("latency", 0) >= 2:
            reason_bits.append("widespread latency elevation vs baseline")
        if kcnt.get("rate", 0) >= 2:
            reason_bits.append("rate/throughput shift vs baseline")
        if kcnt.get("resource", 0) >= 1:
            reason_bits.append("resource pressure (memory/sessions)")
        if not reason_bits:
            reason_bits.append("multi-metric deviation vs baseline")
        reasoning = "; ".join(reason_bits)

        breadth = int((zwin.abs() > zth).sum().max())
        breadth_norm = breadth / max(1, len(metric_cols))
        mag = float(max_abs.iloc[0]) if len(max_abs) else 0.0
        mag_norm = min(mag / 10.0, 1.0)
        dur_norm = min(duration / 60.0, 1.0)

        # Optional IF contribution (if present)
        if "iforest_score" in feat.columns:
            ifo_vals = feat.loc[idxs, "iforest_score"]
            if ifo_vals.max() > ifo_vals.min():
                iforest_norm = (ifo_vals.mean() - ifo_vals.min()) / (ifo_vals.max() - ifo_vals.min())
            else:
                iforest_norm = 0.0
        else:
            iforest_norm = 0.0

        severity = (
            0.35 * mag_norm +
            0.30 * breadth_norm +
            0.20 * dur_norm +
            0.15 * iforest_norm
        )
        label = "major" if severity >= 0.60 else ("moderate" if severity >= 0.35 else "minor")

        ev = AnomalyEvent(
            event_id=evt_id,
            start_ts=start_ts,
            end_ts=end_ts,
            duration_s=duration,
            phase_mix=ph,
            n_points=n_points,
            metrics_top=top,
            max_abs_z=float(mag),
            metrics_count=breadth,
            reasoning=reasoning,
            severity=float(severity),
            label=label,
        )
        events.append(ev)

    evt_id = 1
    group = [anom_idx[0]]
    for i in range(1, len(anom_idx)):
        prev_i = anom_idx[i - 1]
        cur_i = anom_idx[i]
        gap = (ts.iloc[cur_i] - ts.iloc[prev_i]).total_seconds()
        if gap <= max_gap_sec:
            group.append(cur_i)
        else:
            flush(group, evt_id)
            evt_id += 1
            group = [cur_i]
    flush(group, evt_id)
    return events


def save_anomaly_events(events, outdir: Path):
    if not events:
        pd.DataFrame().to_csv(outdir / "anomaly_events.csv", index=False)
        (outdir / "anomaly_events.json").write_text("[]")
        return
    rows = []
    for e in events:
        row = asdict(e)
        row["start_ts"] = e.start_ts.isoformat()
        row["end_ts"] = e.end_ts.isoformat()
        row["metrics_top"] = ";".join([f"{m}:{z:.1f}:{d}" for (m, z, d) in e.metrics_top])
        rows.append(row)
    pd.DataFrame(rows).to_csv(outdir / "anomaly_events.csv", index=False)
    with open(outdir / "anomaly_events.json", "w") as f:
        json.dump([asdict(e) for e in events], f, indent=2, default=str)


# -----------------------------
# Alerts (groups of subevents)
# -----------------------------
@dataclass
class Alert:
    alert_id: int
    start_ts: pd.Timestamp
    end_ts: pd.Timestamp
    duration_s: float
    n_subevents: int
    n_points: int
    phase_mix: dict
    max_abs_z: float
    max_severity: float
    label: str          # major / moderate / minor
    metrics_top: list   # [(metric, max_abs_z, direction), ...]
    subevent_ids: list  # list of event_id inside this alert


def _alert_aggregate_from_group(group, alert_id: int):
    """Aggregate a list of AnomalyEvent into Alert-level stats (without subevents list)."""
    start_ts = min(e.start_ts for e in group)
    end_ts = max(e.end_ts for e in group)
    duration = (end_ts - start_ts).total_seconds()
    n_subevents = len(group)
    n_points = sum(e.n_points for e in group)

    phase_mix = {}
    for e in group:
        for k, v in e.phase_mix.items():
            phase_mix[k] = phase_mix.get(k, 0) + v

    max_abs_z = max(e.max_abs_z for e in group)
    max_severity = max(e.severity for e in group)
    labels = {e.label for e in group}
    if "major" in labels:
        label = "major"
    elif "moderate" in labels:
        label = "moderate"
    else:
        label = "minor"

    metric_best = {}
    for e in group:
        for m, z, d in e.metrics_top:
            if m not in metric_best or z > metric_best[m][0]:
                metric_best[m] = (z, d)
    metrics_top = sorted(
        [(m, z, d) for m, (z, d) in metric_best.items()],
        key=lambda t: t[1],
        reverse=True
    )[:10]

    sub_ids = [e.event_id for e in group]

    ep = Alert(
        alert_id=alert_id,
        start_ts=start_ts,
        end_ts=end_ts,
        duration_s=duration,
        n_subevents=n_subevents,
        n_points=n_points,
        phase_mix=phase_mix,
        max_abs_z=max_abs_z,
        max_severity=max_severity,
        label=label,
        metrics_top=metrics_top,
        subevent_ids=sub_ids,
    )
    return ep


def _serialize_subevent(e: AnomalyEvent) -> dict:
    """Turn a subevent into a dict suitable for embedding inside the closed alert."""
    return {
        "event_id": e.event_id,
        "start_ts": e.start_ts,
        "end_ts": e.end_ts,
        "duration_s": e.duration_s,
        "phase_mix": e.phase_mix,
        "n_points": e.n_points,
        "max_abs_z": e.max_abs_z,
        "metrics_count": e.metrics_count,
        "severity": e.severity,
        "label": e.label,
        "reasoning": e.reasoning,
        "metrics_top": e.metrics_top,
    }


def build_alerts(subevents, alert_gap_sec: float, log_path: Path | None = None):
    """
    Group subevents (AnomalyEvent) into higher-level alerts.

    Logging semantics (JSON lines to log_path, if given):

      - On first detection of an alert:
        status: "new"
        -> single line with alert summary based on current subevents (usually 1),
           including metrics_top etc, but WITHOUT embedded subevents list.

      - When alert is closed:
        status: "closed"
        -> single line with full alert summary AND
           `subevents`: [ all subevents in that alert, serialized ].

    Example JSON lines:

      {"status":"new",    "alert_id":6, ...}
      {"status":"closed", "alert_id":6, "subevents":[{...},{...}], ...}
    """
    if not subevents:
        if log_path is not None:
            # ensure file exists (even empty)
            log_path.write_text("", encoding="utf-8")
        return []

    subevents = sorted(subevents, key=lambda e: e.start_ts)
    alerts = []
    alert_id = 1

    log_f = None

    def log_record(rec: dict):
        nonlocal log_f
        if log_path is None:
            return
        if log_f is None:
            log_f = log_path.open("w", encoding="utf-8")
        json.dump(rec, log_f, default=str)
        log_f.write("\n")
        log_f.flush()

    def log_alert_status(status: str, group, eid: int):
        """Log 'new' or 'closed' record for the given alert group."""
        ep = _alert_aggregate_from_group(group, eid)
        rec = {
            "status": status,              # "new" or "closed"
            "alert_id": ep.alert_id,
            "start_ts": ep.start_ts,
            "end_ts": ep.end_ts,
            "duration_s": ep.duration_s,
            "n_subevents": ep.n_subevents,
            "n_points": ep.n_points,
            "phase_mix": ep.phase_mix,
            "max_abs_z": ep.max_abs_z,
            "max_severity": ep.max_severity,
            "label": ep.label,
            "metrics_top": ep.metrics_top,
        }
        if status == "closed":
            # include all subevents only on close
            rec["subevents"] = [_serialize_subevent(e) for e in group]
        log_record(rec)

    def start_alert(first: AnomalyEvent, eid: int):
        group = [first]
        # "new" alert appears as soon as first subevent is detected
        log_alert_status("new", group, eid)
        return group

    def close_alert(group, eid: int):
        # "closed" alert with full set of subevents
        log_alert_status("closed", group, eid)
        return _alert_aggregate_from_group(group, eid)

    # First alert
    cur_group = start_alert(subevents[0], alert_id)

    for sub in subevents[1:]:
        prev = cur_group[-1]
        gap = (sub.start_ts - prev.end_ts).total_seconds()
        if gap <= alert_gap_sec:
            # same alert, just extend group
            cur_group.append(sub)
            # we intentionally DO NOT log "ongoing" / "subevent_added" here
            # (you only wanted "new" + "closed").
        else:
            # close previous alert & log "closed"
            alerts.append(close_alert(cur_group, alert_id))
            alert_id += 1
            # start a new one & log "new"
            cur_group = start_alert(sub, alert_id)

    # Close last alert
    alerts.append(close_alert(cur_group, alert_id))

    if log_f is not None:
        log_f.close()

    return alerts

def save_alerts(alerts, outdir: Path):
    if not alerts:
        pd.DataFrame().to_csv(outdir / "alerts.csv", index=False)
        (outdir / "alerts.json").write_text("[]")
        return
    rows = []
    for ep in alerts:
        row = asdict(ep)
        row["start_ts"] = ep.start_ts.isoformat()
        row["end_ts"] = ep.end_ts.isoformat()
        row["metrics_top"] = ";".join([f"{m}:{z:.1f}:{d}" for (m, z, d) in ep.metrics_top])
        row["subevent_ids"] = ",".join(str(i) for i in ep.subevent_ids)
        rows.append(row)
    pd.DataFrame(rows).to_csv(outdir / "alerts.csv", index=False)
    with open(outdir / "alerts.json", "w") as f:
        json.dump([asdict(ep) for ep in alerts], f, indent=2, default=str)


# -----------------------------
# Plotting
# -----------------------------
def plot_metric_series(df: pd.DataFrame, metric: str, outdir: Path, title_suffix: str = ""):
    import matplotlib.pyplot as plt
    fig = plt.figure()
    for phase, grp in df.groupby("phase"):
        plt.plot(grp[TIMECOL], grp[metric], label=phase)
    plt.title(metric + title_suffix)
    plt.xlabel("time")
    plt.legend()
    fig.autofmt_xdate()
    fig.savefig(outdir / f"{metric}.png", dpi=140, bbox_inches="tight")
    plt.close(fig)


def plot_all_signals_paginated(df: pd.DataFrame, metric_cols, z, zth, events, outdir: Path, page_size: int = 10):
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    ts = df[TIMECOL]
    mids, labels = [], []
    for e in events:
        mid = e.start_ts + (e.end_ts - e.start_ts) / 2
        mids.append(mid)
        labels.append(f"E{e.event_id}")
    for page_start in range(0, len(metric_cols), page_size):
        page_metrics = metric_cols[page_start:page_start + page_size]
        n = len(page_metrics)
        if n == 0:
            break
        fig, axes = plt.subplots(n, 1, figsize=(16, 2.6 * n), sharex=True)
        if n == 1:
            axes = [axes]
        for ax, m in zip(axes, page_metrics):
            ax.plot(ts, df[m], lw=0.9)
            anom = z[m].abs() > zth
            ax.scatter(ts[anom], df.loc[anom, m], s=10)
            for mid, lbl in zip(mids, labels):
                ax.axvline(mid, lw=0.8, linestyle="--")
                ax.text(mid, ax.get_ylim()[1], lbl, fontsize=8, va="bottom", ha="center", rotation=90)
            ax.set_ylabel(m.replace("_resid", ""), rotation=0, ha="right", va="center")
            ax.grid(True, ls=":")
        axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
        plt.suptitle(f"All residuals (page {page_start // page_size + 1}) | anomalies |z|>{zth}", y=0.995)
        plt.tight_layout(rect=[0, 0, 1, 0.98])
        fname = f"all_signals_anomalies_p{page_start // page_size + 1}.png"
        fig.savefig(outdir / fname, dpi=170)
        plt.close(fig)


def plot_event_contexts(df: pd.DataFrame, events, top_k_metrics: int, outdir: Path, pad_seconds=60):
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    if not events:
        return
    for e in events:
        metrics = [m for (m, _z, _d) in e.metrics_top[:top_k_metrics]]
        if not metrics:
            continue
        t0 = e.start_ts - pd.Timedelta(seconds=pad_seconds)
        t1 = e.end_ts + pd.Timedelta(seconds=pad_seconds)
        win = (df[TIMECOL] >= t0) & (df[TIMECOL] <= t1)
        n = len(metrics)
        fig, axes = plt.subplots(n, 1, figsize=(14, 2.6 * n), sharex=True)
        if n == 1:
            axes = [axes]
        for ax, m in zip(axes, metrics):
            ax.plot(df.loc[win, TIMECOL], df.loc[win, m], lw=1.0)
            ax.axvspan(e.start_ts, e.end_ts, alpha=0.15)
            ax.set_ylabel(m.replace("_resid", ""), rotation=0, ha="right", va="center")
            ax.grid(True, ls=":")
        axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
        plt.suptitle(
            f"Event E{e.event_id}  [{e.start_ts.strftime('%H:%M:%S')}–{e.end_ts.strftime('%H:%M:%S')}]  |  {e.reasoning}",
            y=0.995,
        )
        plt.tight_layout(rect=[0, 0, 1, 0.98])
        fig.savefig(outdir / f"event_E{e.event_id}_context.png", dpi=170)
        plt.close(fig)


# -----------------------------
# HTML dashboard
# -----------------------------
def write_html_index(outdir: Path, params: dict, events, alerts, all_signal_pages, event_imgs):
    css = """
    <style>
      body { font-family: system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif; margin: 20px; }
      h1, h2, h3 { margin-top: 1.2em; }
      code { background: #f5f5f5; padding: 2px 4px; border-radius: 4px; }
      .grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(420px, 1fr)); gap: 16px; }
      .card { border: 1px solid #e5e7eb; border-radius: 10px; padding: 10px; box-shadow: 0 1px 2px rgba(0,0,0,0.04); }
      .muted { color: #6b7280; font-size: 0.95em; }
      table { border-collapse: collapse; width: 100%; }
      th, td { border: 1px solid #e5e7eb; padding: 6px 8px; text-align: left; }
      th { background: #f9fafb; }
      img { width: 100%; height: auto; border-radius: 8px; }
      .mono { font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace; }
    </style>
    """
    params_pre = json.dumps(params, indent=2)

    # Alerts table
    if alerts:
        ep_rows = ""
        for ep in alerts:
            mix = ", ".join([f"{k}:{v}" for k, v in ep.phase_mix.items()])
            topm = ", ".join([f"{m}({d},{z:.1f})" for (m, z, d) in ep.metrics_top[:6]])
            sub_ids = ", ".join(f"E{i}" for i in ep.subevent_ids)
            ep_rows += (
                f"<tr>"
                f"<td>EP{ep.alert_id}</td>"
                f"<td>{ep.label}</td>"
                f"<td>{ep.start_ts}</td>"
                f"<td>{ep.end_ts}</td>"
                f"<td>{ep.duration_s:.1f}s</td>"
                f"<td>{ep.n_subevents}</td>"
                f"<td>{ep.n_points}</td>"
                f"<td>{ep.max_abs_z:.1f}</td>"
                f"<td>{ep.max_severity:.2f}</td>"
                f"<td>{mix}</td>"
                f"<td class='mono'>{topm}</td>"
                f"<td class='mono'>{sub_ids}</td>"
                f"</tr>"
            )
        alerts_table = f"""
        <table>
          <thead>
            <tr>
              <th>Alert</th><th>Label</th><th>Start</th><th>End</th><th>Duration</th>
              <th>#Subevents</th><th>#Points</th><th>max|z|</th><th>max severity</th>
              <th>Phase mix</th><th>Top metrics</th><th>Subevents</th>
            </tr>
          </thead>
          <tbody>{ep_rows}</tbody>
        </table>"""
    else:
        alerts_table = "<p class='muted'>No alerts detected.</p>"

    # Subevents table
    if events:
        ev_rows = ""
        for e in events:
            mix = ", ".join([f"{k}:{v}" for k, v in e.phase_mix.items()])
            topm = ", ".join([f"{m}({d},{z:.1f})" for (m, z, d) in e.metrics_top[:6]])
            ev_rows += (
                f"<tr>"
                f"<td>E{e.event_id}</td>"
                f"<td>{e.start_ts}</td>"
                f"<td>{e.end_ts}</td>"
                f"<td>{e.duration_s:.1f}s</td>"
                f"<td>{mix}</td>"
                f"<td>{e.metrics_count}</td>"
                f"<td>{e.max_abs_z:.1f}</td>"
                f"<td>{e.severity:.2f}</td>"
                f"<td>{e.label}</td>"
                f"<td>{e.reasoning}</td>"
                f"<td class='mono'>{topm}</td>"
                f"</tr>"
            )
        events_table = f"""
        <table>
          <thead>
            <tr>
              <th>Event</th><th>Start</th><th>End</th><th>Duration</th>
              <th>Phase mix</th><th>#metrics≥z</th><th>max|z|</th>
              <th>Severity</th><th>Label</th><th>Reasoning</th><th>Top metrics</th>
            </tr>
          </thead>
          <tbody>{ev_rows}</tbody>
        </table>"""
    else:
        events_table = "<p class='muted'>No subevents detected.</p>"

    sig_cards = "\n".join(
        [f"<div class='card'><img src='{Path(p).name}' alt='all-signals'><div class='muted'>{Path(p).name}</div></div>"
         for p in all_signal_pages]
    ) or "<p class='muted'>No all-signals plots found.</p>"

    evt_cards = "\n".join(
        [f"<div class='card'><img src='{Path(p).name}' alt='event context'><div class='muted'>{Path(p).name}</div></div>"
         for p in event_imgs]
    ) or "<p class='muted'>No per-event plots found.</p>"

    links = """
      <ul>
        <li><a href="anomaly_summary_per_metric.csv">anomaly_summary_per_metric.csv</a></li>
        <li><a href="anomaly_summary_per_phase.csv">anomaly_summary_per_phase.csv</a></li>
        <li><a href="features_with_z.csv">features_with_z.csv</a></li>
        <li><a href="anomaly_events.csv">anomaly_events.csv</a></li>
        <li><a href="anomaly_events.json">anomaly_events.json</a></li>
        <li><a href="alerts.csv">alerts.csv</a></li>
        <li><a href="alerts.json">alerts.json</a></li>
        <li><a href="INDEX.json">INDEX.json</a></li>
        <li><a href="alerts_output.json">alerts_output.json</a></li>
      </ul>
    """

    html = f"""<!doctype html>
<html lang="en">
<meta charset="utf-8">
<title>MDS Anomaly Report</title>
{css}
<body>
  <h1>MDS Anomaly Report</h1>
  <p class="muted">Generated by mds_anomaly_poc.py</p>

  <h2>Run parameters</h2>
  <pre class="mono">{params_pre}</pre>

  <h2>Artifacts</h2>
  {links}

  <h2>Alerts (cluster hot periods)</h2>
  {alerts_table}

  <h2>Subevents (spikes inside alerts)</h2>
  {events_table}

  <h2>All-signals (paginated)</h2>
  <div class="grid">{sig_cards}</div>

  <h2>Per-event context (zoomed)</h2>
  <div class="grid">{evt_cards}</div>

  <p class="muted">Tip: open images in a new tab for full-size view.</p>
</body>
</html>
"""
    (outdir / "index.html").write_text(html, encoding="utf-8")


# -----------------------------
# CLI
# -----------------------------
def main():
    ap = argparse.ArgumentParser(
        description="CephFS MDS anomaly POC (EWMA + z / optional IF + alerts + HTML)"
    )
    ap.add_argument("--base", nargs="*", default=DEFAULT_BASE_FILES, help="Baseline CSVs")
    ap.add_argument("--stress", nargs="*", default=DEFAULT_STRESS_FILES, help="Stress/recovery CSVs")
    ap.add_argument(
        "--base-first-seconds",
        type=float,
        default=0.0,
        help=(
            "If >0 AND no base files are provided, tag the first X seconds "
            "(from earliest timestamp across all CSVs) as 'base' and the rest as 'stress'. "
            "Ignored if any base file is provided."
        ),
    )
    ap.add_argument("--ewma-span", type=int, default=10, help="EWMA span (samples)")
    ap.add_argument("--zth", type=float, default=3.0, help="Z-score threshold")
    ap.add_argument("--topn", type=int, default=6, help="How many metrics to plot individually")
    ap.add_argument("--page-size", type=int, default=10, help="All-signals rows per page")
    ap.add_argument("--min-metrics", type=int, default=2, help="Min metrics over threshold (z-based row rule)")
    ap.add_argument("--max-gap-sec", type=float, default=30.0,
                    help="Merge anomaly rows into subevents within this gap (seconds)")
    ap.add_argument("--alert-gap-sec", type=float, default=300.0,
                    help="Merge subevents into alerts if gap between them is <= this many seconds")
    ap.add_argument("--iforest-weight", type=float, default=0.0,
                    help="(Kept for compatibility; IF is only used in severity when available)")
    ap.add_argument("--hybrid-thresh", type=float, default=0.5,
                    help="(Unused now; kept for CLI compatibility)")
    ap.add_argument("--warmup-sec-per-phase", type=float, default=10.0,
                    help="Ignore anomalies for N seconds after phase change")
    ap.add_argument("--min-duration-sec", type=float, default=8.0,
                    help="Drop subevents shorter than this duration (seconds)")
    ap.add_argument("--min-severity", type=float, default=0.35,
                    help="Drop subevents below this severity (0..1)")
    ap.add_argument("--outdir", type=str, default="out_anomaly", help="Output directory")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Load CSVs
    frames = []

    def _add_files(file_list, label):
        added = 0
        for fname in file_list:
            if not fname:
                continue
            p = Path(fname)
            if p.is_dir():
                print(f"WARNING: '{p}' is a directory; skipping.")
                continue
            if not p.is_file():
                print(f"WARNING: file not found: {p}")
                continue
            df = load_csv(p, label)
            frames.append(df)
            added += 1
        return added

    n_base = _add_files(args.base, "base")
    n_stress = _add_files(args.stress, "stress")

    if not frames:
        raise SystemExit("No CSVs found. Provide --base/--stress files or a single prod CSV via --stress.")

    print("[Files]")
    for i, df in enumerate(frames):
        phase0 = df["phase"].iloc[0] if len(df) else "?"
        print(
            f"  #{i:02d} phase={phase0:5s} rows={len(df):6d} "
            f"min={df[TIMECOL].min()} max={df[TIMECOL].max()}"
        )

    raw = pd.concat(frames, ignore_index=True)
    cols_present = [c for c in ALL_EXPECTED if c in raw.columns]
    raw = raw[cols_present + ["phase"]].sort_values(TIMECOL).reset_index(drop=True)

    # Phase override if no base files but base-first-seconds specified
    if n_base == 0 and args.base_first_seconds > 0:
        t0 = raw[TIMECOL].min()
        cutoff = t0 + pd.Timedelta(seconds=args.base_first_seconds)
        raw["phase"] = np.where(raw[TIMECOL] <= cutoff, "base", "stress")
        print(
            f"[PhaseOverride] No base files provided; assigned 'base' to first "
            f"{args.base_first_seconds} seconds ({t0.isoformat()} .. {cutoff.isoformat()}), "
            "rest as 'stress'."
        )
    else:
        if n_base > 0 and args.base_first_seconds > 0:
            print("[PhaseOverride] Ignored --base-first-seconds because base files were provided.")

    # Prepare features
    feat = prepare_features(raw, ewma_span=args.ewma_span)
    ord_idx = feat[TIMECOL].argsort().values
    feat = feat.iloc[ord_idx].reset_index(drop=True)

    baseline_mask = feat["phase"] == "base"
    n_base_rows = int(baseline_mask.sum())
    print(f"[Baseline] rows={n_base_rows}")

    # Z-scores vs baseline
    metric_cols, z, mu, sigma = zscore_against_baseline(feat, baseline_mask, use_residuals=True)
    if len(z) != len(feat):
        z = z.reindex_like(feat[metric_cols])

    # Warm-up suppression
    warmup_mask = pd.Series(False, index=feat.index)
    if args.warmup_sec_per_phase > 0:
        flips = feat.index[feat["phase"] != feat["phase"].shift(1)]
        for i in flips:
            t0 = feat.loc[i, TIMECOL]
            warm = (feat[TIMECOL] >= t0) & (
                feat[TIMECOL] <= t0 + pd.Timedelta(seconds=args.warmup_sec_per_phase)
            )
            warmup_mask |= warm

    # Summaries
    per_metric, per_phase, _ = anomaly_summary(
        feat[[TIMECOL, "phase"]], metric_cols, z, args.zth
    )
    pd.DataFrame({"metric": per_metric.index, "anomaly_points": per_metric.values}).to_csv(
        outdir / "anomaly_summary_per_metric.csv", index=False
    )
    per_phase.to_csv(outdir / "anomaly_summary_per_phase.csv", header=True)

    # Optional IF (for severity only)
    iforest_score = None
    if n_base_rows > 0:
        try:
            train_df = feat.loc[baseline_mask, metric_cols]
            test_df = feat.loc[:, metric_cols]
            iforest_score = try_isolation_forest(train_df, test_df, metric_cols)
            if iforest_score is not None:
                feat["iforest_score"] = np.nan
                feat.loc[:, "iforest_score"] = iforest_score
        except Exception as e:
            print(f"[IForest] Error: {e}")

    # Build subevents
    events = build_anomaly_events(
        feat,
        z,
        metric_cols,
        zth=args.zth,
        min_metrics=args.min_metrics,
        max_gap_sec=args.max_gap_sec,
        suppress_mask=warmup_mask,
        min_duration_sec=args.min_duration_sec,
    )

    # Filter subevents by severity
    events = [e for e in events if e.severity >= args.min_severity]

    # Persist features + z + auxiliaries
    feat_out = feat.copy()
    z_out = z.copy()
    z_out.columns = [f"z_{c}" for c in z_out.columns]
    joined = pd.concat([feat_out, z_out], axis=1)
    joined["z_max_abs"] = z.abs().max(axis=1)
    joined.to_csv(outdir / "features_with_z.csv", index=False)

    # Save subevents
    save_anomaly_events(events, outdir)

    # Build alerts with logging
    alert_log_path = outdir / "alerts_output.json"
    alerts = build_alerts(events, alert_gap_sec=args.alert_gap_sec, log_path=alert_log_path)
    save_alerts(alerts, outdir)

    # Visuals
    plot_all_signals_paginated(feat, metric_cols, z, args.zth, events, outdir, page_size=args.page_size)

    top_metrics = per_metric.sort_values(ascending=False).head(args.topn).index.tolist()
    for m_resid in top_metrics:
        base_col = m_resid.replace("_resid", "")
        to_plot = feat[[TIMECOL, "phase"]].copy()
        if base_col in feat.columns:
            to_plot[base_col] = feat[base_col]
        if base_col + "_ewm" in feat.columns:
            to_plot[base_col + "_ewm"] = feat[base_col + "_ewm"]
        if base_col in to_plot.columns:
            plot_metric_series(to_plot, base_col, outdir, title_suffix=" (raw)")
        if base_col + "_ewm" in to_plot.columns:
            plot_metric_series(to_plot, base_col + "_ewm", outdir, title_suffix=" (ewma)")

    plot_event_contexts(feat, events, top_k_metrics=4, outdir=outdir, pad_seconds=60)

    # HTML
    all_signal_pages = sorted([str(p) for p in outdir.glob("all_signals_anomalies_p*.png")])
    event_imgs = sorted([str(p) for p in outdir.glob("event_E*_context.png")])

    index_info = {
        "outputs": [
            "anomaly_summary_per_metric.csv",
            "anomaly_summary_per_phase.csv",
            "features_with_z.csv",
            "anomaly_events.csv",
            "anomaly_events.json",
            "alerts.csv",
            "alerts.json",
            "alerts_output.json",
            "all_signals_anomalies_p*.png",
            "event_E*_context.png",
            "index.html",
        ],
        "params": {
            "ewma_span": args.ewma_span,
            "z_threshold": args.zth,
            "min_metrics_for_subevent": args.min_metrics,
            "max_gap_sec_for_subevent": args.max_gap_sec,
            "alert_gap_sec": args.alert_gap_sec,
            "page_size": args.page_size,
            "baseline_files": args.base,
            "stress_files": args.stress,
            "base_first_seconds": args.base_first_seconds,
            "warmup_sec_per_phase": args.warmup_sec_per_phase,
            "min_duration_sec": args.min_duration_sec,
            "min_severity": args.min_severity,
        },
        "subevents_detected": len(events),
        "alerts_detected": len(alerts),
    }
    (outdir / "INDEX.json").write_text(json.dumps(index_info, indent=2))
    write_html_index(outdir, index_info["params"], events, alerts, all_signal_pages, event_imgs)

    print(json.dumps(index_info, indent=2))
    print(f"Done. Open: {outdir.resolve()}/index.html")


if __name__ == "__main__":
    main()
