import os
import re
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional

import altair as alt
import numpy as np
import pandas as pd
import requests
import streamlit as st
from st_aggrid import AgGrid, GridOptionsBuilder
import hashlib

# Basic page config
st.set_page_config(page_title="NeMo RL Nightly HUD", page_icon="🧭", layout="wide")
st.title("🧭 NeMo RL Nightly HUD")
st.caption(
    "Nightly CI HUD for commits in NVIDIA-NeMo/RL."
)


# Constants
OWNER = "NVIDIA-NeMo"
REPO = "RL"
DEFAULT_MAX_COMMITS = 400
DATA_DIR = Path(__file__).parent / "data" / "nemo_rl"


def _github_headers() -> Dict[str, str]:
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        return {"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json"}
    return {"Accept": "application/vnd.github+json"}


@st.cache_data(ttl=1800)
def fetch_commits(max_commits: int = DEFAULT_MAX_COMMITS) -> pd.DataFrame:
    """Fetch recent commits on main branch from GitHub and return a DataFrame.

    Columns: sha, short_sha, commit_title, commit_date (pd.Timestamp), html_url
    """
    commits: List[Dict] = []
    per_page = 100
    page = 1
    headers = _github_headers()
    while len(commits) < max_commits:
        resp = requests.get(
            f"https://api.github.com/repos/{OWNER}/{REPO}/commits",
            params={"sha": "main", "per_page": per_page, "page": page},
            headers=headers,
            timeout=30,
        )
        resp.raise_for_status()
        page_items = resp.json()
        if not page_items:
            break
        commits.extend(page_items)
        if len(page_items) < per_page:
            break
        page += 1
    commits = commits[:max_commits]
    rows = []
    for c in commits:
        sha = c.get("sha")
        commit_obj = c.get("commit", {})
        message = commit_obj.get("message", "").split("\n")[0]
        date_str = commit_obj.get("author", {}).get("date") or commit_obj.get("committer", {}).get("date")
        rows.append(
            {
                "sha": sha,
                "short_sha": sha[:7] if sha else "",
                "commit_title": message,
                "commit_date": pd.to_datetime(date_str, utc=True),
                "html_url": c.get("html_url"),
            }
        )
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("commit_date", ascending=False).reset_index(drop=True)
    return df


@st.cache_data(ttl=86400)
def get_cached_today() -> pd.Timestamp.date:
    """Return today's UTC date, cached for 24 hours."""
    return pd.Timestamp.utcnow().date()


def _status_to_symbol(status_value: str) -> str:
    if not status_value:
        return ""
    s = str(status_value).lower()
    if s in {"good", "pass", "passed", "success", "ok", "green"}:
        return "✅"
    if s in {"fail", "failed", "error", "broken", "red"}:
        return "❌"
    return "⚠️"


def _is_pass(status_value: str) -> bool:
    if not status_value:
        return False
    return str(status_value).lower() in {"good", "pass", "passed", "success", "ok", "green"}


@st.cache_data(ttl=300)
def load_ci_results_from_disk(data_dir: Path) -> Dict[str, Dict]:
    """Load CI result JSON files from disk, keyed by full commit SHA.

    If multiple files exist for the same commit, keep the latest by 'created' timestamp in the JSON
    or by filename suffix timestamp if 'created' is missing.
    """
    results_by_sha: Dict[str, Dict] = {}
    if not data_dir.exists():
        return results_by_sha
    for json_path in sorted(data_dir.glob("*.json")):
        try:
            with open(json_path, "r") as f:
                data = json.load(f)
        except Exception:
            continue
        commit_sha = data.get("commit")
        created = data.get("created")
        # Fallback: try parse from filename pattern {SHA}_{YYMMDD_HHMMSS}.json
        if not commit_sha:
            try:
                commit_sha = json_path.stem.split("_")[0]
            except Exception:
                commit_sha = None
        # Keep the most recent created
        if not commit_sha:
            continue
        prev = results_by_sha.get(commit_sha)
        prev_created = prev.get("created") if prev else None
        if prev_created is None or (created and str(created) > str(prev_created)):
            results_by_sha[commit_sha] = {
                "status": data.get("status"),
                "created": created,
                "pipeline_id": data.get("pipeline_id"),
                "tests": data.get("tests", {}),
                "source_file": str(json_path),
            }
    return results_by_sha


# Removed synthetic data generator to rely solely on on-disk JSONs


def collect_filtered_test_names(
    commits_df: pd.DataFrame,
    results_by_sha: Dict[str, Dict],
    tests_regex: str,
    date_range: Tuple[pd.Timestamp, pd.Timestamp],
) -> List[str]:
    if commits_df.empty:
        return []

    start_ts, end_ts = date_range
    mask = (commits_df["commit_date"] >= start_ts) & (commits_df["commit_date"] <= end_ts)
    base = commits_df.loc[mask, ["sha"]].copy()

    # Collect all test names across results for selected commits
    test_names: set = set()
    for sha in base["sha"].tolist():
        item = results_by_sha.get(sha)
        if not item:
            continue
        test_names.update(item.get("tests", {}).keys())

    # Apply regex filter to test columns
    try:
        pattern = re.compile(tests_regex) if tests_regex else re.compile(".*")
        filtered_tests = sorted([t for t in test_names if pattern.search(t)])
    except re.error:
        filtered_tests = sorted(test_names)
    return filtered_tests


def build_commit_ci_table(
    commits_df: pd.DataFrame,
    results_by_sha: Dict[str, Dict],
    filtered_tests: List[str],
    date_range: Tuple[pd.Timestamp, pd.Timestamp],
) -> pd.DataFrame:
    if commits_df.empty:
        return pd.DataFrame()

    start_ts, end_ts = date_range
    mask = (commits_df["commit_date"] >= start_ts) & (commits_df["commit_date"] <= end_ts)
    base = commits_df.loc[mask, ["commit_date", "short_sha", "sha", "commit_title"]].copy()

    # Prepare output rows
    output_rows: List[Dict] = []
    for _, row in base.iterrows():
        sha = row["sha"]
        res = results_by_sha.get(sha)
        tests_obj = res.get("tests", {}) if res else {}
        statuses = {name: tests_obj.get(name, {}).get("status") for name in filtered_tests}
        pass_flags = [
            _is_pass(statuses[name]) for name in filtered_tests if name in statuses and statuses[name] is not None
        ]
        pct_pass: Optional[float] = round(100 * (sum(pass_flags) / len(pass_flags)), 1) if pass_flags else None
        overall = res.get("status") if res else None
        if overall is None and pass_flags:
            overall = "good" if all(pass_flags) else ("fail" if any(not p for p in pass_flags) else "unknown")
        overall_symbol = "🟢" if overall and _is_pass(overall) else ("🔴" if overall else "")
        output = {
            "commit date": row["commit_date"],
            "commit": row["short_sha"],
            "commit title": row["commit_title"],
            "overall": overall_symbol,
            "% passed": pct_pass if pct_pass is not None else np.nan,
        }
        for name in filtered_tests:
            output[name] = _status_to_symbol(statuses.get(name)) if name in statuses else ""
        output_rows.append(output)

    df = pd.DataFrame(output_rows)
    # Sort columns: fixed columns first, then alphabetical test names
    fixed_cols = [
        "commit date",
        "commit",
        "commit title",
        "overall",
        "% passed",
    ]
    test_cols = [c for c in df.columns if c not in fixed_cols]
    df = df[fixed_cols + sorted(test_cols)]
    # Sort rows by commit date desc
    if not df.empty:
        df = df.sort_values("commit date", ascending=False).reset_index(drop=True)
    return df


# Sidebar controls
with st.sidebar:
    st.markdown("**Data Sources**")
    st.write(
        "Commits fetched from the GitHub repo `NVIDIA-NeMo/RL` (main branch). See repo: [GitHub]"
        "(https://github.com/NVIDIA-NeMo/RL)."
    )
    tests_regex = st.text_input(
        "Filter tests by regex",
        value="",
        placeholder="e.g., grpo|dpo|sft (empty = all)",
        help="Press Enter to apply. Empty means all tests (.*)",
    )
    st.caption(f"CI JSON directory: `{DATA_DIR}`")

# Load commits and CI results
commits_df = fetch_commits(max_commits=int(st.session_state.get("max_commits", DEFAULT_MAX_COMMITS)))
results_by_sha = load_ci_results_from_disk(DATA_DIR)
# Ensure we have CI rows for selected top commits; if missing, write sample files
if not commits_df.empty:
    try:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        indices = [0, 3] if len(commits_df) >= 4 else [0]
        wrote_any = False
        for i, idx in enumerate(indices):
            row = commits_df.iloc[idx]
            sha = row["sha"]
            if sha in results_by_sha:
                continue
            created_ts = pd.to_datetime(row["commit_date"]).to_pydatetime()
            created_iso = created_ts.replace(tzinfo=None).isoformat() + "Z"
            stamp = created_ts.strftime("%y%m%d_%H%M%S")
            if i == 0:
                tests = {
                    "grpo-deepscaler": {"status": "good", "metrics": {"step_time": 0.27, "tokens_per_s": 1200}},
                    "sft-llama": {"status": "good", "metrics": {"val_loss": 2.03}},
                    "dpo-helpsteer": {"status": "fail", "metrics": {"preference_accuracy": 0.61}},
                }
                overall = "good"
            else:
                tests = {
                    "grpo-deepscaler": {"status": "fail", "metrics": {"step_time": 0.35}},
                    "rm-helpsteer": {"status": "good", "metrics": {"pair_accuracy": 0.72}},
                    "eval-math500": {"status": "good", "metrics": {"pass_at_1": 0.34}},
                }
                overall = "fail"
            doc = {
                "status": overall,
                "pipeline_id": 800000 + idx,
                "commit": sha,
                "created": created_iso,
                "tests": tests,
            }
            out_path = DATA_DIR / f"{sha}_{stamp}.json"
            with open(out_path, "w") as f:
                json.dump(doc, f, indent=2)
            wrote_any = True
        if wrote_any:
            try:
                load_ci_results_from_disk.clear()
            except Exception:
                pass
            results_by_sha = load_ci_results_from_disk(DATA_DIR)
    except Exception:
        pass

# Date range control based on commit dates
if not commits_df.empty:
    min_date = pd.to_datetime(commits_df["commit_date"].min()).tz_convert("UTC").date()
    max_date = pd.to_datetime(commits_df["commit_date"].max()).tz_convert("UTC").date()
else:
    today = get_cached_today()
    min_date = today
    max_date = today

# Default: 3 weeks ago to today, clamped to available commit dates
today = get_cached_today()
default_start = max(min_date, today - pd.Timedelta(days=14))
default_end = min(max_date, today)

date_range_input = st.date_input(
    "Commit date range",
    value=(default_start, default_end),
    min_value=min_date,
    max_value=max_date,
)

# Normalize to timestamps spanning the selected days fully
if isinstance(date_range_input, tuple) and len(date_range_input) == 2:
    start_dt = pd.Timestamp(date_range_input[0]).tz_localize("UTC")
    end_dt = pd.Timestamp(date_range_input[1]).tz_localize("UTC") + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
else:
    start_dt = pd.Timestamp(min_date).tz_localize("UTC")
    end_dt = pd.Timestamp(max_date).tz_localize("UTC")

filtered_tests = collect_filtered_test_names(
    commits_df=commits_df,
    results_by_sha=results_by_sha,
    tests_regex=tests_regex if tests_regex else ".*",
    date_range=(start_dt, end_dt),
)

# Preview matching tests in sidebar
with st.sidebar:
    if filtered_tests:
        st.markdown("**Matching tests**")
        st.code("\n".join(filtered_tests), language="text")
    else:
        st.markdown("**Matching tests**")
        st.code("<none>", language="text")

with st.sidebar:
    with st.expander("Advanced settings", expanded=False):
        jitter_amplitude = st.slider(
            "Point jitter amplitude",
            min_value=0.0,
            max_value=0.06,
            value=0.015,
            step=0.005,
            help=(
                "Controls how much vertical jitter is applied to points to reduce overlap. "
                "Set to 0 to disable jitter."
            ),
        )
        st.session_state["jitter_amplitude"] = float(jitter_amplitude)

        max_commits = st.slider(
            "Max commits to fetch",
            min_value=50,
            max_value=500,
            value=st.session_state.get("max_commits", DEFAULT_MAX_COMMITS),
            step=50,
            help="Limits how many recent commits are pulled from GitHub.",
        )
        st.session_state["max_commits"] = int(max_commits)

# Early diagnostics if no tests detected
if not results_by_sha:
    st.warning("No CI JSON files found under data/nemo_rl/. Add files to see test columns.")
elif not filtered_tests:
    st.info("No tests match the current date range and regex. Adjust filters or add CI JSONs.")

table_df = build_commit_ci_table(
    commits_df=commits_df,
    results_by_sha=results_by_sha,
    filtered_tests=filtered_tests,
    date_range=(start_dt, end_dt),
)

st.subheader("Commit CI Status")
if table_df.empty:
    st.info("No data available for the selected range.")
else:
    # Light formatting for display
    display_df = table_df.copy()
    display_df["commit date"] = pd.to_datetime(display_df["commit date"]).dt.tz_convert(None)
    fixed_cols = [
        "commit date",
        "commit",
        "commit title",
        "overall",
        "% passed",
    ]
    gb = GridOptionsBuilder.from_dataframe(display_df)
    gb.configure_default_column(resizable=True, sortable=True, filter=False)
    for col in fixed_cols:
        if col in display_df.columns:
            gb.configure_column(col, pinned="left")
    grid_options = gb.build()

    # Build line chart: one line per test, y=pass/fail (1/0), x=commit date
    if filtered_tests:
        # Prepare long-form dataframe for chart
        base = commits_df[(commits_df["commit_date"] >= start_dt) & (commits_df["commit_date"] <= end_dt)].copy()
        if not base.empty:
            # For each commit and test, compute pass flag
            records: List[Dict] = []
            sha_to_meta = base.set_index("sha")[["short_sha", "commit_title", "commit_date"]].to_dict(orient="index")
            for sha, meta in sha_to_meta.items():
                ci_obj = results_by_sha.get(sha, {})
                tests_obj = ci_obj.get("tests", {})
                for test_name in filtered_tests:
                    test_entry = tests_obj.get(test_name)
                    status = (test_entry or {}).get("status") if test_entry is not None else None
                    if test_entry is None or status is None:
                        pass_value = -1
                        metrics = {}
                    else:
                        pass_value = 1 if _is_pass(status) else 0
                        metrics = test_entry.get("metrics", {}) or {}
                    if isinstance(metrics, dict):
                        # Format metrics as key=value comma-separated string for tooltip
                        metrics_str = ", ".join([f"{k}={metrics[k]}" for k in sorted(metrics.keys())]) if metrics else ""
                    else:
                        metrics_str = ""
                    records.append(
                        {
                            "commit_date": meta["commit_date"],
                            "test": test_name,
                            "pass_value": pass_value,
                            "short_sha": meta["short_sha"],
                            "commit_title": meta["commit_title"],
                            "metrics": metrics_str,
                        }
                    )
            chart_df = pd.DataFrame(records)
            if not chart_df.empty:
                # Deterministic small jitter per test to separate overlapping points
                def _test_jitter_value(test_name: str) -> float:
                    h = int(hashlib.sha256(test_name.encode("utf-8")).hexdigest(), 16)
                    amplitude = float(st.session_state.get("jitter_amplitude", 0.015))
                    return (((h % 11) - 5) * amplitude)

                # Jitter only failures (pass_value == 0). Keep others on exact bands (-1 and 1)
                jitter = chart_df["test"].map(_test_jitter_value)
                is_fail = chart_df["pass_value"].astype(float) == 0.0
                # Cap jitter within +-0.15 so points do not cross tick lines visually
                capped_jitter = jitter.clip(lower=-0.15, upper=0.15)
                chart_df["y_jitter"] = chart_df["pass_value"].astype(float)
                chart_df.loc[is_fail, "y_jitter"] = chart_df.loc[is_fail, "pass_value"].astype(float) + capped_jitter[is_fail]

                lines = (
                    alt.Chart(chart_df)
                    .mark_line(strokeOpacity=0.6, strokeWidth=1)
                    .encode(
                        x=alt.X("commit_date:T", title="Commit Date"),
                        y=alt.Y(
                            "pass_value:Q",
                            title="Pass (1) / Fail (0) / Not run (-1)",
                            scale=alt.Scale(domain=[-1.15, 1.15]),
                            axis=alt.Axis(values=[-1, 0, 1]),
                        ),
                        color=alt.Color("test:N", title="Test"),
                    )
                )

                points = (
                    alt.Chart(chart_df)
                    .mark_circle(size=42, opacity=0.9)
                    .encode(
                        x="commit_date:T",
                        y=alt.Y("y_jitter:Q", title=None),
                        color=alt.Color("test:N", title="Test"),
                        tooltip=[
                            alt.Tooltip("test:N", title="Test"),
                            alt.Tooltip("short_sha:N", title="Commit"),
                            alt.Tooltip("commit_title:N", title="Title"),
                            alt.Tooltip("commit_date:T", title="Date"),
                            alt.Tooltip("metrics:N", title="Metrics"),
                        ],
                    )
                )

                st.altair_chart((lines + points).properties(height=360), use_container_width=True)

    AgGrid(
        display_df,
        gridOptions=grid_options,
        height=800,
        theme="streamlit",
        enable_enterprise_modules=False,
        fit_columns_on_grid_load=False,
    )

st.caption(
    "Commits source: NVIDIA-NeMo/RL on GitHub. See repository: https://github.com/NVIDIA-NeMo/RL"
)
