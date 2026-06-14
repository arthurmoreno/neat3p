#!/usr/bin/env python3
"""
neat3p benchmark suite — run registered benchmarks N times, collect statistics,
and generate a self-contained Plotly HTML report.

Usage:
    python benchmarks/suite.py                                 # all benchmarks, 3 runs each
    python benchmarks/suite.py --benchmarks cartpole           # single benchmark
    python benchmarks/suite.py --runs 5                        # 5 runs each
    python benchmarks/suite.py --seeds 42,123,456              # explicit seeds (sets --runs too)
    python benchmarks/suite.py --output my_report.html         # custom output path
    python benchmarks/suite.py --generations 200               # override max generations

Adding a new benchmark:
    1. Create benchmarks/<name>.py with BENCHMARK_NAME, SOLVE_THRESHOLD, and run_benchmark(seed, **kwargs) -> dict.
    2. Add it to REGISTRY below.
"""

import argparse
import importlib.util
import os
import sys
import time

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ---------------------------------------------------------------------------
# Registry — name → file path relative to this script's directory
# ---------------------------------------------------------------------------

_HERE = os.path.dirname(os.path.abspath(__file__))

REGISTRY: dict[str, str] = {
    "cartpole": "cartpole.py",
    "cartpole_feature_attention": "cartpole_feature_attention.py",
}

# Consistent colour per benchmark across all charts
_PALETTE = [
    "#636EFA", "#EF553B", "#00CC96", "#AB63FA",
    "#FFA15A", "#19D3F3", "#FF6692", "#B6E880",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_module(name: str, rel_path: str):
    path = os.path.join(_HERE, rel_path)
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    # Make sure the benchmarks dir is importable so relative imports inside work
    if _HERE not in sys.path:
        sys.path.insert(0, _HERE)
    spec.loader.exec_module(mod)
    return mod


def _seeds_for(n_runs: int, explicit: list[int] | None) -> list[int]:
    if explicit:
        return explicit
    return list(range(42, 42 + n_runs))


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def _summary_table(all_results: list[dict], solve_threshold: dict[str, float]) -> str:
    rows = []
    for name, group in _group_by_name(all_results).items():
        solved = [r for r in group if r["solve_generation"] is not None]
        solve_gens = [r["solve_generation"] for r in solved]
        rewards = [r["final_mean_reward"] for r in group]
        times = [r["wall_time_seconds"] for r in group]
        rss = [r["training_rss_mb"] for r in group if r["training_rss_mb"] is not None]
        gpu = [r["peak_gpu_mb"] for r in group if r["peak_gpu_mb"] is not None]
        nodes = [r["winner_nodes"] for r in group]
        conns = [r["winner_connections"] for r in group]
        rows.append({
            "Benchmark": name,
            "Runs": len(group),
            "Solved %": f"{100 * len(solved) / len(group):.0f}%",
            "Median solve gen": f"{int(np.median(solve_gens))}" if solve_gens else "—",
            "P25–P75 gen": (
                f"{int(np.percentile(solve_gens, 25))}–{int(np.percentile(solve_gens, 75))}"
                if len(solve_gens) > 1 else "—"
            ),
            "Median reward": f"{np.median(rewards):.1f}",
            "Reward std": f"{np.std(rewards):.1f}",
            "Median time (s)": f"{np.median(times):.1f}",
            "Median RSS Δ (MB)": f"{np.median(rss):.1f}" if rss else "—",
            "Peak GPU (MB)": f"{np.median(gpu):.1f}" if gpu else "—",
            "Median nodes": f"{np.median(nodes):.0f}",
            "Median conns": f"{np.median(conns):.0f}",
        })
    df = pd.DataFrame(rows)
    return df.to_html(index=False, border=0, classes="summary-table")


def _group_by_name(results: list[dict]) -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = {}
    for r in results:
        groups.setdefault(r["benchmark_name"], []).append(r)
    return groups


def _fig_convergence(all_results: list[dict], colors: dict[str, str]) -> go.Figure:
    groups = _group_by_name(all_results)
    n = len(groups)
    fig = make_subplots(
        rows=1, cols=n,
        subplot_titles=[f"{name}" for name in groups],
        shared_yaxes=True,
    )
    for col, (name, group) in enumerate(groups.items(), start=1):
        color = colors[name]
        for i, run in enumerate(group):
            gens = [s["generation"] for s in run["generation_stats"]]
            bests = [s["best"] for s in run["generation_stats"]]
            fig.add_trace(
                go.Scatter(
                    x=gens, y=bests,
                    mode="lines",
                    line=dict(color=color, width=1),
                    opacity=0.45,
                    name=f"{name} run {i + 1}",
                    legendgroup=name,
                    showlegend=(i == 0),
                ),
                row=1, col=col,
            )
        # Mean across runs (aligned by generation)
        all_bests = [[s["best"] for s in r["generation_stats"]] for r in group]
        min_len = min(len(b) for b in all_bests)
        mean_bests = np.mean([b[:min_len] for b in all_bests], axis=0)
        fig.add_trace(
            go.Scatter(
                x=list(range(min_len)), y=mean_bests.tolist(),
                mode="lines",
                line=dict(color=color, width=3),
                name=f"{name} mean",
                legendgroup=name,
            ),
            row=1, col=col,
        )
        # Solve threshold line
        threshold = next(
            (r.get("_solve_threshold") for r in group if r.get("_solve_threshold")), None
        )
        if threshold is not None:
            fig.add_hline(y=threshold, line_dash="dot", line_color="grey", row=1, col=col)

    fig.update_layout(
        title="Convergence curves — best fitness per generation",
        height=400,
        template="plotly_dark",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    fig.update_yaxes(title_text="Best fitness")
    fig.update_xaxes(title_text="Generation")
    return fig


def _fig_boxes(all_results: list[dict], colors: dict[str, str]) -> go.Figure:
    groups = _group_by_name(all_results)

    metrics = [
        ("solve_generation", "Solve generation", "Generation"),
        ("final_mean_reward", "Final mean reward (100 ep)", "Reward"),
        ("wall_time_seconds", "Training wall time", "Seconds"),
    ]
    fig = make_subplots(rows=1, cols=len(metrics), subplot_titles=[m[1] for m in metrics])

    for col, (key, _, yaxis_label) in enumerate(metrics, start=1):
        for name, group in groups.items():
            vals = [r[key] for r in group if r[key] is not None]
            if not vals:
                continue
            fig.add_trace(
                go.Box(
                    y=vals,
                    name=name,
                    marker_color=colors[name],
                    legendgroup=name,
                    showlegend=(col == 1),
                    boxpoints="all",
                    jitter=0.3,
                    pointpos=-1.5,
                ),
                row=1, col=col,
            )
        fig.update_yaxes(title_text=yaxis_label, row=1, col=col)

    fig.update_layout(
        title="Distribution across runs",
        height=420,
        template="plotly_dark",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    return fig


def _fig_network_complexity(all_results: list[dict], colors: dict[str, str]) -> go.Figure:
    groups = _group_by_name(all_results)
    fig = make_subplots(rows=1, cols=2, subplot_titles=["Winner nodes", "Winner connections"])

    for name, group in groups.items():
        nodes = [r["winner_nodes"] for r in group]
        conns = [r["winner_connections"] for r in group]
        for col, vals in [(1, nodes), (2, conns)]:
            fig.add_trace(
                go.Box(
                    y=vals,
                    name=name,
                    marker_color=colors[name],
                    legendgroup=name,
                    showlegend=(col == 1),
                    boxpoints="all",
                    jitter=0.3,
                    pointpos=-1.5,
                ),
                row=1, col=col,
            )

    fig.update_layout(
        title="Winner genome complexity",
        height=380,
        template="plotly_dark",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    fig.update_yaxes(title_text="Count", col=1)
    fig.update_yaxes(title_text="Count", col=2)
    return fig


def _fig_memory(all_results: list[dict], colors: dict[str, str]) -> go.Figure:
    groups = _group_by_name(all_results)
    has_rss = any(r["training_rss_mb"] is not None for r in all_results)
    has_gpu = any(r["peak_gpu_mb"] is not None for r in all_results)

    cols = int(has_rss) + int(has_gpu)
    if cols == 0:
        return None

    titles = []
    if has_rss:
        titles.append("Training RSS delta (MB)")
    if has_gpu:
        titles.append("Peak GPU memory (MB)")

    fig = make_subplots(rows=1, cols=cols, subplot_titles=titles)

    for name, group in groups.items():
        col = 1
        if has_rss:
            vals = [r["training_rss_mb"] for r in group if r["training_rss_mb"] is not None]
            if vals:
                fig.add_trace(
                    go.Box(y=vals, name=name, marker_color=colors[name],
                           legendgroup=name, showlegend=True, boxpoints="all", jitter=0.3),
                    row=1, col=col,
                )
            col += 1
        if has_gpu:
            vals = [r["peak_gpu_mb"] for r in group if r["peak_gpu_mb"] is not None]
            if vals:
                fig.add_trace(
                    go.Box(y=vals, name=name, marker_color=colors[name],
                           legendgroup=name, showlegend=(not has_rss), boxpoints="all", jitter=0.3),
                    row=1, col=col,
                )

    fig.update_layout(
        title="Memory usage during training",
        height=380,
        template="plotly_dark",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    return fig


def _fig_mean_fitness_bands(all_results: list[dict], colors: dict[str, str]) -> go.Figure:
    """Mean ± std fitness band per benchmark, overlaid on one chart."""
    groups = _group_by_name(all_results)
    fig = go.Figure()

    for name, group in groups.items():
        color = colors[name]
        all_means = [[s["mean"] for s in r["generation_stats"]] for r in group]
        min_len = min(len(m) for m in all_means)
        arr = np.array([m[:min_len] for m in all_means])
        mu = arr.mean(axis=0)
        sigma = arr.std(axis=0)
        gens = list(range(min_len))

        fig.add_trace(go.Scatter(
            x=gens + gens[::-1],
            y=(mu + sigma).tolist() + (mu - sigma).tolist()[::-1],
            fill="toself",
            fillcolor=color,
            opacity=0.15,
            line=dict(color="rgba(0,0,0,0)"),
            name=f"{name} ±σ",
            legendgroup=name,
            showlegend=True,
        ))
        fig.add_trace(go.Scatter(
            x=gens, y=mu.tolist(),
            mode="lines",
            line=dict(color=color, width=2),
            name=f"{name} mean",
            legendgroup=name,
        ))

    fig.update_layout(
        title="Population mean fitness ± std across runs",
        xaxis_title="Generation",
        yaxis_title="Mean fitness",
        height=400,
        template="plotly_dark",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    return fig


def build_report(all_results: list[dict], benchmark_names: list[str], output_path: str) -> None:
    colors = {name: _PALETTE[i % len(_PALETTE)] for i, name in enumerate(benchmark_names)}

    summary_html = _summary_table(all_results, {})
    fig_convergence = _fig_convergence(all_results, colors)
    fig_mean_bands = _fig_mean_fitness_bands(all_results, colors)
    fig_boxes = _fig_boxes(all_results, colors)
    fig_complexity = _fig_network_complexity(all_results, colors)
    fig_memory = _fig_memory(all_results, colors)

    def _to_div(fig):
        if fig is None:
            return ""
        return fig.to_html(full_html=False, include_plotlyjs=False)

    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    run_summary = ", ".join(f"{k}: {len(v)} runs" for k, v in _group_by_name(all_results).items())

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>neat3p Benchmark Report</title>
  <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
            background: #111; color: #eee; margin: 0; padding: 24px; }}
    h1   {{ font-size: 1.8em; margin-bottom: 4px; }}
    h2   {{ font-size: 1.2em; color: #aaa; border-bottom: 1px solid #333; padding-bottom: 6px; margin-top: 40px; }}
    .meta {{ color: #888; font-size: 0.85em; margin-bottom: 32px; }}
    .chart-wrap {{ margin: 24px 0; }}
    table.summary-table {{ border-collapse: collapse; width: 100%; font-size: 0.85em; }}
    table.summary-table th {{ background: #222; padding: 8px 12px; text-align: left; color: #aaa; }}
    table.summary-table td {{ padding: 7px 12px; border-top: 1px solid #2a2a2a; }}
    table.summary-table tr:hover td {{ background: #1a1a1a; }}
  </style>
</head>
<body>
  <h1>neat3p Benchmark Report</h1>
  <div class="meta">Generated {timestamp} &nbsp;·&nbsp; {run_summary}</div>

  <h2>Summary</h2>
  {summary_html}

  <h2>Convergence — best fitness per generation</h2>
  <div class="chart-wrap">{_to_div(fig_convergence)}</div>

  <h2>Population mean fitness ± std across runs</h2>
  <div class="chart-wrap">{_to_div(fig_mean_bands)}</div>

  <h2>Distributions across runs</h2>
  <div class="chart-wrap">{_to_div(fig_boxes)}</div>

  <h2>Winner genome complexity</h2>
  <div class="chart-wrap">{_to_div(fig_complexity)}</div>

  <h2>Memory usage</h2>
  <div class="chart-wrap">{_to_div(fig_memory)}</div>
</body>
</html>"""

    with open(output_path, "w") as f:
        f.write(html)
    print(f"\nReport saved → {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="neat3p benchmark suite")
    parser.add_argument(
        "--benchmarks", nargs="+",
        default=list(REGISTRY.keys()),
        choices=list(REGISTRY.keys()),
        help="Which benchmarks to run (default: all).",
    )
    parser.add_argument("--runs", type=int, default=3, help="Runs per benchmark (default: 3).")
    parser.add_argument(
        "--seeds", type=str, default=None,
        help="Comma-separated explicit seeds, e.g. 42,123,456. Overrides --runs.",
    )
    parser.add_argument("--generations", type=int, default=None, help="Override max generations per benchmark.")
    parser.add_argument("--output", type=str, default="benchmark_report.html")
    args = parser.parse_args()

    explicit_seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else None
    seeds = _seeds_for(args.runs, explicit_seeds)

    print("=" * 60)
    print("neat3p benchmark suite")
    print(f"  benchmarks : {', '.join(args.benchmarks)}")
    print(f"  seeds      : {seeds}")
    print(f"  output     : {args.output}")
    print("=" * 60)

    all_results: list[dict] = []

    for bname in args.benchmarks:
        mod = _load_module(bname, REGISTRY[bname])
        bench_kwargs = {}
        if args.generations is not None:
            bench_kwargs["generations"] = args.generations

        print(f"\n{'─' * 60}")
        print(f"Benchmark: {bname}")
        print(f"{'─' * 60}")

        for run_i, seed in enumerate(seeds):
            print(f"\n  Run {run_i + 1}/{len(seeds)}  seed={seed}")
            t0 = time.perf_counter()
            result = mod.run_benchmark(seed=seed, verbose=True, **bench_kwargs)
            elapsed = time.perf_counter() - t0
            solved = result["solve_generation"] is not None
            print(
                f"  ✓  {'SOLVED' if solved else 'TIMEOUT'} "
                f"| gen={result.get('solve_generation', result['total_generations'])} "
                f"| reward={result['final_mean_reward']:.1f} "
                f"| {elapsed:.0f}s total"
            )
            all_results.append(result)

    print(f"\n{'=' * 60}")
    print("Building report...")
    build_report(all_results, args.benchmarks, args.output)


if __name__ == "__main__":
    main()
