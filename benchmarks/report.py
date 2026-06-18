"""
Report writers for benchmark suite results.

Provides two output formats from the same list-of-dicts result data:
  - to_html(results, path)     — interactive Plotly HTML (requires plotly, pandas)
  - to_markdown(results, path) — GitHub-readable .md with Mermaid convergence chart

The result dict contract (each element produced by runner.run_benchmark) is the
~15-key schema from runner.stats_dict; mandatory keys:
  benchmark_name, seed, solve_generation, total_generations, winner_fitness,
  final_mean_reward, final_std_reward, wall_time_seconds, winner_nodes,
  winner_connections, generation_stats
Optional: training_rss_mb, peak_gpu_mb, validation_stats, winner_path, env_id.
"""

from __future__ import annotations

import time

import numpy as np

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_PALETTE = [
    "#636EFA",
    "#EF553B",
    "#00CC96",
    "#AB63FA",
    "#FFA15A",
    "#19D3F3",
    "#FF6692",
    "#B6E880",
]


def _group_by_name(results: list[dict]) -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = {}
    for r in results:
        groups.setdefault(r["benchmark_name"], []).append(r)
    return groups


def _summary_table_rows(all_results: list[dict]) -> list[dict]:
    rows = []
    for name, group in _group_by_name(all_results).items():
        solved = [r for r in group if r.get("solve_generation") is not None]
        solve_gens = [r["solve_generation"] for r in solved]
        rewards = [r["final_mean_reward"] for r in group]
        times = [r["wall_time_seconds"] for r in group]
        rss = [r["training_rss_mb"] for r in group if r.get("training_rss_mb") is not None]
        gpu = [r["peak_gpu_mb"] for r in group if r.get("peak_gpu_mb") is not None]
        nodes = [r["winner_nodes"] for r in group]
        conns = [r["winner_connections"] for r in group]
        rows.append(
            {
                "Benchmark": name,
                "Runs": len(group),
                "Solved %": f"{100 * len(solved) / len(group):.0f}%",
                "Median solve gen": f"{int(np.median(solve_gens))}" if solve_gens else "—",
                "P25–P75 gen": (
                    f"{int(np.percentile(solve_gens, 25))}–{int(np.percentile(solve_gens, 75))}"
                    if len(solve_gens) > 1
                    else "—"
                ),
                "Median reward": f"{np.median(rewards):.1f}",
                "Reward std": f"{np.std(rewards):.1f}",
                "Median time (s)": f"{np.median(times):.1f}",
                "Median RSS Δ (MB)": f"{np.median(rss):.1f}" if rss else "—",
                "Peak GPU (MB)": f"{np.median(gpu):.1f}" if gpu else "—",
                "Median nodes": f"{np.median(nodes):.0f}",
                "Median conns": f"{np.median(conns):.0f}",
            }
        )
    return rows


# ---------------------------------------------------------------------------
# HTML report (Plotly)
# ---------------------------------------------------------------------------


def _summary_table_html(all_results: list[dict]) -> str:
    import pandas as pd

    df = pd.DataFrame(_summary_table_rows(all_results))
    return df.to_html(index=False, border=0, classes="summary-table")


def _fig_convergence(all_results, colors):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    groups = _group_by_name(all_results)
    n = len(groups)
    fig = make_subplots(rows=1, cols=n, subplot_titles=list(groups.keys()), shared_yaxes=True)
    for col, (name, group) in enumerate(groups.items(), start=1):
        color = colors[name]
        for i, run in enumerate(group):
            gens = [s["generation"] for s in run["generation_stats"]]
            bests = [s["best"] for s in run["generation_stats"]]
            fig.add_trace(
                go.Scatter(
                    x=gens,
                    y=bests,
                    mode="lines",
                    line=dict(color=color, width=1),
                    opacity=0.45,
                    name=f"{name} run {i + 1}",
                    legendgroup=name,
                    showlegend=(i == 0),
                ),
                row=1,
                col=col,
            )
        all_bests = [[s["best"] for s in r["generation_stats"]] for r in group]
        min_len = min(len(b) for b in all_bests)
        mean_bests = np.mean([b[:min_len] for b in all_bests], axis=0)
        fig.add_trace(
            go.Scatter(
                x=list(range(min_len)),
                y=mean_bests.tolist(),
                mode="lines",
                line=dict(color=color, width=3),
                name=f"{name} mean",
                legendgroup=name,
            ),
            row=1,
            col=col,
        )
        threshold = next((r.get("_solve_threshold") for r in group if r.get("_solve_threshold")), None)
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


def _fig_boxes(all_results, colors):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    groups = _group_by_name(all_results)
    metrics = [
        ("solve_generation", "Solve generation", "Generation"),
        ("final_mean_reward", "Final mean reward", "Reward"),
        ("wall_time_seconds", "Training wall time", "Seconds"),
    ]
    fig = make_subplots(rows=1, cols=len(metrics), subplot_titles=[m[1] for m in metrics])
    for col, (key, _, ylabel) in enumerate(metrics, start=1):
        for name, group in groups.items():
            vals = [r[key] for r in group if r.get(key) is not None]
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
                row=1,
                col=col,
            )
        fig.update_yaxes(title_text=ylabel, row=1, col=col)
    fig.update_layout(
        title="Distribution across runs",
        height=420,
        template="plotly_dark",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    return fig


def _fig_network_complexity(all_results, colors):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    groups = _group_by_name(all_results)
    fig = make_subplots(rows=1, cols=2, subplot_titles=["Winner nodes", "Winner connections"])
    for name, group in groups.items():
        for col, vals in [(1, [r["winner_nodes"] for r in group]), (2, [r["winner_connections"] for r in group])]:
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
                row=1,
                col=col,
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


def _fig_memory(all_results, colors):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    groups = _group_by_name(all_results)
    has_rss = any(r.get("training_rss_mb") is not None for r in all_results)
    has_gpu = any(r.get("peak_gpu_mb") is not None for r in all_results)
    cols = int(has_rss) + int(has_gpu)
    if cols == 0:
        return None
    titles = (["Training RSS delta (MB)"] if has_rss else []) + (["Peak GPU memory (MB)"] if has_gpu else [])
    fig = make_subplots(rows=1, cols=cols, subplot_titles=titles)
    for name, group in groups.items():
        col = 1
        if has_rss:
            vals = [r["training_rss_mb"] for r in group if r.get("training_rss_mb") is not None]
            if vals:
                fig.add_trace(
                    go.Box(
                        y=vals,
                        name=name,
                        marker_color=colors[name],
                        legendgroup=name,
                        showlegend=True,
                        boxpoints="all",
                        jitter=0.3,
                    ),
                    row=1,
                    col=col,
                )
            col += 1
        if has_gpu:
            vals = [r["peak_gpu_mb"] for r in group if r.get("peak_gpu_mb") is not None]
            if vals:
                fig.add_trace(
                    go.Box(
                        y=vals,
                        name=name,
                        marker_color=colors[name],
                        legendgroup=name,
                        showlegend=(not has_rss),
                        boxpoints="all",
                        jitter=0.3,
                    ),
                    row=1,
                    col=col,
                )
    fig.update_layout(
        title="Memory usage during training",
        height=380,
        template="plotly_dark",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    return fig


def _fig_mean_fitness_bands(all_results, colors):
    import plotly.graph_objects as go

    groups = _group_by_name(all_results)
    fig = go.Figure()
    for name, group in groups.items():
        color = colors[name]
        all_means = [
            [s["mean"] for s in r["generation_stats"]] for r in group if all("mean" in s for s in r["generation_stats"])
        ]
        if not all_means:
            continue
        min_len = min(len(m) for m in all_means)
        arr = np.array([m[:min_len] for m in all_means])
        mu = arr.mean(axis=0)
        sigma = arr.std(axis=0)
        gens = list(range(min_len))
        fig.add_trace(
            go.Scatter(
                x=gens + gens[::-1],
                y=(mu + sigma).tolist() + (mu - sigma).tolist()[::-1],
                fill="toself",
                fillcolor=color,
                opacity=0.15,
                line=dict(color="rgba(0,0,0,0)"),
                name=f"{name} ±σ",
                legendgroup=name,
                showlegend=True,
            )
        )
        fig.add_trace(
            go.Scatter(
                x=gens,
                y=mu.tolist(),
                mode="lines",
                line=dict(color=color, width=2),
                name=f"{name} mean",
                legendgroup=name,
            )
        )
    fig.update_layout(
        title="Population mean fitness ± std across runs",
        xaxis_title="Generation",
        yaxis_title="Mean fitness",
        height=400,
        template="plotly_dark",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    return fig


def _fig_validation(all_results, colors):
    import plotly.graph_objects as go

    groups = _group_by_name(all_results)
    if not any(r.get("validation_stats") for r in all_results):
        return None
    fig = go.Figure()
    for name, group in groups.items():
        series = [[s["val_mean"] for s in r.get("validation_stats", [])] for r in group]
        series = [s for s in series if len(s) >= 2]
        if not series:
            continue
        min_len = min(len(s) for s in series)
        mu = np.mean([s[:min_len] for s in series], axis=0)
        fig.add_trace(
            go.Scatter(
                x=list(range(min_len)),
                y=mu.tolist(),
                mode="lines",
                line=dict(color=colors[name], width=2),
                name=name,
                legendgroup=name,
            )
        )
    fig.update_layout(
        title="Held-out validation — champion mean reward per generation (clean progress signal)",
        xaxis_title="Generation",
        yaxis_title="Held-out mean reward",
        height=400,
        template="plotly_dark",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    return fig


def to_html(all_results: list[dict], output_path: str) -> None:
    """Build a self-contained Plotly HTML report and write it to output_path."""
    benchmark_names = list(dict.fromkeys(r["benchmark_name"] for r in all_results))
    colors = {name: _PALETTE[i % len(_PALETTE)] for i, name in enumerate(benchmark_names)}

    summary_html = _summary_table_html(all_results)
    fig_convergence = _fig_convergence(all_results, colors)
    fig_validation = _fig_validation(all_results, colors)
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
    h2   {{ font-size: 1.2em; color: #aaa; border-bottom: 1px solid #333;
            padding-bottom: 6px; margin-top: 40px; }}
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

  <h2>Held-out validation — clean progress signal</h2>
  <div class="chart-wrap">{_to_div(fig_validation)}</div>

  <h2>Convergence — best fitness per generation</h2>
  <div class="chart-wrap">{_to_div(fig_convergence)}</div>

  <h2>Population mean fitness ± std across runs (training worlds — jittery by design)</h2>
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
    print(f"\nHTML report saved → {output_path}")


# ---------------------------------------------------------------------------
# Markdown report (GitHub-readable, Mermaid convergence chart)
# ---------------------------------------------------------------------------


def to_markdown(all_results: list[dict], output_path: str) -> None:
    """Write a GitHub-readable .md report with summary table + Mermaid convergence chart."""
    benchmark_names = list(dict.fromkeys(r["benchmark_name"] for r in all_results))
    groups = _group_by_name(all_results)
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    run_summary = ", ".join(f"{k}: {len(v)} runs" for k, v in groups.items())

    rows = _summary_table_rows(all_results)
    if not rows:
        return

    cols = list(rows[0].keys())
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    body_lines = ["| " + " | ".join(str(r.get(c, "—")) for c in cols) + " |" for r in rows]
    table_md = "\n".join([header, sep] + body_lines)

    mermaid_lines = [
        "```mermaid",
        "xychart-beta",
        '  title "Best fitness per generation (mean across runs)"',
        '  x-axis "Generation"',
        '  y-axis "Best fitness"',
    ]
    for name, group in groups.items():
        all_bests = [[s["best"] for s in r["generation_stats"]] for r in group]
        if not all_bests:
            continue
        min_len = min(len(b) for b in all_bests)
        mean_bests = np.mean([b[:min_len] for b in all_bests], axis=0)
        vals = ", ".join(f"{v:.1f}" for v in mean_bests.tolist())
        label = name.replace("_", " ")
        mermaid_lines.append(f'  line "{label}" [{vals}]')
    mermaid_lines.append("```")
    mermaid_md = "\n".join(mermaid_lines)

    winner_links = []
    for r in all_results:
        if r.get("winner_path"):
            label = f"{r['benchmark_name']} seed={r.get('seed', '?')}"
            winner_links.append(f"- [{label}]({r['winner_path']})")
    winners_md = "\n".join(winner_links) if winner_links else "_No winner paths recorded._"

    md = f"""# neat3p Benchmark Report

Generated {timestamp} · {run_summary}

## Summary

{table_md}

## Convergence

{mermaid_md}

## Saved winners

{winners_md}
"""

    with open(output_path, "w") as f:
        f.write(md)
    print(f"\nMarkdown report saved → {output_path}")


def build_report(all_results: list[dict], benchmark_names: list[str], output_path: str, format: str = "html") -> None:
    """Dispatch to to_html / to_markdown / both based on format string."""
    if format in ("html", "both"):
        html_path = output_path if output_path.endswith(".html") else output_path + ".html"
        to_html(all_results, html_path)
    if format in ("md", "both"):
        md_path = (output_path[:-5] if output_path.endswith(".html") else output_path) + ".md"
        to_markdown(all_results, md_path)
