#!/usr/bin/env python3
"""
Generate presentation-ready plots (SVG + PNG when matplotlib is available) comparing three RL methods and benchmarks.

Outputs (saved to presentation/plot):
- equity_methods.svg/.png: three RL methods
- equity_vs_benchmark.svg/.png: three methods + B&H Equal + Hold VOO
- metrics_bar.svg/.png: annual return, Sharpe, and max drawdown comparison (RL + benchmarks)
"""
from __future__ import annotations
import csv
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import math
from typing import List

ROOT = Path(__file__).resolve().parents[1]
PLOT_DIR = ROOT / "presentation" / "plot"
FREQ_PER_YEAR = 252 * 390  # trading minutes per year for annualization


@dataclass
class Curve:
    name: str
    dates: List[datetime]
    values: List[float]
    returns: List[float]


def load_curve(
    path: Path,
    label: str,
    value_col: str = "account_value",
    return_col: str | None = "return",
    date_col: str = "date",
) -> Curve:
    dates: List[datetime] = []
    values: List[float] = []
    returns: List[float] = []

    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        prev_value = None
        for row in reader:
            date_str = row.get(date_col)
            if not date_str:
                continue
            try:
                dt = datetime.fromisoformat(date_str)
            except ValueError:
                dt = datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S")
            val = float(row[value_col])
            if return_col and return_col in row and row[return_col] not in (None, ""):
                ret = float(row[return_col])
            else:
                ret = 0.0 if prev_value is None else val / prev_value - 1.0
            dates.append(dt)
            values.append(val)
            returns.append(ret)
            prev_value = val

    if not dates:
        raise ValueError(f"No data loaded from {path}")

    return Curve(name=label, dates=dates, values=values, returns=returns)


def compute_stats(curve: Curve) -> dict:
    acc = curve.values
    rets = curve.returns
    n = len(acc)
    total_return = acc[-1] / acc[0] - 1.0 if n > 1 else float("nan")
    ann_return = (1.0 + total_return) ** (FREQ_PER_YEAR / n) - 1.0 if n > 1 else float("nan")
    mean_rets = sum(rets) / n
    var_rets = sum((r - mean_rets) ** 2 for r in rets) / n
    ann_vol = math.sqrt(var_rets) * math.sqrt(FREQ_PER_YEAR)
    sharpe = ann_return / ann_vol if ann_vol > 0 else float("nan")

    running_max = []
    current_max = -float("inf")
    for v in acc:
        current_max = max(current_max, v)
        running_max.append(current_max)
    drawdowns = [v / m - 1.0 for v, m in zip(acc, running_max)]
    max_drawdown = min(drawdowns)

    return {
        "name": curve.name,
        "total_return": total_return,
        "annual_return": ann_return,
        "annual_volatility": ann_vol,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown,
    }


def _svg_header(width: int, height: int) -> List[str]:
    return [
        f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}' viewBox='0 0 {width} {height}'>",
        "<style>text{font-family:Arial,sans-serif;font-size:12px;} .title{font-size:16px;font-weight:bold;}" \
        " .axis{stroke:#333;stroke-width:1;} .grid{stroke:#ccc;stroke-width:0.5;}" \
        "</style>",
    ]


def _svg_footer() -> str:
    return "</svg>"


def svg_line_chart(curves: List[Curve], out_path: Path, title: str, normalize: bool = True):
    width, height = 1100, 560
    margin_left, margin_right, margin_top, margin_bottom = 70, 40, 50, 60
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom

    all_dates = [d for c in curves for d in c.dates]
    min_ts = min(all_dates).timestamp()
    max_ts = max(all_dates).timestamp()
    ts_span = max_ts - min_ts or 1.0

    all_vals = []
    for c in curves:
        base = c.values[0] if normalize else 1.0
        all_vals.extend([v / base for v in c.values])
    min_y, max_y = min(all_vals), max(all_vals)
    pad = (max_y - min_y) * 0.05 if max_y != min_y else 0.05
    min_y -= pad
    max_y += pad
    y_span = max_y - min_y or 1.0

    def x_pos(ts: float) -> float:
        return margin_left + (ts - min_ts) / ts_span * plot_w

    def y_pos(val: float) -> float:
        return margin_top + (max_y - val) / y_span * plot_h

    lines = _svg_header(width, height)

    # Gridlines (5 horizontal)
    for i in range(6):
        y = margin_top + i * plot_h / 5
        val = max_y - i * y_span / 5
        lines.append(f"<line class='grid' x1='{margin_left}' y1='{y:.2f}' x2='{margin_left+plot_w}' y2='{y:.2f}' />")
        lines.append(f"<text x='{margin_left-8}' y='{y+4:.2f}' text-anchor='end'>{val:.2f}</text>")

    # Axes
    lines.append(f"<line class='axis' x1='{margin_left}' y1='{margin_top}' x2='{margin_left}' y2='{margin_top+plot_h}' />")
    lines.append(f"<line class='axis' x1='{margin_left}' y1='{margin_top+plot_h}' x2='{margin_left+plot_w}' y2='{margin_top+plot_h}' />")

    colors = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#8c564b", "#e377c2"]
    for idx, curve in enumerate(curves):
        color = colors[idx % len(colors)]
        pts = []
        base = curve.values[0] if normalize else 1.0
        for d, v in zip(curve.dates, curve.values):
            pts.append(f"{x_pos(d.timestamp()):.2f},{y_pos(v/base):.2f}")
        path_data = " L ".join(pts)
        lines.append(f"<path d='M {path_data}' fill='none' stroke='{color}' stroke-width='1.5' />")
        lines.append(f"<text x='{margin_left + 10}' y='{margin_top + 18 + 18*idx}' fill='{color}'>{curve.name}</text>")

    # Time labels (5 ticks)
    for i in range(6):
        ts = min_ts + i * ts_span / 5
        x = x_pos(ts)
        dt = datetime.fromtimestamp(ts)
        label = dt.strftime("%Y-%m")
        y_label = margin_top + plot_h + 28
        lines.append(f"<text x='{x:.2f}' y='{y_label}' transform='rotate(-25 {x:.2f},{y_label})' text-anchor='middle'>{label}</text>")

    out_path.write_text("\n".join(lines + [_svg_footer()]), encoding="utf-8")


def svg_metrics(stats: List[dict], out_path: Path):
    width, height = 1100, 660
    margin_left, margin_right, margin_top, margin_bottom = 140, 40, 60, 60
    plot_w = width - margin_left - margin_right
    rows = [
        ("Annual return (%)", [s["annual_return"] * 100 for s in stats]),
        ("Sharpe", [s["sharpe"] for s in stats]),
        ("Max drawdown (%)", [s["max_drawdown"] * 100 for s in stats]),
    ]

    labels = [s["name"] for s in stats]
    row_h = (height - margin_top - margin_bottom) / len(rows)

    lines = _svg_header(width, height)

    for row_idx, (row_name, values) in enumerate(rows):
        y_top = margin_top + row_idx * row_h
        y_bottom = y_top + row_h * 0.8
        center_y = (y_top + y_bottom) / 2
        max_abs = max(abs(v) for v in values) if values else 1.0
        if max_abs == 0:
            max_abs = 1.0
        scale = (plot_w / 2) / max_abs
        zero_x = margin_left + plot_w / 2

        # Axis line and labels
        lines.append(f"<line class='axis' x1='{zero_x:.2f}' y1='{y_top:.2f}' x2='{zero_x:.2f}' y2='{y_bottom:.2f}' />")
        lines.append(f"<text x='{margin_left-10}' y='{center_y:.2f}' text-anchor='end'>{row_name}</text>")

        for i, (label, val) in enumerate(zip(labels, values)):
            bar_y = y_top + (i + 1) * (row_h * 0.8) / (len(labels) + 1)
            bar_len = val * scale
            x_start = zero_x if val >= 0 else zero_x + bar_len
            width_bar = abs(bar_len)
            color = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd"][i % 4]
            lines.append(
                f"<rect x='{x_start:.2f}' y='{bar_y - 8:.2f}' width='{width_bar:.2f}' height='16' fill='{color}' opacity='0.85' />"
            )
            lines.append(f"<text x='{x_start + width_bar + 6:.2f}' y='{bar_y + 4:.2f}'>{val:.2f}</text>")
            lines.append(f"<text x='{margin_left - 10}' y='{bar_y + 4:.2f}' text-anchor='end' fill='{color}'>{label}</text>")

    out_path.write_text("\n".join(lines + [_svg_footer()]), encoding="utf-8")


def png_equity(curves: List[Curve], out_path: Path, title: str):
    """Generate PNG using matplotlib if available; otherwise warn and skip."""
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception as exc:  # pragma: no cover - optional dependency
        print(f"[WARN] PNG generation skipped for {out_path.name}: matplotlib not available ({exc})")
        return

    plt.figure(figsize=(10, 5))
    for curve in curves:
        base = curve.values[0]
        norm = [v / base for v in curve.values]
        plt.plot(curve.dates, norm, label=curve.name, linewidth=1.2)
        plt.xlabel("Time")
    plt.ylabel("Equity (normalized)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout(rect=(0, 0, 1, 0.95))
    plt.savefig(out_path, dpi=200)
    plt.close()


def png_metrics(stats: List[dict], out_path: Path):
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception as exc:  # pragma: no cover - optional dependency
        print(f"[WARN] PNG generation skipped for {out_path.name}: matplotlib not available ({exc})")
        return

    labels = [s["name"] for s in stats]
    ann_ret = [s["annual_return"] * 100 for s in stats]
    sharpe = [s["sharpe"] for s in stats]
    max_dd = [s["max_drawdown"] * 100 for s in stats]

    x = range(len(labels))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    ax1.bar(x, ann_ret, color="#1f77b4", alpha=0.8)
    ax1.axhline(0, color="black", linewidth=0.8)
    ax1.set_ylabel("Annual return (%)")
    ax1.grid(True, axis="y", alpha=0.3)

    ax2.bar(x, sharpe, color="#2ca02c", alpha=0.8, label="Sharpe")
    ax2.bar(x, max_dd, color="#d62728", alpha=0.4, label="Max drawdown (%)")
    ax2.axhline(0, color="black", linewidth=0.8)
    ax2.set_ylabel("Sharpe / Drawdown (%)")
    ax2.set_xticks(list(x))
    ax2.set_xticklabels(labels, rotation=30)
    ax2.legend()
    ax2.grid(True, axis="y", alpha=0.3)

    plt.tight_layout(rect=(0, 0, 1, 0.95))
    plt.savefig(out_path, dpi=200)
    plt.close(fig)


def main():
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    # RL methods
    method_files = [
        ("RL-Base", ROOT / "RL-Base" / "ppo_oos_account_value.csv"),
        ("RL-LSTM", ROOT / "RL-LSTM" / "ppo_oos_account_value.csv"),
        ("RL-Sharpe", ROOT / "RL-Sharpe" / "ppo_oos_account_value.csv"),
    ]

    curves = []
    stats = []
    for name, path in method_files:
        curve = load_curve(path, name)
        curves.append(curve)
        stats.append(compute_stats(curve))

    # Benchmarks from compare outputs
    compare_path = ROOT / "RL_model" / "compare" / "all_strategies_equity.csv"
    bench_map = {
        "Buy_Hold_Equal": "B&H Equal",
        "Hold_VOO": "Hold VOO",
    }
    benchmark_curves: List[Curve] = []
    if compare_path.exists():
        for col, label in bench_map.items():
            benchmark_curves.append(
                load_curve(compare_path, label, value_col=col, return_col=None, date_col="DateTime")
            )
            stats.append(compute_stats(benchmark_curves[-1]))
    else:
        print(f"[WARN] Benchmark file not found: {compare_path}")

    # SVG outputs
    svg_line_chart(curves, PLOT_DIR / "equity_methods.svg", "OOS Equity: RL Methods", normalize=True)
    svg_line_chart(curves + benchmark_curves, PLOT_DIR / "equity_vs_benchmark.svg", "Equity vs Benchmarks", normalize=True)
    svg_metrics(stats, PLOT_DIR / "metrics_bar.svg")

    # PNG outputs (requires matplotlib)
    png_equity(curves, PLOT_DIR / "equity_methods.png", "OOS Equity: RL Methods")
    png_equity(curves + benchmark_curves, PLOT_DIR / "equity_vs_benchmark.png", "Equity vs Benchmarks")
    png_metrics(stats, PLOT_DIR / "metrics_bar.png")

    print(f"Saved plots to {PLOT_DIR}")


if __name__ == "__main__":
    main()