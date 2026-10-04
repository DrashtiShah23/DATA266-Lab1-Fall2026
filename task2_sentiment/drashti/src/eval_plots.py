"""SVG figures for the Task 2 evaluation. No plotting package is required."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def _xml(value: str) -> str:
    return value.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _thin(values: np.ndarray, limit: int = 400) -> np.ndarray:
    if values.shape[0] <= limit:
        return np.arange(values.shape[0])
    return np.unique(np.linspace(0, values.shape[0] - 1, limit).astype(int))


def write_confusion_svg(path: Path, counts: dict[str, int], title: str) -> None:
    """Draw true-label rows and predicted-label columns."""
    cells = [
        ("True negative", "Predicted negative", counts["true_negative"], 0, 0),
        ("True negative", "Predicted positive", counts["false_positive"], 1, 0),
        ("True positive", "Predicted negative", counts["false_negative"], 0, 1),
        ("True positive", "Predicted positive", counts["true_positive"], 1, 1),
    ]
    peak = max(item[2] for item in cells) or 1
    parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="640" height="460" viewBox="0 0 640 460">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="40" y="28" font-size="16" fill="#222222">{_xml(title)}</text>',
        '<text x="250" y="58" font-size="13" fill="#333333">Predicted label</text>',
        '<text x="150" y="82" font-size="12" fill="#333333">negative</text>',
        '<text x="360" y="82" font-size="12" fill="#333333">positive</text>',
        '<text x="24" y="210" font-size="13" fill="#333333" transform="rotate(-90 24 210)">True label</text>',
    ]
    for _true_name, _pred_name, value, column, row in cells:
        shade = 255 - int(220 * value / peak)
        color = f"rgb({shade},{shade},255)"
        x = 120 + column * 200
        y = 100 + row * 140
        parts.append(f'<rect x="{x}" y="{y}" width="180" height="120" fill="{color}" stroke="#1f4e79"/>')
        parts.append(f'<text x="{x + 90}" y="{y + 68}" font-size="22" text-anchor="middle" fill="#111111">{value}</text>')
    parts.append('<text x="40" y="250" font-size="12" fill="#333333">negative</text>')
    parts.append('<text x="40" y="390" font-size="12" fill="#333333">positive</text>')
    parts.append('<text x="40" y="440" font-size="12" fill="#444444">Rows are true labels. Columns are predicted labels.</text>')
    parts.append("</svg>")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def write_curve_svg(
    path: Path,
    series: list[tuple[str, np.ndarray, np.ndarray, str]],
    title: str,
    x_label: str,
    y_label: str,
    diagonal: bool,
) -> None:
    """Draw one or more curves on the unit square."""
    width, height = 640, 480
    left, top = 60, 40
    plot = 380
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{left}" y="24" font-size="16" fill="#222222">{_xml(title)}</text>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot}" stroke="#888888"/>',
        f'<line x1="{left}" y1="{top + plot}" x2="{left + plot}" y2="{top + plot}" stroke="#888888"/>',
        f'<text x="{left + plot / 2}" y="{top + plot + 36}" font-size="12" text-anchor="middle" fill="#333333">{_xml(x_label)}</text>',
        f'<text x="18" y="{top + plot / 2}" font-size="12" fill="#333333" transform="rotate(-90 18 {top + plot / 2})">{_xml(y_label)}</text>',
    ]
    if diagonal:
        parts.append(
            f'<line x1="{left}" y1="{top + plot}" x2="{left + plot}" y2="{top}" stroke="#bbbbbb" stroke-dasharray="4 4"/>'
        )
    legend_y = 70
    for name, x_values, y_values, color in series:
        index = _thin(x_values)
        points = []
        for position in index:
            x = left + float(x_values[position]) * plot
            y = top + plot - float(y_values[position]) * plot
            points.append(f"{x:.2f},{y:.2f}")
        parts.append(f'<polyline fill="none" stroke="{color}" stroke-width="2" points="{" ".join(points)}"/>')
        parts.append(f'<rect x="{left + plot + 20}" y="{legend_y}" width="12" height="12" fill="{color}"/>')
        parts.append(
            f'<text x="{left + plot + 38}" y="{legend_y + 11}" font-size="12" fill="#222222">{_xml(name)}</text>'
        )
        legend_y += 22
    parts.append("</svg>")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def write_reliability_svg(path: Path, bins: list[dict], title: str) -> None:
    """Plot empirical accuracy against mean confidence."""
    width, height = 640, 480
    left, top, plot = 60, 40, 380
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{left}" y="24" font-size="16" fill="#222222">{_xml(title)}</text>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot}" stroke="#888888"/>',
        f'<line x1="{left}" y1="{top + plot}" x2="{left + plot}" y2="{top + plot}" stroke="#888888"/>',
        f'<line x1="{left}" y1="{top + plot}" x2="{left + plot}" y2="{top}" stroke="#bbbbbb" stroke-dasharray="4 4"/>',
        '<text x="250" y="450" font-size="12" text-anchor="middle" fill="#333333">Mean confidence of the predicted class</text>',
        '<text x="18" y="230" font-size="12" fill="#333333" transform="rotate(-90 18 230)">Empirical accuracy</text>',
    ]
    peak = max((row["count"] for row in bins), default=1) or 1
    for row in bins:
        if row["count"] == 0:
            continue
        x = left + float(row["mean_confidence"]) * plot
        y = top + plot - float(row["empirical_accuracy"]) * plot
        radius = 4 + 10 * (row["count"] / peak) ** 0.5
        parts.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="{radius:.2f}" fill="#1f4e79" fill-opacity="0.75"/>')
    parts.append("</svg>")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")
