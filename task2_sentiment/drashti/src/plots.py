"""SVG plots for review length and class counts. No plotting package is required."""

from __future__ import annotations

from pathlib import Path


def write_length_histogram(path: Path, values: list[int], title: str, x_label: str) -> dict[str, int]:
    """Draw a histogram that keeps a readable body and an overflow bin through the maximum."""
    if len(values) < 1:
        raise ValueError("A histogram needs at least one length.")
    ordered = sorted(values)
    minimum = ordered[0]
    maximum = ordered[-1]
    rank = max(0, min(len(ordered) - 1, int(round(0.99 * (len(ordered) - 1)))))
    body_end = ordered[rank]
    bin_count = 30
    if body_end <= minimum:
        body_end = minimum
        counts = [len(ordered)]
        labels = [f"{minimum}"]
    else:
        width = (body_end - minimum) / bin_count
        counts = [0] * bin_count
        overflow = 0
        for value in ordered:
            if value > body_end:
                overflow += 1
                continue
            index = min(bin_count - 1, int((value - minimum) / width))
            counts[index] += 1
        counts.append(overflow)
        labels = []
        for index in range(bin_count):
            start = minimum + index * width
            labels.append(f"{start:.0f}")
        labels.append(f">{body_end} to {maximum}")
    _write_bars(path, counts, labels, title, x_label, f"full range {minimum} to {maximum}")
    return {"minimum": minimum, "maximum": maximum, "body_end": body_end, "bins": len(counts)}


def write_class_bars(path: Path, series: list[tuple[str, list[int]]], class_names: list[str], title: str) -> None:
    """Draw grouped class counts. Each series is one split."""
    if not series:
        raise ValueError("Class plot needs at least one split.")
    labels = []
    values = []
    for name, counts in series:
        for class_name, count in zip(class_names, counts):
            labels.append(f"{name} {class_name}")
            values.append(count)
    _write_bars(path, values, labels, title, "class", "counts from loaded records")


def _write_bars(path: Path, values: list[int], labels: list[str], title: str, x_label: str, note: str) -> None:
    width, height = 960, 540
    left, right, top, bottom = 70, 20, 40, 110
    plot_width = width - left - right
    plot_height = height - top - bottom
    peak = max(values) if values else 1
    if peak == 0:
        peak = 1
    bar_width = plot_width / max(len(values), 1)
    bars: list[str] = []
    texts: list[str] = []
    for index, value in enumerate(values):
        bar_height = value / peak * plot_height
        x = left + index * bar_width
        y = top + plot_height - bar_height
        bars.append(
            f'<rect x="{x + 1:.2f}" y="{y:.2f}" width="{max(bar_width - 2, 1):.2f}" height="{bar_height:.2f}" fill="#1f4e79"/>'
        )
        if len(values) <= 12:
            texts.append(
                f'<text x="{x + bar_width / 2:.2f}" y="{height - 70}" font-size="11" text-anchor="end" transform="rotate(-40 {x + bar_width / 2:.2f} {height - 70})">{_xml(labels[index])}</text>'
            )
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="#ffffff"/>
<text x="{left}" y="24" font-size="16" fill="#222222">{_xml(title)}</text>
<text x="{left}" y="{height - 16}" font-size="12" fill="#444444">{_xml(x_label)}. {_xml(note)}</text>
<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_height}" stroke="#888888"/>
<line x1="{left}" y1="{top + plot_height}" x2="{left + plot_width}" y2="{top + plot_height}" stroke="#888888"/>
<text x="{left - 8}" y="{top + 4}" font-size="12" text-anchor="end" fill="#444444">{peak}</text>
{''.join(bars)}
{''.join(texts)}
</svg>
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(svg, encoding="utf-8")


def _xml(value: str) -> str:
    return value.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
