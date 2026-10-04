"""SVG curves drawn from measured series. No extra plotting package is required."""

from __future__ import annotations

from pathlib import Path


def write_series_svg(
    path: Path,
    series: list[tuple[str, list[float]]],
    title: str,
    y_label: str,
    marker_fraction: float | None = None,
    marker_label: str | None = None,
) -> None:
    """Write one SVG with one polyline per named series."""
    if not series or any(len(values) < 2 for _name, values in series):
        raise ValueError("Each plotted series needs at least two measured points.")
    width, height = 960, 540
    left, right, top, bottom = 70, 20, 40, 50
    plot_width = width - left - right
    plot_height = height - top - bottom
    flattened = [value for _name, values in series for value in values]
    low = min(flattened)
    high = max(flattened)
    if high == low:
        high = low + 1.0
    colors = ("#1f4e79", "#b85c38", "#2f6f4e")
    marker = ""
    if marker_fraction is not None:
        if isinstance(marker_fraction, bool) or not isinstance(marker_fraction, (int, float)):
            raise ValueError("marker_fraction must be a number from 0 to 1.")
        if marker_fraction < 0 or marker_fraction > 1:
            raise ValueError("marker_fraction must be from 0 to 1.")
        marker_x = left + float(marker_fraction) * plot_width
        label = _xml(marker_label or "marker")
        marker = (
            f'<line x1="{marker_x:.2f}" y1="{top}" x2="{marker_x:.2f}" y2="{top + plot_height}" '
            f'stroke="#666666" stroke-dasharray="4 4"/>'
            f'<text x="{marker_x + 4:.2f}" y="{top + 16}" font-size="12" fill="#666666">{label}</text>'
        )
    polylines: list[str] = []
    legend: list[str] = []
    longest = max(len(values) for _name, values in series)
    for index, (name, values) in enumerate(series):
        points: list[str] = []
        for step, value in enumerate(values):
            x_ratio = 0.0 if len(values) == 1 else step / (len(values) - 1)
            x = left + x_ratio * plot_width
            y = top + (high - value) / (high - low) * plot_height
            points.append(f"{x:.2f},{y:.2f}")
        color = colors[index % len(colors)]
        polylines.append(f'<polyline fill="none" stroke="{color}" stroke-width="2" points="{" ".join(points)}"/>')
        legend.append(
            f'<text x="{left + index * 220}" y="{height - 16}" fill="{color}" font-size="14">{_xml(name)}</text>'
        )
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="#ffffff"/>
<text x="{left}" y="24" font-size="16" fill="#222222">{_xml(title)}</text>
<text x="16" y="{top + 12}" font-size="12" fill="#444444">{_xml(y_label)}</text>
<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_height}" stroke="#888888"/>
<line x1="{left}" y1="{top + plot_height}" x2="{left + plot_width}" y2="{top + plot_height}" stroke="#888888"/>
<text x="{left}" y="{top + plot_height + 32}" font-size="12" fill="#444444">progress across {longest} measured points</text>
<text x="{left - 8}" y="{top + 4}" font-size="12" text-anchor="end" fill="#444444">{high:.4g}</text>
<text x="{left - 8}" y="{top + plot_height}" font-size="12" text-anchor="end" fill="#444444">{low:.4g}</text>
{marker}
{''.join(polylines)}
{''.join(legend)}
</svg>
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(svg, encoding="utf-8")


def _xml(value: str) -> str:
    return value.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
