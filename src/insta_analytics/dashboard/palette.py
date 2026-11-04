"""Light and dark colour palettes plus the shared Plotly layout for the charts."""

from __future__ import annotations

from ..transform.features import FORMATS
from ..transform.labels import THEMES

Mode = str  # "light" | "dark"

# --- Categorical hues (8 slots; fixed order is the CVD-safety mechanism) -----
CATEGORICAL: dict[Mode, list[str]] = {
    "light": ["#2a78d6", "#1baf7a", "#eda100", "#008300",
              "#4a3aa7", "#e34948", "#e87ba4", "#eb6834"],
    "dark":  ["#3987e5", "#199e70", "#c98500", "#008300",
              "#9085e9", "#e66767", "#d55181", "#d95926"],
}

# --- Sequential blue ramp (magnitude: choropleth) light→dark -----------------
SEQUENTIAL: dict[Mode, list[str]] = {
    "light": ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"],
    "dark":  ["#0d366b", "#184f95", "#256abf", "#3987e5", "#6da7ec", "#9ec5f4", "#cde2fb"],
}

# --- Diverging blue↔red with a neutral-gray midpoint (polarity: ratio vs 1) --
DIVERGING: dict[Mode, list[tuple[float, str]]] = {
    # (position, hex) for a Plotly colorscale; midpoint = "nothing" (gray).
    "light": [(0.0, "#184f95"), (0.5, "#f0efec"), (1.0, "#b0201f")],
    "dark":  [(0.0, "#6da7ec"), (0.5, "#383835"), (1.0, "#e66767")],
}

# --- Chart chrome & ink ------------------------------------------------------
INK: dict[Mode, dict[str, str]] = {
    "light": {
        "surface": "#fcfcfb", "paper": "#f9f9f7",
        "primary": "#0b0b0b", "secondary": "#52514e", "muted": "#898781",
        "grid": "#e1e0d9", "baseline": "#c3c2b7",
    },
    "dark": {
        "surface": "#1a1a19", "paper": "#0d0d0d",
        "primary": "#ffffff", "secondary": "#c3c2b7", "muted": "#898781",
        "grid": "#2c2c2a", "baseline": "#383835",
    },
}

FONT_FAMILY = 'system-ui, -apple-system, "Segoe UI", sans-serif'


def _entity_colors(order: tuple[str, ...]) -> dict[Mode, dict[str, str]]:
    """Map each entity to a fixed categorical slot, per mode (color follows entity)."""
    return {mode: {name: CATEGORICAL[mode][i] for i, name in enumerate(order)}
            for mode in ("light", "dark")}


# Stable entity→color maps so a series keeps its hue across every panel.
THEME_COLORS = _entity_colors(THEMES)      # 7 themes → slots 0..6
FORMAT_COLORS = _entity_colors(FORMATS)    # Reel/Video/Carousel/Image → slots 0..3


def theme_colors(mode: Mode = "light") -> dict[str, str]:
    """Theme → hex for ``mode`` (taxonomy order → fixed categorical slots)."""
    return THEME_COLORS[mode]


def format_colors(mode: Mode = "light") -> dict[str, str]:
    """Format → hex for ``mode`` (Reel/Video/Carousel/Image → fixed slots)."""
    return FORMAT_COLORS[mode]


def series(mode: Mode = "light", i: int = 0) -> str:
    """Categorical slot ``i`` for a single-series chart with no entity identity."""
    return CATEGORICAL[mode][i]


def layout(mode: Mode = "light", **overrides) -> dict:
    """Plotly ``layout`` defaults for ``mode``, shallow-merged with ``overrides``."""
    ink = INK[mode]
    base = dict(
        paper_bgcolor=ink["surface"],
        plot_bgcolor=ink["surface"],
        font=dict(family=FONT_FAMILY, color=ink["primary"], size=13),
        title=dict(font=dict(size=16, color=ink["primary"]), x=0.0, xanchor="left"),
        xaxis=dict(gridcolor=ink["grid"], zerolinecolor=ink["baseline"],
                   linecolor=ink["baseline"], tickfont=dict(color=ink["muted"]),
                   automargin=True),
        yaxis=dict(gridcolor=ink["grid"], zerolinecolor=ink["baseline"],
                   linecolor=ink["baseline"], tickfont=dict(color=ink["muted"]),
                   automargin=True),
        legend=dict(font=dict(color=ink["secondary"])),
        hoverlabel=dict(font=dict(family=FONT_FAMILY)),
        margin=dict(l=60, r=24, t=48, b=48),
        colorway=CATEGORICAL[mode],
    )
    base.update(overrides)
    return base
