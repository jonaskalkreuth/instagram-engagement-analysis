"""Chart builders: one ``DataFrame -> plotly.Figure`` function per dashboard panel."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from . import palette

Mode = str  # "light" | "dark"


def _diverging_scale(mode: Mode) -> list[list]:
    """Plotly colorscale (list of ``[pos, hex]``) for the diverging ratio maps."""
    return [[pos, hex_] for pos, hex_ in palette.DIVERGING[mode]]


def _sequential_scale(mode: Mode) -> list[list]:
    ramp = palette.SEQUENTIAL[mode]
    return [[i / (len(ramp) - 1), hex_] for i, hex_ in enumerate(ramp)]


def _empty_fig(mode: Mode, msg: str) -> go.Figure:
    """A styled placeholder when a panel has no data (e.g. demographics un-run)."""
    fig = go.Figure()
    fig.update_layout(**palette.layout(mode))
    fig.add_annotation(text=msg, showarrow=False,
                       font=dict(color=palette.INK[mode]["muted"], size=14),
                       xref="paper", yref="paper", x=0.5, y=0.5)
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return fig


def _ci_arrays(lo: pd.Series, val: pd.Series, hi: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """Asymmetric error arrays (plus/minus), NaN CIs (small n) → 0-width whisker."""
    plus = np.nan_to_num((hi - val).to_numpy(dtype=float), nan=0.0)
    minus = np.nan_to_num((val - lo).to_numpy(dtype=float), nan=0.0)
    return plus, minus


def _hbar_ci(
    df: pd.DataFrame,
    by: str,
    value: str,
    lo: str,
    hi: str,
    *,
    mode: Mode,
    title: str,
    ratio: bool,
    colors: dict[str, str] | None,
    xtitle: str,
) -> go.Figure:
    """Shared horizontal-bar-with-CI engine for effect (ratio) and rate charts."""
    d = df.copy()
    labels = d[by].astype(str)
    plus, minus = _ci_arrays(d[lo], d[value], d[hi])

    if colors is not None:
        bar_colors = [colors.get(name, palette.series(mode)) for name in labels]
    else:
        bar_colors = palette.series(mode)

    # Emphasis: bars whose CI clears the reference (1.0 for ratios) read solid;
    # inconclusive bars are dimmed. Rates (no reference) all read solid.
    if ratio:
        clears = ((d[lo] > 1.0) | (d[hi] < 1.0)).fillna(False)
        opacity = np.where(clears, 1.0, 0.45)
    else:
        opacity = 1.0

    suffix = "×" if ratio else ""
    text = [f"{v:.2f}{suffix}" if pd.notna(v) else "" for v in d[value]]

    fig = go.Figure(
        go.Bar(
            x=d[value], y=labels, orientation="h",
            marker=dict(color=bar_colors, opacity=opacity,
                        line=dict(width=1, color=palette.INK[mode]["surface"])),
            error_x=dict(type="data", symmetric=False, array=plus, arrayminus=minus,
                         color=palette.INK[mode]["secondary"], thickness=1.5, width=4),
            text=text, textposition="outside",
            textfont=dict(color=palette.INK[mode]["secondary"]),
            customdata=d["n"],
            hovertemplate=f"%{{y}}<br>{xtitle}: %{{x:.2f}}{suffix}<br>n=%{{customdata}}<extra></extra>",
        )
    )
    fig.update_layout(**palette.layout(mode, title=title, showlegend=False,
                                       bargap=0.35))
    fig.update_yaxes(autorange="reversed")  # highest value on top
    fig.update_xaxes(title_text=xtitle)
    if ratio:
        fig.add_vline(x=1.0, line=dict(color=palette.INK[mode]["baseline"],
                                       dash="dash", width=1))
    return fig


def effect_bars(df: pd.DataFrame, by: str, *, mode: Mode = "light",
                title: str = "", colors: dict[str, str] | None = None) -> go.Figure:
    """Horizontal bars of an :func:`eda.group_effect` ratio with CIs and a 1.0 line."""
    return _hbar_ci(df, by, "ratio", "ratio_lo", "ratio_hi", mode=mode, title=title,
                    ratio=True, colors=colors, xtitle="ratio vs era baseline")


def rate_bars(df: pd.DataFrame, by: str, *, mode: Mode = "light",
              title: str = "", colors: dict[str, str] | None = None) -> go.Figure:
    """Horizontal bars of an :func:`eda.group_median` rate with CIs."""
    return _hbar_ci(df, by, "median", "median_lo", "median_hi", mode=mode, title=title,
                    ratio=False, colors=colors, xtitle="median rate")


def spearman_bars(df: pd.DataFrame, *, mode: Mode = "light",
                  title: str = "Continuous predictors (Spearman ρ)") -> go.Figure:
    """Horizontal ρ bars (with bootstrap CI) centered at 0 — monotone association."""
    d = df.copy()
    plus, minus = _ci_arrays(d["rho_lo"], d["rho"], d["rho_hi"])
    text = [f"{v:+.2f}" for v in d["rho"]]
    fig = go.Figure(
        go.Bar(
            x=d["rho"], y=d["predictor"].astype(str), orientation="h",
            marker=dict(color=palette.series(mode),
                        line=dict(width=1, color=palette.INK[mode]["surface"])),
            error_x=dict(type="data", symmetric=False, array=plus, arrayminus=minus,
                         color=palette.INK[mode]["secondary"], thickness=1.5, width=4),
            text=text, textposition="outside",
            textfont=dict(color=palette.INK[mode]["secondary"]),
            customdata=d["n"],
            hovertemplate="%{y}<br>ρ=%{x:+.2f}<br>n=%{customdata}<extra></extra>",
        )
    )
    fig.update_layout(**palette.layout(mode, title=title, showlegend=False, bargap=0.35))
    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(title_text="Spearman ρ vs rel. engagement")
    fig.add_vline(x=0.0, line=dict(color=palette.INK[mode]["baseline"], dash="dash", width=1))
    return fig


def seasonality_bars(df: pd.DataFrame, *, mode: Mode = "light",
                     title: str = "Seasonality (ratio vs era)") -> go.Figure:
    """Vertical month bars of median ratio (1.0 = era baseline) with CI whiskers."""
    d = df.copy()
    plus, minus = _ci_arrays(d["ratio_lo"], d["ratio"], d["ratio_hi"])
    fig = go.Figure(
        go.Bar(
            x=d["month_name"], y=d["ratio"],
            marker=dict(color=palette.series(mode),
                        line=dict(width=1, color=palette.INK[mode]["surface"])),
            error_y=dict(type="data", symmetric=False, array=plus, arrayminus=minus,
                         color=palette.INK[mode]["secondary"], thickness=1.5, width=4),
            customdata=d["n"],
            hovertemplate="%{x}<br>%{y:.2f}×<br>n=%{customdata}<extra></extra>",
        )
    )
    fig.update_layout(**palette.layout(mode, title=title, showlegend=False, bargap=0.3))
    fig.update_yaxes(title_text="ratio vs era baseline")
    fig.add_hline(y=1.0, line=dict(color=palette.INK[mode]["baseline"], dash="dash", width=1))
    return fig


_DOW_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def timing_heatmap(pivot: pd.DataFrame, *, mode: Mode = "light",
                   title: str = "When posts beat their era (day × hour)") -> go.Figure:
    """Day × hour ratio heatmap; colour is ``log2(ratio)`` so 0.5× and 2× look equally far from 1×."""
    ratio = pivot.to_numpy(dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        z = np.log2(ratio)
    y = [_DOW_NAMES[i] if 0 <= i < 7 else str(i) for i in pivot.index]
    ticks = [-1.0, -0.5, 0.0, 0.5, 1.0]
    fig = go.Figure(
        go.Heatmap(
            # Clamp colour to 0.5×–2×: many day×hour cells hold a single post, so
            # an unguarded ratio can spike to ~10× and wash out the scale.
            z=z, customdata=ratio, x=[str(c) for c in pivot.columns], y=y,
            colorscale=_diverging_scale(mode), zmin=-1.0, zmax=1.0, zmid=0.0,
            colorbar=dict(
                title="ratio", tickvals=ticks, ticktext=[f"{2**t:.2g}×" for t in ticks],
                tickfont=dict(color=palette.INK[mode]["muted"]),
            ),
            hovertemplate="%{y} %{x}:00<br>%{customdata:.2f}×<extra></extra>",
            xgap=1, ygap=1,
        )
    )
    fig.update_layout(**palette.layout(mode, title=title))
    fig.update_xaxes(title_text="hour of day", type="category")
    fig.update_yaxes(autorange="reversed")
    return fig


def cell_heatmap(ratio: pd.DataFrame, n: pd.DataFrame, flags: pd.DataFrame,
                 *, mode: Mode = "light",
                 title: str = "Theme × format (ratio vs era)") -> go.Figure:
    """Theme × format ratio heatmap; ★ marks cells beating both marginals, NaN cells stay blank."""
    z = ratio.to_numpy(dtype=float)
    xs = [str(c) for c in ratio.columns]
    ys = [str(r) for r in ratio.index]
    fig = go.Figure(
        go.Heatmap(
            z=z, x=xs, y=ys, colorscale=_diverging_scale(mode), zmid=1.0,
            colorbar=dict(title="ratio", tickfont=dict(color=palette.INK[mode]["muted"])),
            hovertemplate="%{y} · %{x}<br>%{z:.2f}×<extra></extra>",
            xgap=2, ygap=2,
        )
    )
    zmax = np.nanmax(z) if np.isfinite(z).any() else 1.0
    zmin = np.nanmin(z) if np.isfinite(z).any() else 1.0
    mid = (zmax + zmin) / 2
    for i, r in enumerate(ratio.index):
        for j, c in enumerate(ratio.columns):
            v = ratio.iat[i, j]
            if pd.isna(v):
                continue
            star = "★" if bool(flags.iat[i, j]) else ""
            # Flip annotation ink with cell brightness for legibility.
            txt_color = "#ffffff" if (v > mid) else palette.INK[mode]["primary"]
            fig.add_annotation(
                x=xs[j], y=ys[i], showarrow=False,
                text=f"{v:.2f}{star}<br><span style='font-size:10px'>n={int(n.iat[i, j])}</span>",
                font=dict(color=txt_color, size=12),
            )
    fig.update_layout(**palette.layout(mode, title=title))
    fig.update_xaxes(title_text="format", type="category")
    fig.update_yaxes(autorange="reversed", type="category")
    return fig


def theme_mix_area(mix: pd.DataFrame, *, mode: Mode = "light",
                   title: str = "Theme mix per year (share of posts)") -> go.Figure:
    """Stacked area of each theme's posting share over the years (fixed hues)."""
    colors = palette.theme_colors(mode)
    fig = go.Figure()
    for theme in mix.columns:
        fig.add_trace(go.Scatter(
            x=mix.index.astype(int), y=mix[theme], name=str(theme),
            mode="lines", stackgroup="mix",
            line=dict(width=0.5, color=palette.INK[mode]["surface"]),
            fillcolor=colors.get(str(theme), palette.series(mode)),
            hovertemplate=f"{theme} %{{y:.0%}} (%{{x}})<extra></extra>",
        ))
    fig.update_layout(**palette.layout(mode, title=title))
    fig.update_yaxes(title_text="share of posts", tickformat=".0%", range=[0, 1])
    fig.update_xaxes(title_text="year", dtick=1)
    return fig


def cadence_line(cadence: pd.DataFrame, *, mode: Mode = "light",
                 title: str = "Posts per year") -> go.Figure:
    """Single line of posting cadence (posts per year)."""
    fig = go.Figure(
        go.Scatter(
            x=cadence["year"].astype(int), y=cadence["posts"],
            mode="lines+markers",
            line=dict(color=palette.series(mode), width=2),
            marker=dict(size=8, color=palette.series(mode)),
            hovertemplate="%{x}: %{y} posts<extra></extra>",
        )
    )
    fig.update_layout(**palette.layout(mode, title=title, showlegend=False))
    fig.update_yaxes(title_text="posts", rangemode="tozero")
    fig.update_xaxes(title_text="year", dtick=1)
    return fig


def theme_era_bars(era: pd.DataFrame, *, mode: Mode = "light",
                   title: str = "Early vs late era (ratio vs baseline)") -> go.Figure:
    """Grouped early/late effect bars per theme (2 series, legend) + 1.0 line."""
    d = era.copy()
    cat = palette.CATEGORICAL[mode]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=d["theme"].astype(str), y=d["early_ratio"], name="early",
                         marker_color=cat[0],
                         hovertemplate="%{x} early: %{y:.2f}×<extra></extra>"))
    fig.add_trace(go.Bar(x=d["theme"].astype(str), y=d["late_ratio"], name="late",
                         marker_color=cat[2],
                         hovertemplate="%{x} late: %{y:.2f}×<extra></extra>"))
    fig.update_layout(**palette.layout(mode, title=title, barmode="group", bargap=0.3))
    fig.update_yaxes(title_text="ratio vs era baseline")
    fig.add_hline(y=1.0, line=dict(color=palette.INK[mode]["baseline"], dash="dash", width=1))
    return fig


def _share_bars(df: pd.DataFrame, *, mode: Mode, title: str, orientation: str) -> go.Figure:
    """Share bars (% of a breakdown) — vertical for age/gender, horizontal for geo."""
    if df.empty:
        return _empty_fig(mode, "No demographics yet — run collect.demographics")
    labels = df["dimension"].astype(str)
    text = [f"{s:.0%}" for s in df["share"]]
    if orientation == "h":
        bar = go.Bar(x=df["share"], y=labels, orientation="h",
                     text=text, textposition="outside",
                     hovertemplate="%{y}: %{x:.1%}<extra></extra>")
    else:
        bar = go.Bar(x=labels, y=df["share"], text=text, textposition="outside",
                     hovertemplate="%{x}: %{y:.1%}<extra></extra>")
    bar.marker = dict(color=palette.series(mode),
                      line=dict(width=1, color=palette.INK[mode]["surface"]))
    bar.textfont = dict(color=palette.INK[mode]["secondary"])
    fig = go.Figure(bar)
    fig.update_layout(**palette.layout(mode, title=title, showlegend=False, bargap=0.3))
    if orientation == "h":
        fig.update_xaxes(tickformat=".0%")
        fig.update_yaxes(autorange="reversed")
    else:
        fig.update_yaxes(tickformat=".0%")
    return fig


def gender_bars(df: pd.DataFrame, *, mode: Mode = "light",
                title: str = "Audience by gender") -> go.Figure:
    """Vertical share bars, GENDER_ORDER (F/M/U)."""
    return _share_bars(df, mode=mode, title=title, orientation="v")


def age_bars(df: pd.DataFrame, *, mode: Mode = "light",
             title: str = "Audience by age") -> go.Figure:
    """Vertical share bars, AGE_ORDER buckets."""
    return _share_bars(df, mode=mode, title=title, orientation="v")


def city_bars(df: pd.DataFrame, *, mode: Mode = "light",
              title: str = "Top cities (share)") -> go.Figure:
    """Horizontal share bars — long city labels + an aggregated ``Other (k)`` row."""
    return _share_bars(df, mode=mode, title=title, orientation="h")


def country_choropleth(df: pd.DataFrame, *, mode: Mode = "light",
                       title: str = "Audience by country (share)") -> go.Figure:
    """World choropleth of follower share; rows without an ISO-3 (Other) are skipped."""
    if df.empty:
        return _empty_fig(mode, "No demographics yet — run collect.demographics")
    d = df.dropna(subset=["iso_alpha"])
    ink = palette.INK[mode]
    fig = go.Figure(
        go.Choropleth(
            locations=d["iso_alpha"], z=d["share"], text=d["country_name"],
            colorscale=_sequential_scale(mode), zmin=0,
            marker_line_color=ink["surface"], marker_line_width=0.4,
            colorbar=dict(title="share", tickformat=".0%",
                          tickfont=dict(color=ink["muted"])),
            hovertemplate="%{text}: %{z:.1%}<extra></extra>",
        )
    )
    fig.update_layout(**palette.layout(mode, title=title))
    fig.update_geos(bgcolor=ink["surface"], showframe=False,
                    showcoastlines=False, landcolor=ink["grid"],
                    lakecolor=ink["surface"], projection_type="natural earth")
    return fig
