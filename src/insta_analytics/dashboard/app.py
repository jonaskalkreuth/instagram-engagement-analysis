"""Streamlit app: renders the cached ``DashboardData`` bundle with the ``charts`` builders.

    streamlit run src/insta_analytics/dashboard/app.py

Imports are absolute because ``streamlit run`` executes this file as ``__main__``.
"""

from __future__ import annotations

import streamlit as st

from insta_analytics.dashboard import charts, data, palette

st.set_page_config(
    page_title="Instagram Musician Analytics",
    page_icon="🎧",
    layout="wide",
    initial_sidebar_state="expanded",
)


def active_mode() -> str:
    """Return "light" or "dark" for the charts.

    The configured ``theme.base`` wins, because the browser theme can disagree with it.
    """
    base = st.get_option("theme.base")
    if base in ("light", "dark"):
        return base
    try:
        t = st.context.theme.type  # Streamlit ≥1.46
        if t in ("light", "dark"):
            return t
    except Exception:
        pass
    return "dark"


MODE = active_mode()
INK = palette.INK[MODE]


st.markdown(
    f"""
    <style>
      .block-container {{ padding-top: 2.2rem; max-width: 1400px; }}
      h1, h2, h3 {{ letter-spacing: -0.01em; }}
      .kpi-row {{ display: flex; gap: 14px; flex-wrap: wrap; margin: 8px 0 4px; }}
      .kpi {{
        flex: 1 1 150px; background: {INK['surface']};
        border: 1px solid {INK['grid']}; border-radius: 14px;
        padding: 16px 18px; box-shadow: 0 1px 3px rgba(0,0,0,0.18);
      }}
      .kpi .label {{ color: {INK['muted']}; font-size: 0.72rem;
        text-transform: uppercase; letter-spacing: 0.06em; }}
      .kpi .value {{ color: {INK['primary']}; font-size: 1.65rem;
        font-weight: 650; line-height: 1.25; margin-top: 2px; }}
      .kpi .sub {{ color: {INK['secondary']}; font-size: 0.8rem; }}
      .hero-sub {{ color: {INK['secondary']}; font-size: 1.02rem; margin-top: -6px; }}
    </style>
    """,
    unsafe_allow_html=True,
)


def kpi_card(label: str, value: str, sub: str = "") -> str:
    return (f'<div class="kpi"><div class="label">{label}</div>'
            f'<div class="value">{value}</div>'
            f'<div class="sub">{sub}</div></div>')


def show(fig) -> None:
    """Render a Plotly figure without Streamlit overriding its colors."""
    st.plotly_chart(fig, use_container_width=True, theme=None)


DB_PATH = data.DEFAULT_DB_PATH
IS_SYNTHETIC = DB_PATH.parent.name == "synthetic"
SOURCE_LABEL = "synthetic (public demo)" if IS_SYNTHETIC else "real (private)"
lo_year, hi_year = data.year_bounds(DB_PATH)

with st.sidebar:
    st.markdown("### 🎧 Analytics")
    st.caption(f"Source: {SOURCE_LABEL}")
    if hi_year > lo_year:
        year_range = st.slider(
            "Post year range", min_value=lo_year, max_value=hi_year,
            value=(lo_year, hi_year), step=1,
            help="Filters the posts feeding every chart. Demographics is a current "
                 "snapshot and is unaffected.",
        )
    else:
        year_range = (lo_year, hi_year)
        st.caption(f"All posts from {lo_year}.")
    st.divider()
    st.caption(
        "Category comparisons (by theme / format) always use the full selection — "
        "filtering them out would remove the comparison itself."
    )
    st.caption("Relative / rate / aggregate metrics only — publish-safe.")

bundle = data.load(DB_PATH, year_range=year_range)
tcolors, fcolors = palette.theme_colors(MODE), palette.format_colors(MODE)


st.title("Instagram Musician Analytics")
st.markdown(
    '<p class="hero-sub">Which content is associated with engagement for a touring DJ — '
    "measured against each post's own era baseline, so growth doesn't mask the signal.</p>",
    unsafe_allow_html=True,
)
if IS_SYNTHETIC:
    st.info(
        "Demo on synthetic data, built to mirror the private real-data findings. "
        "Numbers are illustrative, not the account's real figures."
    )

k = bundle.kpis
top_theme = bundle.eng_theme.iloc[0] if not bundle.eng_theme.empty else None
top_format = bundle.eng_format.iloc[0] if not bundle.eng_format.empty else None
followers = f"{k['followers']:,}" if k.get("followers") else "—"
snap = k.get("demo_snapshot_date")
snap_str = snap.strftime("%d %b %Y") if snap is not None else "—"

cards = [
    kpi_card("Posts analysed", f"{k['n_posts']:,}", f"{year_range[0]}–{year_range[1]}"),
    kpi_card("Reach-tracked", f"{k['n_reach']:,}", "posts with reach"),
    kpi_card("Followers", followers, f"snapshot {snap_str}"),
]
if top_theme is not None:
    cards.append(kpi_card("Top theme", str(top_theme["theme"]),
                          f"{top_theme['ratio']:.2f}× vs era"))
if top_format is not None:
    cards.append(kpi_card("Top format", str(top_format["format"]),
                          f"{top_format['ratio']:.2f}× vs era"))
st.markdown('<div class="kpi-row">' + "".join(cards) + "</div>", unsafe_allow_html=True)
st.write("")


tab_eda, tab_saves, tab_tf, tab_trends, tab_aud = st.tabs(
    ["📊 Engagement", "🔖 Saves & Shares", "🎛 Theme × Format",
     "📈 Seasonality & Drift", "🌍 Audience"]
)

with tab_eda:
    st.subheader("Relative engagement by content type")
    st.caption("Median engagement ÷ each post's era baseline. Solid bars clear the "
               "bootstrap CI (signal); dimmed bars straddle 1.0 (inconclusive).")
    c1, c2 = st.columns(2)
    with c1:
        show(charts.effect_bars(bundle.eng_theme, "theme", mode=MODE,
                                title="By theme", colors=tcolors))
    with c2:
        show(charts.effect_bars(bundle.eng_format, "format", mode=MODE,
                                title="By format", colors=fcolors))
    show(charts.timing_heatmap(bundle.timing, mode=MODE))
    show(charts.spearman_bars(bundle.spearman, mode=MODE))
    with st.expander("Reach (descriptive companion)"):
        rc1, rc2 = st.columns(2)
        with rc1:
            show(charts.effect_bars(bundle.reach_theme, "theme", mode=MODE,
                                    title="Reach by theme", colors=tcolors))
        with rc2:
            show(charts.effect_bars(bundle.reach_format, "format", mode=MODE,
                                    title="Reach by format", colors=fcolors))

with tab_saves:
    st.subheader("Saves & shares")
    st.caption("Era-relative effect (ratio vs baseline) for the currency that "
               "actually matters for a musician: bookmarks and reshares.")
    for metric in ("saves", "shares"):
        st.markdown(f"**{metric.capitalize()}**")
        c1, c2 = st.columns(2)
        with c1:
            show(charts.effect_bars(bundle.saves_shares[metric]["theme"], "theme",
                                    mode=MODE, title=f"{metric} by theme", colors=tcolors))
        with c2:
            show(charts.effect_bars(bundle.saves_shares[metric]["format"], "format",
                                    mode=MODE, title=f"{metric} by format", colors=fcolors))
    st.markdown("**Rates (per reach)**")
    rc1, rc2 = st.columns(2)
    with rc1:
        show(charts.rate_bars(bundle.rates["save_rate"]["theme"], "theme",
                              mode=MODE, title="save rate by theme", colors=tcolors))
    with rc2:
        show(charts.rate_bars(bundle.rates["share_rate"]["theme"], "theme",
                              mode=MODE, title="share rate by theme", colors=tcolors))

with tab_tf:
    st.subheader("Theme × format — which combinations stand out")
    st.caption("Ratio vs era per cell; ★ marks cells beating both their row and "
               "column marginals. Blank cells have too few posts (n<5).")
    show(charts.cell_heatmap(bundle.cell_ratio, bundle.cell_n, bundle.cell_flags, mode=MODE))
    st.markdown("**Sweet spots** — is there a non-linear caption / hashtag peak?")
    c1, c2 = st.columns(2)
    with c1:
        show(charts.effect_bars(bundle.caption_sweet, "caption_chars", mode=MODE,
                                title="Caption length (quartiles)"))
    with c2:
        show(charts.effect_bars(bundle.hashtag_sweet, "hashtag_count", mode=MODE,
                                title="Hashtag count (quartiles)"))

with tab_trends:
    st.subheader("How the account changed over time")
    show(charts.theme_mix_area(bundle.theme_mix, mode=MODE))
    c1, c2 = st.columns(2)
    with c1:
        show(charts.cadence_line(bundle.cadence, mode=MODE))
    with c2:
        show(charts.seasonality_bars(bundle.seasonality, mode=MODE))
    show(charts.theme_era_bars(bundle.theme_era, mode=MODE))

with tab_aud:
    st.subheader("Who's following (current snapshot)")
    st.caption("Shares only — coarse/aggregate, publish-safe by construction.")
    show(charts.country_choropleth(bundle.country, mode=MODE))
    c1, c2 = st.columns(2)
    with c1:
        show(charts.gender_bars(bundle.gender, mode=MODE))
    with c2:
        show(charts.age_bars(bundle.age, mode=MODE))
    show(charts.city_bars(bundle.city, mode=MODE))
