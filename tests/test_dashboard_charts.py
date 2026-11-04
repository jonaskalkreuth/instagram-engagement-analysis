"""Tests for the Plotly chart builders.

Drives every builder off a real :class:`~.data.DashboardData` bundle (built on the
shared throwaway DuckDB in ``conftest.py`` — no real DB, no secrets, no Streamlit) and asserts each
returns a ``plotly.graph_objects.Figure`` with the expected traces. Also covers the empty-demographics placeholder path and the
choropleth's ISO-3 skip.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

from insta_analytics.dashboard import charts, data, palette


@pytest.fixture
def bundle(db_path):
    return data.compute_dashboard_data(db_path)


def test_effect_bars(bundle):
    fig = charts.effect_bars(bundle.eng_theme, "theme",
                             colors=palette.theme_colors("light"))
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 1 and fig.data[0].type == "bar"
    assert fig.data[0].orientation == "h"
    # 1.0 reference line present
    assert any(s.type == "line" for s in fig.layout.shapes)


def test_rate_bars_and_spearman(bundle):
    rate = charts.rate_bars(bundle.rates["save_rate"]["theme"], "theme")
    assert isinstance(rate, go.Figure) and rate.data[0].type == "bar"
    sp = charts.spearman_bars(bundle.spearman)
    assert sp.data[0].type == "bar"
    assert any(s.type == "line" for s in sp.layout.shapes)  # 0 line


def test_seasonality_bars(bundle):
    fig = charts.seasonality_bars(bundle.seasonality)
    assert fig.data[0].type == "bar"
    assert len(fig.data[0].x) == 12  # Jan..Dec


def test_heatmaps(bundle):
    t = charts.timing_heatmap(bundle.timing)
    assert t.data[0].type == "heatmap"
    c = charts.cell_heatmap(bundle.cell_ratio, bundle.cell_n, bundle.cell_flags)
    assert c.data[0].type == "heatmap"
    # one annotation per non-NaN cell
    n_cells = int(bundle.cell_ratio.notna().to_numpy().sum())
    assert len(c.layout.annotations) == n_cells


def test_trends(bundle):
    area = charts.theme_mix_area(bundle.theme_mix)
    assert len(area.data) == len(bundle.theme_mix.columns)  # one trace per theme
    assert all(tr.type == "scatter" for tr in area.data)
    assert charts.cadence_line(bundle.cadence).data[0].type == "scatter"
    era = charts.theme_era_bars(bundle.theme_era)
    assert len(era.data) == 2 and era.layout.barmode == "group"  # early + late


def test_demographics(bundle):
    assert charts.gender_bars(bundle.gender).data[0].type == "bar"
    assert charts.age_bars(bundle.age).data[0].type == "bar"
    assert charts.city_bars(bundle.city).data[0].type == "bar"
    ch = charts.country_choropleth(bundle.country)
    assert ch.data[0].type == "choropleth"
    # BR resolved to ISO-3 → present in the choropleth locations
    assert "BRA" in list(ch.data[0].locations)


def test_theme_and_format_colors_are_fixed():
    """A series keeps its hue across panels (color follows entity, not rank)."""
    lt = palette.theme_colors("light")
    assert lt["Release/Promo"] == palette.CATEGORICAL["light"][0]
    assert lt["Live/Performance"] == palette.CATEGORICAL["light"][2]
    assert palette.format_colors("dark")["Reel"] == palette.CATEGORICAL["dark"][0]


def test_empty_demographics_placeholder():
    empty = pd.DataFrame({"dimension": [], "share": []})
    fig = charts.gender_bars(empty)
    assert isinstance(fig, go.Figure) and len(fig.data) == 0
    assert fig.layout.annotations  # placeholder text shown


def test_choropleth_skips_missing_iso3():
    df = pd.DataFrame({"dimension": ["BR", "Other (3)"], "share": [0.7, 0.3],
                       "country_name": ["Brazil", "Other (3)"], "iso_alpha": ["BRA", None]})
    fig = charts.country_choropleth(df)
    assert list(fig.data[0].locations) == ["BRA"]  # None ISO-3 dropped


def test_timing_heatmap_colours_log2_ratio_centred_on_one():
    pivot = pd.DataFrame([[1.0, 2.0], [0.5, 4.0]], index=[0, 1], columns=[9, 18])
    trace = charts.timing_heatmap(pivot).data[0]
    np.testing.assert_allclose(np.asarray(trace.z, dtype=float), [[0.0, 1.0], [-1.0, 2.0]])
    np.testing.assert_allclose(np.asarray(trace.customdata, dtype=float), pivot.to_numpy())
    assert (trace.zmin, trace.zmid, trace.zmax) == (-1.0, 0.0, 1.0)
