"""EDA — pure effect-size + correlation functions on inline fixtures."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from insta_analytics.analysis import eda


def _posts(theme, log2):
    """Minimal analysis frame: a categorical cut + the log2-relative target."""
    return pd.DataFrame({"theme": list(theme), "rel_engagement_log2": list(log2)})


# --- group_effect -----------------------------------------------------------
def test_group_effect_ratio_is_two_to_the_median():
    # log2 medians 1 and -1 → ratios 2.0 and 0.5.
    df = _posts(["a"] * 6 + ["b"] * 6, [1.0] * 6 + [-1.0] * 6)
    eff = eda.group_effect(df, "theme", min_n=1)
    got = dict(zip(eff["theme"], eff["ratio"]))
    assert got["a"] == pytest.approx(2.0)
    assert got["b"] == pytest.approx(0.5)


def test_group_effect_sorted_strongest_first_with_n():
    df = _posts(["hi"] * 6 + ["lo"] * 6, [2.0] * 6 + [-2.0] * 6)
    eff = eda.group_effect(df, "theme", min_n=1)
    assert list(eff["theme"]) == ["hi", "lo"]  # descending ratio
    assert list(eff["n"]) == [6, 6]


def test_group_effect_ci_brackets_the_ratio_and_is_seeded():
    df = _posts(["a"] * 40, list(np.linspace(-1, 1, 40)))
    a = eda.group_effect(df, "theme", n_boot=500, seed=7)
    b = eda.group_effect(df, "theme", n_boot=500, seed=7)
    row = a.iloc[0]
    assert row["ratio_lo"] <= row["ratio"] <= row["ratio_hi"]
    pd.testing.assert_frame_equal(a, b)  # same seed → identical


def test_group_effect_small_group_has_nan_ci_but_a_point_ratio():
    df = _posts(["big"] * 10 + ["tiny"] * 2, [0.0] * 10 + [1.0, 1.0])
    eff = eda.group_effect(df, "theme", min_n=5).set_index("theme")
    assert eff.loc["tiny", "n"] == 2
    assert np.isnan(eff.loc["tiny", "ratio_lo"])
    assert eff.loc["tiny", "ratio"] == pytest.approx(2.0)  # 2**median([1,1])
    assert not np.isnan(eff.loc["big", "ratio_lo"])  # big group keeps its CI


def test_group_effect_drops_nan_target_rows():
    df = _posts(["a"] * 4, [1.0, 1.0, np.nan, np.nan])
    eff = eda.group_effect(df, "theme", min_n=1)
    assert int(eff.iloc[0]["n"]) == 2  # the two NaN rows excluded


# --- group_median (raw-scale rate twin) -------------------------------------
def test_group_effect_empty_input_returns_empty_frame_with_columns():
    # A dashboard year filter can leave no rows with a valid target.
    df = _posts(["a"], [np.nan])
    eff = eda.group_effect(df, "theme")
    assert eff.empty
    assert list(eff.columns) == ["theme", "n", "ratio", "ratio_lo", "ratio_hi", "median_log2"]


def test_group_median_empty_input_returns_empty_frame_with_columns():
    df = pd.DataFrame({"format": ["Reel"], "save_rate": [np.nan]})
    med = eda.group_median(df, "format", target="save_rate")
    assert med.empty
    assert list(med.columns) == ["format", "n", "median", "median_lo", "median_hi"]


def test_group_median_reports_raw_median_not_a_ratio():
    # raw rates: group 'a' median 0.4, group 'b' median 0.1 — no 2** transform.
    df = pd.DataFrame(
        {"theme": ["a"] * 6 + ["b"] * 6, "save_rate": [0.4] * 6 + [0.1] * 6}
    )
    out = eda.group_median(df, "theme", target="save_rate", min_n=1)
    got = dict(zip(out["theme"], out["median"]))
    assert got["a"] == pytest.approx(0.4)
    assert got["b"] == pytest.approx(0.1)
    assert list(out["theme"]) == ["a", "b"]  # sorted highest median first


def test_group_median_ci_brackets_and_is_seeded():
    df = pd.DataFrame(
        {"theme": ["a"] * 40, "save_rate": list(np.linspace(0.0, 1.0, 40))}
    )
    a = eda.group_median(df, "theme", target="save_rate", n_boot=500, seed=3)
    b = eda.group_median(df, "theme", target="save_rate", n_boot=500, seed=3)
    row = a.iloc[0]
    assert row["median_lo"] <= row["median"] <= row["median_hi"]
    pd.testing.assert_frame_equal(a, b)  # same seed → identical


def test_group_median_small_group_nan_ci_and_drops_nan_target():
    df = pd.DataFrame(
        {
            "theme": ["big"] * 10 + ["tiny"] * 2 + ["big"] * 2,
            "save_rate": [0.2] * 10 + [0.5, 0.5] + [np.nan, np.nan],
        }
    )
    out = eda.group_median(df, "theme", target="save_rate", min_n=5).set_index("theme")
    assert out.loc["big", "n"] == 10  # two NaN-target rows dropped
    assert out.loc["tiny", "n"] == 2
    assert np.isnan(out.loc["tiny", "median_lo"])  # too few to bootstrap
    assert not np.isnan(out.loc["big", "median_lo"])


# --- quantile_bucket --------------------------------------------------------
def test_quantile_bucket_makes_ordered_quartiles():
    s = pd.Series(np.arange(100, dtype=float))
    out = eda.quantile_bucket(s, 4, labels=["q1", "q2", "q3", "q4"])
    assert out.cat.ordered
    assert list(out.cat.categories) == ["q1", "q2", "q3", "q4"]
    assert out.value_counts().nunique() == 1  # balanced 25/25/25/25


def test_quantile_bucket_collapses_duplicate_edges():
    # a spiky predictor: >half identical → fewer bins, no ValueError.
    s = pd.Series([0.0] * 60 + list(np.linspace(1.0, 2.0, 40)))
    out = eda.quantile_bucket(s, 4)
    assert out.dropna().cat.categories.size < 4


def test_quantile_bucket_all_nan_returns_nan_not_crash():
    s = pd.Series([np.nan] * 10)
    out = eda.quantile_bucket(s, 4)
    assert out.isna().all()


# --- spearman_table ---------------------------------------------------------
def test_spearman_detects_monotone_relationship():
    n = 60
    x = np.arange(n, dtype=float)
    df = pd.DataFrame({"caption_chars": x, "rel_engagement_log2": x})  # perfectly monotone
    tbl = eda.spearman_table(df, ("caption_chars",), n_boot=200)
    assert tbl.iloc[0]["rho"] == pytest.approx(1.0)
    assert tbl.iloc[0]["rho_lo"] <= 1.0


def test_spearman_sorted_by_abs_rho():
    n = 80
    x = np.arange(n, dtype=float)
    df = pd.DataFrame(
        {
            "caption_chars": x,          # strong +
            "hashtag_count": -x,         # strong -
            "days_since_prev": np.tile([0.0, 1.0], n // 2),  # weak/none
            "rel_engagement_log2": x,
        }
    )
    tbl = eda.spearman_table(
        df, ("days_since_prev", "caption_chars", "hashtag_count"), n_boot=200
    )
    # days_since_prev (weakest |rho|) must sort last.
    assert tbl.iloc[-1]["predictor"] == "days_since_prev"
    assert abs(tbl.iloc[0]["rho"]) >= abs(tbl.iloc[-1]["rho"])


def test_spearman_constant_predictor_is_nan_not_crash():
    df = pd.DataFrame(
        {"hashtag_count": [3.0] * 20, "rel_engagement_log2": np.linspace(0, 1, 20)}
    )
    tbl = eda.spearman_table(df, ("hashtag_count",), n_boot=100)
    assert np.isnan(tbl.iloc[0]["rho"])


# --- clears_ci --------------------------------------------------------------
def test_clears_ci_keeps_only_rows_off_baseline():
    eff = pd.DataFrame(
        {
            "theme": ["up", "flat", "down"],
            "ratio": [2.0, 1.0, 0.5],
            "ratio_lo": [1.5, 0.8, 0.3],
            "ratio_hi": [2.5, 1.2, 0.7],
        }
    )
    kept = set(eda.clears_ci(eff)["theme"])
    assert kept == {"up", "down"}  # 'flat' CI straddles 1.0


# --- cell_medians -----------------------------------------------------------
def _cells(rows):
    """rows = list of (theme, format, log2) tuples → analysis_posts-shaped frame."""
    theme, fmt, log2 = zip(*rows)
    return pd.DataFrame(
        {"theme": theme, "format": fmt, "rel_engagement_log2": log2}
    )


def test_cell_medians_ratio_is_two_to_the_cell_median():
    df = _cells([("a", "Reel", 1.0)] * 5 + [("a", "Image", -1.0)] * 5)
    ratio, n = eda.cell_medians(df, min_n=1)
    assert ratio.loc["a", "Reel"] == pytest.approx(2.0)
    assert ratio.loc["a", "Image"] == pytest.approx(0.5)
    assert n.loc["a", "Reel"] == 5


def test_cell_medians_blanks_thin_cells_but_keeps_n():
    df = _cells([("a", "Reel", 1.0)] * 6 + [("a", "Image", 1.0)] * 2)
    ratio, n = eda.cell_medians(df, min_n=5)
    assert not np.isnan(ratio.loc["a", "Reel"])   # n=6 kept
    assert np.isnan(ratio.loc["a", "Image"])       # n=2 blanked
    assert n.loc["a", "Image"] == 2                # count still reported


# --- beats_both_marginals ---------------------------------------------------
def test_beats_both_marginals_flags_a_real_interaction():
    # a×Reel is high; a's other format and Reel's other theme are low → beats both.
    df = _cells(
        [("a", "Reel", 2.0)] * 6    # the standout combo
        + [("a", "Image", 0.0)] * 6
        + [("b", "Reel", 0.0)] * 6
        + [("b", "Image", -1.0)] * 6  # the worst combo
    )
    flags = eda.beats_both_marginals(df, min_n=1)
    assert bool(flags.loc["a", "Reel"])       # 4.0 > both marginals (2.0)
    assert not bool(flags.loc["b", "Image"])  # 0.5 < b-row marginal (0.71)


def test_beats_both_marginals_thin_cell_is_false():
    df = _cells([("a", "Reel", 3.0)] * 2 + [("a", "Image", 0.0)] * 6)
    flags = eda.beats_both_marginals(df, min_n=5)
    assert not bool(flags.loc["a", "Reel"])  # blanked cell can't be a winner


# --- sweet_spot -------------------------------------------------------------
def test_sweet_spot_returns_buckets_in_order_not_by_ratio():
    x = np.arange(40, dtype=float)
    df = pd.DataFrame({"caption_chars": x, "rel_engagement_log2": np.zeros(40)})
    eff = eda.sweet_spot(df, "caption_chars", n_boot=100, min_n=1)
    assert len(eff) == 4
    # bucket edges ascend down the rows (order preserved, not ratio-sorted)
    lows = [float(s.strip("([]").split(",")[0]) for s in eff["caption_chars"]]
    assert lows == sorted(lows)


def test_sweet_spot_recovers_a_middle_peak():
    x = np.arange(40, dtype=float)
    y = -((x - 20) ** 2) / 100.0  # inverted-U → a middle quartile should top ratio
    df = pd.DataFrame({"caption_chars": x, "rel_engagement_log2": y})
    eff = eda.sweet_spot(df, "caption_chars", n_boot=100, min_n=1)
    top = eff.loc[eff["ratio"].idxmax()]
    assert top.name in (1, 2)  # a middle (not edge) quartile is the peak


# --- figures don't crash ----------------------------------------------------
def test_save_figures_writes_all_pngs(tmp_path):
    n = 30
    common = {
        "format": (["Reel", "Image", "Carousel"] * (n // 3)),
        "theme": (["a", "b", "c"] * (n // 3)),
        "dow": list(np.arange(n) % 7),
        "hour": list(9 + np.arange(n) % 12),
        "caption_chars": list(np.arange(n) * 3.0),
        "hashtag_count": list(np.arange(n) % 10),
        "rel_engagement_log2": list(np.linspace(-1, 1, n)),
    }
    posts = pd.DataFrame(common)
    reach = posts.assign(rel_reach_log2=np.linspace(-0.5, 0.5, n))
    paths = eda.save_figures(posts, reach, figures_dir=tmp_path, n_boot=100, min_n=1)
    assert len(paths) == 8
    assert all(p.exists() for p in paths)
