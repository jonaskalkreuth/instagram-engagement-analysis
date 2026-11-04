"""Predictive model — pure fit/evaluate on an inline synthetic fixture."""

from __future__ import annotations

import numpy as np
import pandas as pd

from insta_analytics.analysis import model


def _frame(n: int = 240, seed: int = 0) -> pd.DataFrame:
    """Analysis-shaped frame where the target depends on ``format`` (Reel↑, Image↓).

    Every model feature column is present; the planted format signal gives the models
    something real to beat the baseline on, while the rest is noise.
    """
    rng = np.random.default_rng(seed)
    fmt = rng.choice(["Reel", "Image", "Carousel", "Video"], n)
    signal = np.where(fmt == "Reel", 1.0, np.where(fmt == "Image", -1.0, 0.0))
    y = signal + rng.normal(0, 0.3, n)
    return pd.DataFrame(
        {
            "theme": rng.choice(["Show/Tour", "Live/Performance", "Studio/BTS"], n),
            "format": fmt,
            "dow": rng.integers(0, 7, n),
            "month": rng.integers(1, 13, n),
            "caption_chars": rng.integers(0, 800, n),
            "caption_words": rng.integers(0, 120, n),
            "line_count": rng.integers(1, 10, n),
            "hashtag_count": rng.integers(0, 25, n),
            "mention_count": rng.integers(0, 5, n),
            "emoji_count": rng.integers(0, 10, n),
            "has_question": rng.integers(0, 2, n),
            "has_cta": rng.integers(0, 2, n),
            "hour": rng.integers(0, 24, n),
            "is_weekend": rng.integers(0, 2, n),
            "days_since_prev": rng.random(n) * 10,
            "posts_prev_30d": rng.integers(0, 30, n),
            model.TARGET: y,
        }
    )


# --- build_design -----------------------------------------------------------
def test_build_design_drops_nan_target_and_sets_dtypes():
    df = _frame(n=20)
    df.loc[0, model.TARGET] = np.nan
    df.loc[1, model.TARGET] = np.nan
    X, y = model.build_design(df)
    assert len(X) == len(y) == 18  # two NaN-target rows dropped
    assert list(X.columns) == list(model.FEATURES)
    for c in model.CATEGORICAL_FEATURES:
        assert isinstance(X[c].dtype, pd.CategoricalDtype)
    assert y.dtype == float


def test_build_design_handles_nan_numeric_leading_cadence():
    # days_since_prev's first row is NaN in real data — must survive to the model.
    df = _frame(n=20)
    df.loc[0, "days_since_prev"] = np.nan
    X, _ = model.build_design(df)
    assert X["days_since_prev"].isna().sum() == 1  # kept, not dropped


# --- cross-validation -------------------------------------------------------
def test_models_beat_the_mean_baseline():
    res = model.fit_evaluate(_frame(), n_perm=5)
    cv = res.cv.set_index("model")
    assert set(cv.index) == {"linear", "trees", "baseline"}
    assert cv.loc["baseline", "r2_mean"] < 0.05  # mean predictor ≈ 0
    assert cv.loc["linear", "r2_mean"] > cv.loc["baseline", "r2_mean"]
    assert cv.loc["trees", "r2_mean"] > cv.loc["baseline", "r2_mean"]


def test_cv_reports_mae_and_positive_std():
    cv = model.cross_validate_models(*model.build_design(_frame()), folds=5)
    assert (cv["mae_mean"] > 0).all()
    assert (cv["r2_std"] >= 0).all()


# --- importances ------------------------------------------------------------
def test_permutation_ranks_the_signal_feature_top():
    res = model.fit_evaluate(_frame(), n_perm=20)
    for m in ("linear", "trees"):
        sub = res.permutation[res.permutation["model"] == m].set_index("feature")
        # format carries the planted signal → most important, and clearly positive.
        assert sub["importance"].idxmax() == "format"
        assert sub.loc["format", "importance"] > sub.loc["mention_count", "importance"]


def test_coefficients_cover_features_and_sort_by_magnitude():
    coefs = model.linear_coefficients(*model.build_design(_frame()))
    assert (coefs["abs_coef"].to_numpy() == np.sort(coefs["abs_coef"])[::-1]).all()
    # a Reel / Image one-hot level should surface among the strongest coefficients.
    assert coefs.head(6)["feature"].str.contains("format").any()


# --- determinism ------------------------------------------------------------
def test_fit_evaluate_is_seeded():
    a = model.fit_evaluate(_frame(), n_perm=5, seed=7)
    b = model.fit_evaluate(_frame(), n_perm=5, seed=7)
    pd.testing.assert_frame_equal(a.cv, b.cv)
    pd.testing.assert_frame_equal(a.coefficients, b.coefficients)
    pd.testing.assert_frame_equal(a.permutation, b.permutation)


# --- figures ----------------------------------------------------------------
def test_save_figures_writes_four_pngs(tmp_path):
    res = model.fit_evaluate(_frame(), n_perm=5)
    paths = model.save_figures(res, figures_dir=tmp_path)
    assert [p.name for p in paths] == [
        "model_cv.png",
        "model_perm_linear.png",
        "model_perm_trees.png",
        "model_coefs.png",
    ]
    assert all(p.exists() and p.stat().st_size > 0 for p in paths)
