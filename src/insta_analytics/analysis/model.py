"""Predict relative engagement from features known before a post goes out.

- Ridge and shallow gradient-boosted trees, scored against a mean baseline.
- Shuffled, seeded k-fold CV: R² and MAE on the log2 scale.
- Importance: permutation on a held-out split, plus Ridge coefficients.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import duckdb
import matplotlib

matplotlib.use("Agg")  # headless: render to file, never open a window
import matplotlib.pyplot as plt  # noqa: E402  (must follow the backend switch)
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.compose import ColumnTransformer  # noqa: E402
from sklearn.dummy import DummyRegressor  # noqa: E402
from sklearn.ensemble import HistGradientBoostingRegressor  # noqa: E402
from sklearn.impute import SimpleImputer  # noqa: E402
from sklearn.inspection import permutation_importance  # noqa: E402
from sklearn.linear_model import Ridge  # noqa: E402
from sklearn.model_selection import KFold, cross_validate, train_test_split  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import OneHotEncoder, StandardScaler  # noqa: E402

from ..config import Settings, get_settings  # noqa: E402
from ..storage import db  # noqa: E402
from ..transform.analysis_table import PRIMARY_TABLE  # noqa: E402
from . import eda  # noqa: E402

TARGET = "rel_engagement_log2"

# Pre-publication features only. Non-monotone ones are one-hot encoded.
CATEGORICAL_FEATURES: tuple[str, ...] = ("theme", "format", "dow", "month")
NUMERIC_FEATURES: tuple[str, ...] = (
    "caption_chars",
    "caption_words",
    "line_count",
    "hashtag_count",
    "mention_count",
    "emoji_count",
    "has_question",
    "has_cta",
    "hour",
    "is_weekend",
    "days_since_prev",
    "posts_prev_30d",
)
FEATURES: tuple[str, ...] = CATEGORICAL_FEATURES + NUMERIC_FEATURES

DEFAULT_SEED = 0
DEFAULT_FOLDS = 5
DEFAULT_ALPHA = 1.0  # fixed, not tuned, at n=440
DEFAULT_N_PERM = 30  # permutation-importance repeats
DEFAULT_TEST_SIZE = 0.25  # held-out fraction for permutation importance


def build_design(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """``(X, y)`` without NaN-target rows; categoricals as ``category``, the rest float."""
    d = df.dropna(subset=[TARGET])
    X = d[list(FEATURES)].copy()
    for c in CATEGORICAL_FEATURES:
        X[c] = X[c].astype("category")
    for c in NUMERIC_FEATURES:
        X[c] = X[c].astype(float)
    y = d[TARGET].astype(float)
    return X.reset_index(drop=True), y.reset_index(drop=True)


def make_linear_model(alpha: float = DEFAULT_ALPHA) -> Pipeline:
    """Ridge over one-hot categoricals and imputed, standardized numerics.

    Ridge absorbs collinear pairs (caption chars/words, is_weekend/dow) that would
    break plain OLS.
    """
    pre = ColumnTransformer(
        [
            ("cat", OneHotEncoder(handle_unknown="ignore"), list(CATEGORICAL_FEATURES)),
            (
                "num",
                Pipeline(
                    [("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]
                ),
                list(NUMERIC_FEATURES),
            ),
        ]
    )
    return Pipeline([("pre", pre), ("model", Ridge(alpha=alpha))])


def make_tree_model(seed: int = DEFAULT_SEED) -> HistGradientBoostingRegressor:
    """Shallow gradient-boosted trees, kept small so they don't memorize 440 rows."""
    return HistGradientBoostingRegressor(
        categorical_features=list(CATEGORICAL_FEATURES),
        learning_rate=0.05,
        max_iter=200,
        max_depth=3,
        min_samples_leaf=20,
        l2_regularization=1.0,
        random_state=seed,
    )


def cross_validate_models(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    folds: int = DEFAULT_FOLDS,
    seed: int = DEFAULT_SEED,
) -> pd.DataFrame:
    """R² and MAE (mean ± std over folds) for linear, trees and a mean baseline."""
    cv = KFold(n_splits=folds, shuffle=True, random_state=seed)
    estimators = {
        "linear": make_linear_model(),
        "trees": make_tree_model(seed),
        "baseline": DummyRegressor(strategy="mean"),
    }
    rows = []
    for name, est in estimators.items():
        res = cross_validate(
            est, X, y, cv=cv, scoring=["r2", "neg_mean_absolute_error"]
        )
        rows.append(
            {
                "model": name,
                "r2_mean": float(res["test_r2"].mean()),
                "r2_std": float(res["test_r2"].std()),
                "mae_mean": float(-res["test_neg_mean_absolute_error"].mean()),
                "mae_std": float(res["test_neg_mean_absolute_error"].std()),
            }
        )
    return pd.DataFrame(rows)


def linear_coefficients(
    X: pd.DataFrame, y: pd.Series, *, alpha: float = DEFAULT_ALPHA
) -> pd.DataFrame:
    """Fit the linear model on all rows and return its coefficients, sorted by |coef|."""
    m = make_linear_model(alpha)
    m.fit(X, y)
    names = m.named_steps["pre"].get_feature_names_out()
    coefs = np.asarray(m.named_steps["model"].coef_, dtype=float)
    out = pd.DataFrame({"feature": names, "coef": coefs})
    out["abs_coef"] = out["coef"].abs()
    return out.sort_values("abs_coef", ascending=False, kind="stable").reset_index(drop=True)


def permutation_importances(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    seed: int = DEFAULT_SEED,
    n_repeats: int = DEFAULT_N_PERM,
    test_size: float = DEFAULT_TEST_SIZE,
) -> pd.DataFrame:
    """Permutation importance (R² drop) for both models on a held-out split.

    Shuffles original columns, not one-hot levels, so a categorical stays whole.
    """
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_size, random_state=seed)
    frames = []
    for name, est in (("linear", make_linear_model()), ("trees", make_tree_model(seed))):
        est.fit(X_tr, y_tr)
        pi = permutation_importance(
            est, X_te, y_te, scoring="r2", n_repeats=n_repeats, random_state=seed
        )
        frames.append(
            pd.DataFrame(
                {
                    "feature": list(FEATURES),
                    "model": name,
                    "importance": pi.importances_mean,
                    "importance_std": pi.importances_std,
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


@dataclass
class ModelResults:
    cv: pd.DataFrame
    coefficients: pd.DataFrame
    permutation: pd.DataFrame


def fit_evaluate(
    df: pd.DataFrame,
    *,
    seed: int = DEFAULT_SEED,
    folds: int = DEFAULT_FOLDS,
    n_perm: int = DEFAULT_N_PERM,
    test_size: float = DEFAULT_TEST_SIZE,
) -> ModelResults:
    """Run CV, coefficients and permutation importance on an analysis frame."""
    X, y = build_design(df)
    return ModelResults(
        cv=cross_validate_models(X, y, folds=folds, seed=seed),
        coefficients=linear_coefficients(X, y),
        permutation=permutation_importances(
            X, y, seed=seed, n_repeats=n_perm, test_size=test_size
        ),
    )


_MODEL_COLORS = {"linear": "#5779a3", "trees": "#c13584", "baseline": "#999999"}


def plot_cv(cv: pd.DataFrame) -> plt.Figure:
    """Bars of CV R² per model with ±std whiskers."""
    fig, ax = plt.subplots(figsize=(6, 3.5))
    xpos = np.arange(len(cv))
    colors = [_MODEL_COLORS.get(m, "#c13584") for m in cv["model"]]
    ax.bar(xpos, cv["r2_mean"], yerr=cv["r2_std"], color=colors, alpha=0.85, capsize=4)
    ax.axhline(0.0, color="#666", linestyle="--", linewidth=1)
    ax.set_xticks(xpos)
    ax.set_xticklabels(cv["model"])
    ax.set_ylabel("cross-validated R²  (0 = mean predictor)")
    ax.set_title("Predictive skill on relative engagement (log2)")
    fig.tight_layout()
    return fig


def plot_permutation(perm: pd.DataFrame, model: str, top: int = 12) -> plt.Figure:
    """Horizontal bars of the top permutation-importance features for one model."""
    sub = (
        perm[perm["model"] == model]
        .sort_values("importance", ascending=False, kind="stable")
        .head(top)
        .iloc[::-1]  # strongest at top
    )
    fig, ax = plt.subplots(figsize=(7, max(2.5, 0.4 * len(sub) + 1)))
    ypos = np.arange(len(sub))
    ax.barh(ypos, sub["importance"], xerr=sub["importance_std"],
            color=_MODEL_COLORS.get(model, "#c13584"), alpha=0.85, capsize=3)
    ax.axvline(0.0, color="#666", linewidth=1)
    ax.set_yticks(ypos)
    ax.set_yticklabels(sub["feature"])
    ax.set_xlabel("mean drop in R² when shuffled")
    ax.set_title(f"Permutation importance — {model}")
    fig.tight_layout()
    return fig


def plot_coefficients(coefs: pd.DataFrame, top: int = 15) -> plt.Figure:
    """Horizontal bars of the largest-magnitude standardized linear coefficients."""
    sub = coefs.head(top).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, max(2.5, 0.4 * len(sub) + 1)))
    ypos = np.arange(len(sub))
    colors = ["#c13584" if c >= 0 else "#5779a3" for c in sub["coef"]]
    ax.barh(ypos, sub["coef"], color=colors, alpha=0.85)
    ax.axvline(0.0, color="#666", linewidth=1)
    ax.set_yticks(ypos)
    ax.set_yticklabels(sub["feature"])
    ax.set_xlabel("Ridge coefficient (log2 units; numerics standardized)")
    ax.set_title("Linear model — largest coefficients")
    fig.tight_layout()
    return fig


def save_figures(results: ModelResults, *, figures_dir: Path) -> list[Path]:
    """Write the model figures to ``figures_dir``."""
    figures_dir.mkdir(parents=True, exist_ok=True)
    specs = [
        ("model_cv.png", plot_cv(results.cv)),
        ("model_perm_linear.png", plot_permutation(results.permutation, "linear")),
        ("model_perm_trees.png", plot_permutation(results.permutation, "trees")),
        ("model_coefs.png", plot_coefficients(results.coefficients)),
    ]
    paths = []
    for name, fig in specs:
        p = figures_dir / name
        fig.savefig(p, dpi=120)
        plt.close(fig)
        paths.append(p)
    return paths


def report(
    *,
    figures: bool = True,
    seed: int = DEFAULT_SEED,
    folds: int = DEFAULT_FOLDS,
    n_perm: int = DEFAULT_N_PERM,
    con: duckdb.DuckDBPyConnection | None = None,
    settings: Settings | None = None,
) -> None:
    """Print CV metrics and importances and write figures."""
    settings = settings or get_settings()
    with db.connection(con, settings) as con:
        posts = eda.load_analysis_posts(con=con, settings=settings)

    results = fit_evaluate(posts, seed=seed, folds=folds, n_perm=n_perm)
    pd.set_option("display.float_format", lambda v: f"{v:.3f}")

    print(f"analysis_posts: {len(posts)}  |  target: {TARGET}\n")
    print("=== Cross-validated performance (mean ± std over folds) ===")
    print(results.cv.to_string(index=False))

    print("\n=== Linear model — top coefficients (log2 units) ===")
    print(results.coefficients.head(15).to_string(index=False))

    for model in ("linear", "trees"):
        print(f"\n=== Permutation importance — {model} (top 10) ===")
        sub = results.permutation[results.permutation["model"] == model]
        top = sub.sort_values("importance", ascending=False, kind="stable").head(10)
        print(top.to_string(index=False))

    if figures:
        paths = save_figures(results, figures_dir=settings.figures_dir)
        print("\nfigures:")
        for p in paths:
            print(f"  → {p}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Predictive model: CV'd linear + trees on relative engagement."
    )
    parser.add_argument("--no-figures", action="store_true", help="Skip writing PNGs.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--folds", type=int, default=DEFAULT_FOLDS)
    parser.add_argument("--n-perm", type=int, default=DEFAULT_N_PERM)
    args = parser.parse_args()
    report(figures=not args.no_figures, seed=args.seed, folds=args.folds, n_perm=args.n_perm)


if __name__ == "__main__":
    main()
