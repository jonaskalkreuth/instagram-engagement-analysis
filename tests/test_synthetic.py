"""Synthetic generator: determinism, schema parity, and **signal recovery**.

The recovery tests are the real gate — they run the actual transform pipeline +
analyses on generated data and assert the planted signals come back within
the (noise-robust) bands from ``docs/synthetic-design.md``. Bands, not equalities:
theme×format confounding + lognormal noise move exact values.
"""

from __future__ import annotations

import pytest

from insta_analytics.analysis import eda, insight_effects, trends, demographics
from insta_analytics.storage import db, schema
from insta_analytics.synthetic import build, generate
from insta_analytics.synthetic import signals as S
from insta_analytics.transform import labels

N_BOOT = 500  # keep the bootstrap cheap in tests


# --- pure generator -------------------------------------------------------
def test_generate_all_is_deterministic():
    assert generate.generate_all(S.SEED) == generate.generate_all(S.SEED)


def test_row_keys_match_schema_columns_exactly():
    tables = generate.generate_all(S.SEED)
    assert set(tables) == set(schema.TABLES)
    for name, rows in tables.items():
        ddl = schema.TABLES[name]["ddl"]
        # column names are the identifiers before a type keyword in each DDL line
        cols = {
            line.strip().split()[0]
            for line in ddl.splitlines()
            if line.strip() and line.strip().split()[0].islower()
        }
        cols -= {"create", "primary"}
        assert set(rows[0].keys()) == cols, f"{name}: {set(rows[0]) ^ cols}"


def test_post_counts_and_reach_availability():
    tables = generate.generate_all(S.SEED)
    assert len(tables["posts"]) == 440
    n_null = sum(m["reach"] is None for m in tables["post_metrics"])
    assert n_null == S.N_NO_REACH  # earliest 12 → the real 428/440


def test_themes_are_all_valid_taxonomy():
    posts = generate.generate_all(S.SEED)["posts"]
    themes = {p["theme"] for p in posts}
    assert themes <= set(labels.THEMES)
    assert all(p["theme"] for p in posts)  # every post labeled


# --- end-to-end build + recovery ------------------------------------------
@pytest.fixture(scope="module")
def built(tmp_path_factory):
    out = tmp_path_factory.mktemp("synthetic") / "insta.duckdb"
    counts = build.build(out, fresh=True)
    con = db.connect(out)
    try:
        posts = con.execute("SELECT * FROM analysis_posts").df()
        reach = con.execute("SELECT * FROM analysis_reach").df()
        im = insight_effects.derive_insight_metrics(
            reach, insight_effects.load_like_comment_split(con=con)
        )
        demo = demographics.load_demographics(con=con)
    finally:
        con.close()
    return {"counts": counts, "posts": posts, "reach": reach, "im": im, "demo": demo}


def test_pipeline_builds_both_tables(built):
    assert built["counts"]["analysis_posts"] == 440
    assert built["counts"]["analysis_reach"] == 428


def test_engagement_signal_recovers(built):
    posts = built["posts"]
    assert 0.9 <= posts["rel_engagement"].median() <= 1.1  # baseline calibrated
    theme = eda.group_effect(posts, "theme", n_boot=N_BOOT).set_index("theme")["ratio"]
    assert theme["Live/Performance"] > 1.4
    assert theme["Show/Tour"] < 0.85
    fmt = eda.group_effect(posts, "format", n_boot=N_BOOT).set_index("format")["ratio"]
    assert fmt["Image"] < 0.9
    assert fmt.idxmin() == "Image"  # Image is the weakest format (real story)
    assert all(fmt[f] > fmt["Image"] for f in ("Carousel", "Video", "Reel"))


def test_saves_and_shares_signal_recovers(built):
    im = built["im"]
    saves = eda.group_effect(im, "theme", target="rel_saved_log2", n_boot=N_BOOT).set_index("theme")
    assert saves.loc["Live/Performance", "ratio"] > 1.5
    assert saves.loc["Studio/BTS", "ratio"] > 1.5
    saves_fmt = eda.group_effect(im, "format", target="rel_saved_log2", n_boot=N_BOOT).set_index("format")
    assert saves_fmt.loc["Reel", "ratio"] > 1.3

    sh_t = eda.group_effect(im, "theme", target="rel_shares_log2", n_boot=N_BOOT)
    top = sh_t.iloc[0]  # sorted strongest first
    assert top["theme"] == "Live/Performance" and top["ratio_lo"] > 1.0  # clears CI high
    sh_f = eda.group_effect(im, "format", target="rel_shares_log2", n_boot=N_BOOT)
    topf = sh_f.iloc[0]
    assert topf["format"] == "Reel" and topf["ratio_lo"] > 1.0


def test_theme_format_winners_are_performance_studio(built):
    flags = eda.beats_both_marginals(built["posts"])
    winners = [(t, f) for (t, f), v in flags.stack().items() if v]
    assert ("Studio/BTS", "Reel") in winners
    assert any(t == "Live/Performance" for t, _ in winners)
    assert not any(t == "Show/Tour" for t, _ in winners)  # tour never wins a cell


def test_seasonality_is_flat(built):
    seas = trends.seasonality(built["posts"])
    clears = seas[(seas["ratio_lo"] > 1.0) | (seas["ratio_hi"] < 1.0)]
    assert len(clears) == 0  # no month is a real seasonal signal (real story)


def test_demographics_shape(built):
    demo = built["demo"]
    gender = demographics.shares(demo, "gender").set_index("dimension")["share"]
    assert gender.idxmax() == "M" and gender["M"] > 0.6
    age = demographics.shares(demo, "age")
    assert age.iloc[0]["dimension"] == "25-34"  # share-desc within fixed order
    country = demographics.top_n_shares(demo, "country", n=4)
    top2 = country["dimension"].tolist()[:2]
    assert top2 == ["BR", "MX"]
