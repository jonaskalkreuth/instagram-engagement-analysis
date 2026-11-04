# Instagram Musician Analytics

The findings below come from a real musician's Instagram account, analysed privately.
The repo ships a synthetic sample built to reproduce the pattern of those findings.
Anyone can run the pipeline and the dashboard on it without credentials.
The synthetic numbers are not the account's real numbers.

**Question:** which content characteristics (theme, format, caption, timing) are
associated with higher engagement, once account growth is taken out?

- **Key methodology decision:** [ADR 0001](docs/adr/0001-relative-engagement-normalization.md)
- **Theme taxonomy and coding rules:** [docs/theme-taxonomy.md](docs/theme-taxonomy.md)
- **How the synthetic sample is built:** [docs/synthetic-design.md](docs/synthetic-design.md)

## Method

- **Data.** 440 posts from Dec 2020 to Jul 2026, pulled with the Instagram Graph
  API. Collab posts owned by other accounts are excluded. Reach and saves exist
  for 428 posts.
- **Relative engagement.** The account grew over the period, so raw likes mostly
  measure audience size. Each post's likes + comments are divided by an era
  baseline: the median of the 20 surrounding posts, 10 on each side. A ratio of
  1.5× means 50% above what was typical at the time.
- **Effect sizes.** Each group's median ratio, with a 95% percentile bootstrap CI
  (10,000 resamples). Groups under 5 posts get no CI.
- **Model.** Ridge regression and shallow gradient-boosted trees, compared with a
  mean baseline under shuffled 5-fold cross-validation. The features are known
  before a post goes out: theme, format, caption, timing and cadence.
- **Theme labels.** The taxonomy and its coding rules are hand-designed. An LLM
  pre-labelled all 440 posts against those rules. I reviewed the 90 labels it
  marked as low-confidence and changed 8.
- **Privacy.** Public metrics (likes, comments, captions, timestamps) appear as
  they are. Owner-only metrics (reach, saves, shares, demographics) appear only
  as ratios or coarse shares. Follower growth and raw data stay private.

## Key findings

These findings describe the real account. They are stated as directions,
without the real numbers. The dashboard shows the same patterns on the synthetic
sample. The data is observational, so these are associations, not causes.

**1. Theme is associated with engagement more strongly than anything else.**
Live/Performance and Studio/BTS posts sit well above their era baseline.
Show/Tour is the most common theme and sits furthest below it.

**2. Format matters less, and mostly through the theme it carries.**
Carousels, videos and Reels sit above images. Performance clips are mostly Reels
and Carousels, so theme and format overlap. The strongest theme × format cells
are Live/Performance × Carousel, Live/Performance × Reel and Studio/BTS × Reel.

**3. Captions, hashtags and timing show no clear association.**
A call to action, a question, a weekend post and the month all have CIs that
include 1.0. Caption length and hashtag count barely correlate with engagement.

**4. A model explains only a small share of the variation, and theme carries it.**
Ridge regression and gradient-boosted trees both beat a mean baseline under
cross-validation. Trees do no better than the linear model. Theme dominates the
permutation importance in both models. Most of the variation is not explained
by what is knowable before posting.

**5. Saves and shares follow the same pattern.**
Studio/BTS, Live/Performance and Reels get the most saves relative to their era.
Shares are recorded for fewer posts. Only Live/Performance and Reels have share
CIs above 1.0.

**6. The feed drifted from releases to touring.**
Release/Promo became a smaller share of posts over time, and Show/Tour grew.
Live/Performance and Studio/BTS stayed above baseline throughout.

## Limitations

- **Observational data.** These are associations. Theme may stand in for things
  the data does not record, such as the release cycle or video quality.
- **No multiple-comparison correction.** The analyses read about 60 CIs. At 95%,
  a few will exclude 1.0 by chance. Treat single borderline results with care.
- **The CIs are too narrow.** The bootstrap treats posts as independent. Posts
  close in time share a baseline, so they are not.
- **The baseline moves with the content mix.** As Show/Tour grew, the typical
  post changed. Part of Release/Promo's late drop may reflect that shift, not
  weaker releases.
- **The model's R² may be slightly optimistic.** The baseline window is centred,
  so it uses later posts. Shuffled folds also put neighbouring posts, which share
  a baseline, on both sides of a split.
- **LLM-assisted labels.** Theme is the dominant predictor. Its labels were
  LLM-drafted, and the human review covered the 90 low-confidence labels.
- **One account, 440 posts.** Some groups are small, such as Video and
  Studio/BTS. The results describe this account, not Instagram in general.

## Reproduce

Requires Python 3.12 and [uv](https://docs.astral.sh/uv/). Everything below runs
on the synthetic sample, with no credentials.

```bash
uv sync --locked                  # install from the lock file
uv run --locked pytest -q         # 206 tests, a few minutes

# Rebuild the synthetic DB and run the pipeline on it. Without --out, this
# overwrites the committed sample at data/synthetic/insta.duckdb.
uv run --locked python -m insta_analytics.synthetic.build --out /tmp/insta.duckdb

# Dashboard on the committed sample
uv run --locked streamlit run src/insta_analytics/dashboard/app.py

# Dashboard on your rebuilt copy
INSTA_DUCKDB_PATH=/tmp/insta.duckdb uv run --locked streamlit run src/insta_analytics/dashboard/app.py
```

The build prints the effects it recovers from the synthetic data. By design, they
land near the real findings above.

## Dashboard

A Streamlit dashboard shows every analysis: engagement, saves and shares,
theme × format, seasonality and drift, and audience. It reads a DuckDB file by
path and never loads credentials.

`INSTA_DATA_SOURCE` selects the data, from the environment or `.env`:

- `synthetic` (default): the committed sample at `data/synthetic/insta.duckdb`.
- `real`: the private `data/processed/insta.duckdb`, which git ignores.

### Deploy to Streamlit Community Cloud

1. Push the repo to GitHub. The synthetic DB is tracked; real data stays ignored.
2. On [share.streamlit.io](https://share.streamlit.io), create a new app with
   **Main file path** `src/insta_analytics/dashboard/app.py` and Python **3.12**.
3. Leave `INSTA_DATA_SOURCE` unset, so it defaults to `synthetic`. No secrets are needed.

## Collecting real data

To connect your own Instagram account, follow the one-time auth setup in
[docs/setup-meta-auth.md](docs/setup-meta-auth.md). The collectors refuse to run
unless `INSTA_DATA_SOURCE=real`, so real data never lands in the tracked
synthetic DB.

## Stack

`uv` · `pandas` + `duckdb` · `httpx` · `scikit-learn` · `matplotlib` ·
`streamlit` + `plotly` · `pydantic-settings` · `pytest`

## Repository history

This repository is a fresh copy of the original project, created without its git history.
The original repo had the full development history in its commits.
Real Instagram data was never meant to be committed, but old commits were not all audited.
Re-creating the repo from one checked snapshot makes sure no personal data from old commits can leak.
The original history is kept privately.

## License

MIT. See [LICENSE](LICENSE).
