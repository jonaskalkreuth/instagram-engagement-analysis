# 1. Normalize engagement against a rolling "era baseline"

Date: 2026-07-10

## Status

Accepted

## Context

The account has 440 owned posts spanning 5–6 years. Raw engagement (likes +
comments) is **non-stationary** over that span for three reasons:

1. The follower base grew substantially, inflating raw engagement over time.
2. Instagram's format mix and algorithm changed (Reels did not exist early on).
3. The owner's own posting style evolved.

Comparing a 2020 post to a 2025 post on raw counts therefore measures *audience
size and era*, not *content quality*.

The obvious fix — divide by follower-count-at-time-of-post — is **impossible**:
the Instagram Graph API does not expose historical follower counts (only current
and last-28-days). We need normalization that does not depend on data we cannot
get.

## Decision

1. **Relative engagement (primary metric):** express each post's engagement as a
   ratio (or z-score) against an **era baseline** — a rolling median of
   engagement over a window of temporally-surrounding posts. A post that did
   1.8× its era's typical engagement scores 1.8, whether it's from 2020 or 2025.
2. **Time control (robustness):** additionally include a time / date term as a
   control variable in the regression model, and interpret the other
   coefficients "holding era constant." _Not built yet: this regression is
   future work._

## Consequences

- **Good:** keeps all 440 posts (no data thrown away), neutralizes the growth
  confound without inventing follower numbers, and is a strong, honest
  methodological talking point for the portfolio.
- **Cost:** the headline metric is *relative* ("1.8× era baseline"), less
  intuitive to a layperson than raw likes. Requires a chosen baseline window
  size (a tunable knob to document and sensitivity-check).
- **Rejected — restrict to a recent stable window:** cleaner but discards most
  of the sample and reintroduces the small-n problem.
- **Rejected — per-follower normalization:** impossible (no historical follower
  counts via the API).
