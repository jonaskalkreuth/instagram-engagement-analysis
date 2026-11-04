# Synthetic data design

The seeded generator in `synthetic/` writes fake data with the same schema as the
real tables. Run through the real pipeline, it tells the same story as the real
findings, so the public dashboard is safe to commit. Every constant lives in
`synthetic/signals.py`.

## Principles

1. **Match the story, not the numbers.** Reproduce the ordering and rough size of
   each real effect, not its exact ratio. Exact clones would leak the real numbers.
2. **Plant on the raw scale and let the pipeline normalize.** A per-post
   multiplier survives division by the era baseline. A smooth growth trend is
   absorbed by the rolling median and does not.
3. **Seeded and deterministic.** The same seed gives a byte-identical DB.
4. **No planted signal where the real data had none.** Caption length, hashtag
   count, timing and seasonality are drawn independently of the targets.
5. **Verify by recovery.** `tests/test_synthetic.py` runs the real analyses on
   generated data and checks the recovered effects against the bands below.

## Engagement model

```
E_raw(i) = base(t_i) · f_format · f_theme · f_cell(theme, format) · ε_i
ε_i ~ LogNormal(0, 0.35)
```

`base(t)` is an invented follower-growth curve. `f_cell` is 1.0 except for the
three theme × format winners. Saves, shares and reach follow the same pattern,
each with its own factors and its own era baseline.

## Recovery bands

| check | target |
|-------|--------|
| median `rel_engagement` overall | 0.95–1.05 |
| format ordering | Carousel ≳ Video ≳ Reel > Image |
| theme: Live/Perf, Studio/BTS | > 1.3 |
| theme: Show/Tour | < 0.8 |
| theme × format winners | Live/Perf×Reel, ×Carousel, Studio/BTS×Reel |
| saves: Studio/BTS and Live/Perf | > 1.8; Reel > 1.5 |
| shares: Live/Perf and Reel | CI above 1.0; others straddle |
| seasonality | no month clears its CI |
| demographics | M-majority; 25–44 core; BR > MX > DE/PT |
