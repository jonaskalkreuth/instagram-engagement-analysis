"""Constants for the synthetic generator.

Effects are multiplicative factors. The era baseline removes follower growth,
so they survive as the ratios the analyses report.
"""

from __future__ import annotations

from datetime import date

SEED = 20260712
IG_USER_ID = "synthetic-dj"
FOLLOWER_BASE = 9550  # ~current followers (drives the growth curve top end)
FOLLOWER_START = 1200  # invented start value (not the account's history)
TIMELINE_START = date(2020, 12, 1)
TIMELINE_END = date(2026, 7, 12)  # 2026 is partial (to mid-July)

# posts per year (sum = 440); the real cadence ramp → taper
POSTS_PER_YEAR: dict[int, int] = {
    2020: 5,
    2021: 42,
    2022: 68,
    2023: 86,
    2024: 110,
    2025: 90,
    2026: 39,
}

# theme mix by year: drift from a Release/Promo feed to a touring feed
# (real trends). Probabilities per year; exact counts fall out of the RNG.
THEME_MIX_BY_YEAR: dict[int, dict[str, float]] = {
    2020: {"Release/Promo": 0.80, "Studio/BTS": 0.20},
    2021: {"Release/Promo": 0.55, "Show/Tour": 0.15, "Live/Performance": 0.10,
           "Studio/BTS": 0.15, "Personal/Lifestyle": 0.05},
    2022: {"Release/Promo": 0.30, "Show/Tour": 0.34, "Live/Performance": 0.19,
           "Studio/BTS": 0.08, "Broadcast/Mix": 0.05, "Personal/Lifestyle": 0.04},
    2023: {"Release/Promo": 0.24, "Show/Tour": 0.37, "Live/Performance": 0.24,
           "Studio/BTS": 0.07, "Broadcast/Mix": 0.05, "Personal/Lifestyle": 0.03},
    2024: {"Release/Promo": 0.20, "Show/Tour": 0.38, "Live/Performance": 0.28,
           "Studio/BTS": 0.06, "Broadcast/Mix": 0.05, "Personal/Lifestyle": 0.03},
    2025: {"Release/Promo": 0.18, "Show/Tour": 0.36, "Live/Performance": 0.31,
           "Studio/BTS": 0.06, "Broadcast/Mix": 0.06, "Personal/Lifestyle": 0.03},
    2026: {"Release/Promo": 0.18, "Show/Tour": 0.35, "Live/Performance": 0.31,
           "Studio/BTS": 0.06, "Broadcast/Mix": 0.07, "Personal/Lifestyle": 0.03},
}

# format mix by theme (Live/Perf & Studio skew Reel; Promo skews
# Carousel/Image) → reproduces the real theme × format winners.
FORMAT_BY_THEME: dict[str, dict[str, float]] = {
    "Live/Performance":   {"Reel": 0.55, "Carousel": 0.25, "Image": 0.08, "Video": 0.12},
    "Studio/BTS":         {"Reel": 0.50, "Carousel": 0.20, "Image": 0.15, "Video": 0.15},
    "Show/Tour":          {"Reel": 0.30, "Carousel": 0.25, "Image": 0.35, "Video": 0.10},
    "Release/Promo":      {"Reel": 0.25, "Carousel": 0.35, "Image": 0.35, "Video": 0.05},
    "Broadcast/Mix":      {"Reel": 0.25, "Carousel": 0.30, "Image": 0.40, "Video": 0.05},
    "Personal/Lifestyle": {"Reel": 0.20, "Carousel": 0.30, "Image": 0.45, "Video": 0.05},
    "Other":              {"Reel": 0.25, "Carousel": 0.25, "Image": 0.35, "Video": 0.15},
}

# format label → (media_type, media_product_type); inverse of features.format_of.
FORMAT_MEDIA: dict[str, tuple[str, str]] = {
    "Reel":     ("VIDEO", "REELS"),
    "Video":    ("VIDEO", "FEED"),
    "Carousel": ("CAROUSEL_ALBUM", "FEED"),
    "Image":    ("IMAGE", "FEED"),
}

# Tuned below the real marginals, because theme and format correlate and the recovered
# effect double-counts that overlap (target: Live/Perf ~1.9, Studio ~1.6, Show/Tour ~0.6).
F_FORMAT = {"Carousel": 1.42, "Video": 1.22, "Reel": 1.00, "Image": 0.80}
F_THEME = {
    "Live/Performance": 1.72, "Studio/BTS": 1.48, "Release/Promo": 1.00,
    "Broadcast/Mix": 0.98, "Personal/Lifestyle": 1.02, "Show/Tour": 0.64, "Other": 1.00,
}
# theme×format interaction bump so the cell beats BOTH marginals (real theme × format).
F_CELL = {
    ("Live/Performance", "Reel"): 1.18,
    ("Live/Performance", "Carousel"): 1.10,
    ("Studio/BTS", "Reel"): 1.12,
}

# saves multipliers (real: Studio/Live/Reel lead saves)
S_THEME = {
    "Studio/BTS": 2.53, "Live/Performance": 2.39, "Release/Promo": 1.50,
    "Broadcast/Mix": 1.00, "Show/Tour": 0.75, "Personal/Lifestyle": 0.31, "Other": 1.00,
}
S_FORMAT = {"Reel": 2.0, "Video": 1.55, "Carousel": 1.33, "Image": 0.67}

# shares multipliers (real: only Live/Perf & Reel clear CI)
SH_THEME = {
    "Live/Performance": 3.65, "Studio/BTS": 1.05, "Release/Promo": 1.00,
    "Broadcast/Mix": 0.90, "Show/Tour": 0.90, "Personal/Lifestyle": 0.80, "Other": 1.00,
}
SH_FORMAT = {"Reel": 3.0, "Video": 1.05, "Carousel": 0.90, "Image": 0.80}

# reach echoes engagement, milder (compressed toward 1 via **0.6)
REACH_COMPRESS = 0.6  # r_theme = F_THEME**REACH_COMPRESS, same for format
N_NO_REACH = 12       # earliest posts predating the metric (real 428/440)

# lognormal noise sigmas + base rates (fraction of followers)
# SIGMA_SHARE is deliberately large: real shares were spiky enough that only the
# Live/Perf & Reel signals cleared CI, everything else straddled 1.0.
SIGMA_ENG, SIGMA_REACH, SIGMA_SAVE, SIGMA_SHARE = 0.35, 0.30, 0.40, 0.95
ENG_RATE, REACH_RATE, SAVE_RATE, SHARE_RATE = 0.055, 1.20, 0.020, 0.008
COMMENT_SHARE_MEAN, COMMENT_SHARE_SD = 0.045, 0.015
VIEWS_REACH_LO, VIEWS_REACH_HI = 1.5, 3.0  # views ≈ reach × U(lo,hi) for video

# caption shape (NO planted signal — real sweet-spots were flat)
CAPTION_CHARS_LOG_MEAN, CAPTION_CHARS_LOG_SD = 6.0, 0.6  # exp → ~400 median chars
HASHTAG_LAMBDA = 8
HASHTAG_MAX = 30
_CAPTION_WORDS = (
    "psytrance set live tonight festival stage crowd energy studio session new "
    "track release out now tour dates lineup mixdown master vinyl booking rig "
    "soundcheck backstage sunrise forest bass melody groove production remix"
).split()
_HASHTAGS = (
    "psytrance psychedelic festival livemusic dj producer newmusic tour "
    "studio bassmusic goa progressive rave electronicmusic setlife"
).split()
CTA_PHRASES = ("link in bio", "out now", "get your tickets", "check it out")

# account_snapshots: rolling 30-day daily window ending TIMELINE_END
SNAPSHOT_DAYS = 30
DAILY_REACH_MEAN, DAILY_REACH_SD = 3500, 800

# follower_demographics (one snapshot; shares ≈ real ±1–2pp)
DEMO_SNAPSHOT = date(2026, 7, 11)
GENDER_SHARES = {"M": 0.74, "F": 0.14, "U": 0.12}
AGE_SHARES = {
    "13-17": 0.015, "18-24": 0.08, "25-34": 0.50, "35-44": 0.32,
    "45-54": 0.05, "55-64": 0.025, "65+": 0.01,
}
# ISO-2 → share (pycountry maps these for the choropleth). ~25 dims incl. a tail.
COUNTRY_SHARES = {
    "BR": 0.21, "MX": 0.19, "DE": 0.08, "PT": 0.07, "IN": 0.07, "JP": 0.05,
    "CH": 0.04, "IT": 0.03, "ZA": 0.03, "US": 0.028, "GB": 0.025, "FR": 0.022,
    "ES": 0.02, "NL": 0.018, "AR": 0.017, "CO": 0.016, "AT": 0.014, "PL": 0.012,
    "TR": 0.011, "CL": 0.010, "AU": 0.009, "IL": 0.008, "GR": 0.007, "CZ": 0.006,
    "SE": 0.005,
}
CITY_SHARES = {
    "Mexico City": 0.18, "São Paulo": 0.08, "Cape Town": 0.07, "Berlin": 0.05,
    "Lisbon": 0.045, "Tokyo": 0.04, "Rio de Janeiro": 0.035, "Guadalajara": 0.03,
    "Zürich": 0.028, "Vienna": 0.025, "Madrid": 0.022, "London": 0.02,
    "Buenos Aires": 0.018, "Bogotá": 0.016, "Paris": 0.015, "Amsterdam": 0.014,
    "Rome": 0.013, "Munich": 0.012, "Athens": 0.011, "Warsaw": 0.010,
    "Istanbul": 0.009, "Santiago": 0.008, "Tel Aviv": 0.007, "Prague": 0.006,
    "Stockholm": 0.005,
}
