# Theme taxonomy & coding rules

The locked content taxonomy for hand-labeling every post's **primary theme**
(the `posts.theme` column). This is the coding reference for the labeling
workflow: one primary theme per post, chosen by the rules below.

Grounded in a review of the real back-catalogue: a touring **producer/DJ**
account. This revises the initial
draft — it splits *Show/Tour* and *Broadcast/Mix* out as
their own frequent categories, merges the draft's "Live" + "Performance clip"
into one **Live/Performance**, and folds "Engagement/Community" into
**Personal/Lifestyle** (gratitude/community posts rarely stand alone; they ride
on recaps and releases).

## The 7 categories (6 substantive + catch-all)

### 1. Release/Promo
Music that is **out or announced** — a *product* exists: tracks, EPs, albums,
remixes, free downloads, chart/beatport placements, "grab your copy", streaming
links, compilation features.
- _"Release alert! My next release is around the corner"_
- _"My latest album 'Electric Boogaloo' is now available on all digital platforms"_
- _"'Bust a Move' made it to the top! Grab your copy here…"_

### 2. Show/Tour
An **upcoming in-person** gig: festival lineups, tour dates, booking calls,
"come see me at X".
- _"Next stop: a three-city run this autumn — dates below"_
- _"The festival awaits! The lineup is finally here"_
- _"BIG NEWS — my first headline show abroad is booked"_

### 3. Live/Performance
**Footage or recap of a set already played** in person: dancefloor clips,
"clip from my last set", post-gig thank-yous to a crowd/city.
- _"Thank you to everyone who danced with me last night — what a crowd"_
- _"One more clip from last weekend's open-air set"_
- _"When the bassline came in, the whole room jumped"_

### 4. Broadcast/Mix
A **remote / online** set: radio show, podcast, livestream, "on air now",
SoundCloud/mix upload. The dividing line vs Live/Performance is *online broadcast*
rather than an in-person event.
- _"On air now — my guest mix is streaming live"_
- _"Tune in tonight for a two-hour online set"_
- _"Last week's radio mix is now up to stream"_

### 5. Studio/BTS
**Making music** — process, no product out yet: studio shots, mixing/producing,
"finishing the album", gear, works-in-progress.
- _"Late night in the studio, new track taking shape"_
- _"Testing a new synth for the next EP"_
- _"Final mixdown done on this one"_

### 6. Personal/Lifestyle
**Non-music life** and audience-relationship posts: travel, reflections, New Year
messages, fan-gratitude that isn't a gig recap.
- _"A day off between shows, exploring the old town"_
- _"Back home and still processing the past two weeks"_
- _"Happy New Year, and thank you for all the support this year"_

### 7. Other
Catch-all for anything that fits none above: service ads, memes, off-topic.
- _"Booking open for mixing and mastering services…"_
- _"Monday mood 😎"_

## Coding rules

Posts often do two things at once (a release that also plugs a gig). Assign the
**single primary theme** by:

1. **Dominant call-to-action / main subject wins.** Label the post by what it is
   *mostly asking the reader to do or look at*, not by a secondary mention. A
   post headlined "my new track is out" that also says "catch me at X" → **Release/Promo**.
2. **Tie-breaks between close categories:**
   - Release vs Show — a *music product* out/announced → Release; a *future event*
     → Show. If the headline is the drop, Release wins.
   - Show vs Live/Performance — **timing**: *future* gig → Show; *past/during* set
     footage or recap → Live/Performance.
   - Live/Performance vs Broadcast/Mix — **venue**: *in-person* event → Live;
     *online* radio/podcast/livestream → Broadcast/Mix.
   - Studio vs Release — **product state**: still "working on / finishing" → Studio;
     available/announced → Release.
   - Personal vs the rest — only if the post is *not* primarily about a specific
     music activity (release/show/set/studio).
3. **Read the text, not the emoji.** Captions are emoji-heavy; strip them mentally.
4. **Language-agnostic.** Captions can mix languages — label on meaning.
5. **Carousels/Reels** — judge by the caption + dominant subject, same rules.
6. **Use `Other` sparingly** — only when no substantive category genuinely fits.

## Labeling procedure

- One primary `theme` per post from the 7 values above (exact strings:
  `Release/Promo`, `Show/Tour`, `Live/Performance`, `Broadcast/Mix`, `Studio/BTS`,
  `Personal/Lifestyle`, `Other`).
- An LLM pre-labelled every post against these rules and flagged its
  low-confidence calls. A human reviewed those 90 flagged labels and changed 8.
- Keyword rules do not work here (multilingual, multi-topic, emoji-dense
  captions), so every label is a judgement against these rules.
