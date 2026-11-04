# Meta app + Instagram auth

One-time manual setup to get a **long-lived access token** + **IG user id** into
`.env`. Read-only: the pipeline only *reads* your media and insights — it never
posts, likes, follows, or touches your password.

> **Security:** every real value below goes only into `.env`, which is
> git-ignored. Never paste a real token/secret into a commit, screenshot, or
> issue. The public repo ships synthetic data instead.

---

## 0. Prerequisites

- Your Instagram account must be a **Business** or **Creator** account
  (Instagram app → Settings → *Account type and tools*). Switching is free and
  reversible.
- It must be **linked to a Facebook Page** (Instagram app → Settings →
  *Accounts Center* / *Linked accounts*). A Page you own is fine; it can stay
  unpublished.
- A Facebook account you can log into at <https://developers.facebook.com>.

## 1. Create a Meta developer app

1. Go to <https://developers.facebook.com/apps> → **Create app**.
2. **Wizard trap — read this.** The current "What do you want your app to do?"
   screen funnels you toward the *Instagram API with Instagram Login* use case,
   which is the **wrong** flow for this project (it doesn't match the auth helper
   and complicates account insights/demographics). Instead: choose **"Other"** →
   **Next** → app type **"Business"** → **Next**.
3. Name it (e.g. `insta-analytics`), add a contact email, attach your
   Business/Meta account (optional) → **Create app**.
4. On the app dashboard → **Add product** → add **Instagram Graph API**
   (a.k.a. *Instagram* → *Instagram API with Instagram Login* is a different
   product — pick the **Graph API** one that works via a Facebook Page).

The app stays in **Development mode** — that's all you need to read *your own*
account as the app admin. No App Review required.

## 2. Copy the App ID + App Secret into `.env`

1. Copy `.env.example` → `.env` (git-ignored):
   ```bash
   cp .env.example .env
   ```
2. On the app dashboard → **App settings → Basic**, copy **App ID** and
   **App Secret** (click *Show*) into `.env`:
   ```
   INSTA_META_APP_ID=<app id>
   INSTA_META_APP_SECRET=<app secret>
   ```

## 3. Generate a short-lived token (Graph API Explorer)

1. Open the **Graph API Explorer**:
   <https://developers.facebook.com/tools/explorer>
2. Top-right: select your app (`insta-analytics`).
3. **Add permissions** — grant these scopes:
   - `instagram_basic`
   - `instagram_manage_insights`
   - `pages_show_list`
   - `pages_read_engagement`
   - `business_management`
4. Click **Generate Access Token** → approve the popup (choose the linked Page +
   IG account when asked).
5. Copy the token string shown. This is a **short-lived** token (~1 hour) — we
   exchange it for a 60-day one next.

## 4. Exchange + verify with the helper

With `.env` holding the app id/secret, run (paste the short token):

```bash
uv run python -m insta_analytics.collect.auth --short-token <SHORT_TOKEN> --write
```

> **`zsh: command not found: uv`?** `uv` installs to `~/.local/bin`, which a
> fresh VS Code terminal may not have on `PATH`. Fix for the session with
> `export PATH="$HOME/.local/bin:$PATH"`, and permanently with
> `echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.zshrc` (then restart the
> terminal). Run from the **project root** (where `.env` and `pyproject.toml`
> live); `uv run` handles the virtualenv for you.

It will:
- exchange the short token → **long-lived** (~60-day) token,
- find your **IG user id** via the linked Page,
- verify the token can read the account (prints username + follower/media count),
- with `--write`, save `INSTA_IG_ACCESS_TOKEN` + `INSTA_IG_USER_ID` to `.env`.

The token is only ever printed **masked** (`EAAGab…WXYZ`). Drop `--write` to do a
dry run first.

Expected output:

```
→ Exchanging short-lived token for a long-lived one…
  long-lived token: EAAGab…WXYZ  (expires in ~60 days)
→ Finding the linked Instagram Business account…
  ig_user_id: 17890000000000000  (@yourhandle)
→ Verifying the token can read the account…
  ok: @yourhandle — 12345 followers, 123 media
✓ Wrote INSTA_IG_ACCESS_TOKEN + INSTA_IG_USER_ID to .env (git-ignored).
```

## 5. Done

`.env` now has all four values. Run `python -m insta_analytics.collect.client`
to fetch a few posts and confirm the fields.

---

## Troubleshooting

- **`No Instagram Business account linked to any Page`** — your IG account isn't
  Business/Creator, isn't linked to the Page, or the token lacks
  `pages_show_list` / `instagram_basic`. Fix in steps 0 / 3.
- **`Graph API error 101: Invalid Client ID`** — `INSTA_META_APP_ID` in `.env` is
  still the `your_app_id` placeholder, wasn't saved, or has a typo. It must be the
  ~15–16 digit App ID from **App settings → Basic** (no quotes/spaces). Save `.env`
  and re-run.
- **`Graph API error 190`** — token invalid/expired. Generate a fresh short-lived
  token (step 3) and re-run.
- **Token expiry** — long-lived tokens last ~60 days. Re-run this helper to mint
  a new one. There is no automated refresh.
