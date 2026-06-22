"""
collect_and_prelabel.py
-----------------------
Collects r/nba posts and comments from Reddit's public JSON API,
pre-labels each example using Groq, and saves a CSV ready for human review.

Usage:
    GROQ_API_KEY=your_key_here python scripts/collect_and_prelabel.py

Output:
    data/nba_raw_prelabeled.csv  — review this, correct the 'label' column,
                                   then upload to Colab as your labeled dataset.

Columns in output CSV:
    text          — post/comment text (what goes into the model)
    label         — Groq's pre-assigned label (YOU MUST REVIEW AND CORRECT THIS)
    groq_prelabel — same as label initially; keeps the original for comparison
    source        — where this example came from (post_top, post_controversial, comment)
    notes         — blank; fill in during your annotation review
"""

import csv
import os
import time
import requests
from groq import Groq

# ── Config ─────────────────────────────────────────────────────────────────
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
assert GROQ_API_KEY, "Set GROQ_API_KEY environment variable before running."

TARGET_PER_LABEL = 75      # aim for 75 of each → 225 total
MIN_TEXT_LENGTH  = 40      # skip posts/comments shorter than this
OUTPUT_PATH      = "data/nba_raw_prelabeled.csv"

REDDIT_HEADERS = {"User-Agent": "TakeMeter-DataCollector/0.1"}

client = Groq(api_key=GROQ_API_KEY)

# ── Groq classification prompt ──────────────────────────────────────────────
SYSTEM_PROMPT = """\
You are classifying posts and comments from r/nba, the NBA basketball subreddit.
Assign each post to exactly one of the following three categories.

analysis: The post makes a structured argument backed by at least one specific, verifiable piece of evidence — a statistic, historical comparison, tactical breakdown, or direct player/team comparison. The evidence does real argumentative work: removing it would weaken the claim. The post reasons toward a conclusion rather than simply asserting one.
Example: "Jokic's playoff PER of 28.1 is the highest among active players with 10+ postseason games. He's not just good in the regular season — he's historically dominant when it matters most."

hot_take: A bold, confident opinion stated without supporting evidence, or with evidence that is vague, cherry-picked, or decorative (present to sound credible but not doing real argumentative work). The claim stands on its own assertiveness. Hot takes often use charged language like overrated, trash, the GOAT, not even top 5, and invite disagreement.
Example: "LeBron is the most overrated player in NBA history. Always had superteams and still chokes in big games. His fans are completely delusional."

reaction: An immediate emotional response to a specific recent event — a game result, trade announcement, injury report, draft pick, or in-game moment. The post expresses a feeling about what just happened with little to no argumentative content.
Example: "I CANNOT BELIEVE THEY JUST BLEW A 20-POINT LEAD. Three games in a row. This team is going to give me a heart attack."

Respond with ONLY the label name — nothing else.
Do not explain your reasoning.
Do not add punctuation or extra words.

Valid labels:
analysis
hot_take
reaction\
"""


def classify(text: str) -> str | None:
    """Return one of: 'analysis', 'hot_take', 'reaction', or None on failure."""
    try:
        resp = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": f"Classify this post:\n\n{text[:1200]}"},
            ],
            temperature=0,
            max_tokens=10,
        )
        raw = resp.choices[0].message.content.strip().lower()
        for label in ["analysis", "hot_take", "reaction"]:
            if raw == label or label in raw:
                return label
        return None
    except Exception as e:
        print(f"  [Groq error] {e}")
        return None


# ── Reddit fetchers ─────────────────────────────────────────────────────────

def reddit_get(url: str, params: dict | None = None) -> dict:
    """Single rate-limited Reddit API call."""
    time.sleep(1.1)
    r = requests.get(url, headers=REDDIT_HEADERS, params=params, timeout=15)
    r.raise_for_status()
    return r.json()


def fetch_posts(sort: str, time_filter: str = "month", limit: int = 100) -> list[dict]:
    """Fetch r/nba posts from a given sort (top, controversial, hot, new)."""
    url = f"https://www.reddit.com/r/nba/{sort}.json"
    params = {"limit": limit, "t": time_filter}
    data = reddit_get(url, params)
    posts = []
    for child in data["data"]["children"]:
        p = child["data"]
        # Combine title and selftext; skip pure image/link posts with no body
        title    = p.get("title", "").strip()
        selftext = p.get("selftext", "").strip()
        if selftext in ("[removed]", "[deleted]", ""):
            text = title
        else:
            text = f"{title}\n\n{selftext}"
        if len(text) >= MIN_TEXT_LENGTH and not p.get("is_video", False):
            posts.append({"text": text, "source": f"post_{sort}", "id": p["id"]})
    return posts


def fetch_game_thread_comments(limit_threads: int = 5) -> list[dict]:
    """
    Find recent game threads on r/nba and pull top-level comments.
    Game threads are the richest source of reaction-mode examples.
    """
    comments = []
    # Search for game threads posted in the last week
    search_url = "https://www.reddit.com/r/nba/search.json"
    params = {
        "q": "Game Thread",
        "restrict_sr": "true",
        "sort": "new",
        "t": "week",
        "limit": limit_threads,
    }
    try:
        data = reddit_get(search_url, params)
        threads = data["data"]["children"]
    except Exception as e:
        print(f"  [game thread search error] {e}")
        return comments

    for thread in threads[:limit_threads]:
        post_id = thread["data"]["id"]
        try:
            thread_data = reddit_get(
                f"https://www.reddit.com/r/nba/comments/{post_id}.json",
                {"limit": 50, "depth": 1},
            )
            comment_listing = thread_data[1]["data"]["children"]
            for c in comment_listing:
                if c["kind"] != "t1":
                    continue
                body = c["data"].get("body", "").strip()
                if (
                    len(body) >= MIN_TEXT_LENGTH
                    and body not in ("[removed]", "[deleted]")
                ):
                    comments.append({
                        "text":   body,
                        "source": "comment_game_thread",
                        "id":     c["data"]["id"],
                    })
        except Exception as e:
            print(f"  [comment fetch error for {post_id}] {e}")

    return comments


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    print("=== TakeMeter Data Collection ===\n")

    # Collect raw examples
    print("Fetching r/nba top posts (past month)...")
    top_posts = fetch_posts("top", time_filter="month", limit=100)
    print(f"  Got {len(top_posts)} posts")

    print("Fetching r/nba controversial posts (past month)...")
    controversial_posts = fetch_posts("controversial", time_filter="month", limit=100)
    print(f"  Got {len(controversial_posts)} posts")

    print("Fetching r/nba hot posts...")
    hot_posts = fetch_posts("hot", limit=100)
    print(f"  Got {len(hot_posts)} posts")

    print("Fetching game thread comments...")
    game_comments = fetch_game_thread_comments(limit_threads=8)
    print(f"  Got {len(game_comments)} comments")

    # Deduplicate by id
    seen_ids: set[str] = set()
    all_examples: list[dict] = []
    for ex in top_posts + controversial_posts + hot_posts + game_comments:
        if ex["id"] not in seen_ids:
            seen_ids.add(ex["id"])
            all_examples.append(ex)

    print(f"\nTotal unique examples to label: {len(all_examples)}")

    # Pre-label with Groq
    print("\nPre-labeling with Groq (this takes ~1–2 minutes)...")
    label_counts: dict[str, int] = {"analysis": 0, "hot_take": 0, "reaction": 0, None: 0}
    labeled: list[dict] = []

    for i, ex in enumerate(all_examples, 1):
        label = classify(ex["text"])
        labeled.append({
            "text":          ex["text"],
            "label":         label or "REVIEW",
            "groq_prelabel": label or "UNPARSEABLE",
            "source":        ex["source"],
            "notes":         "",
        })
        label_counts[label] = label_counts.get(label, 0) + 1
        if i % 20 == 0:
            print(f"  {i}/{len(all_examples)} labeled — "
                  f"analysis={label_counts['analysis']} "
                  f"hot_take={label_counts['hot_take']} "
                  f"reaction={label_counts['reaction']} "
                  f"unparseable={label_counts.get(None, 0)}")
        time.sleep(0.15)  # stay well under Groq free-tier rate limit

    # Save CSV
    os.makedirs("data", exist_ok=True)
    fieldnames = ["text", "label", "groq_prelabel", "source", "notes"]
    with open(OUTPUT_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(labeled)

    print(f"\n✅ Saved {len(labeled)} examples to {OUTPUT_PATH}")
    print("\nFinal label distribution (Groq pre-labels):")
    for label, count in label_counts.items():
        if label is not None:
            print(f"  {label}: {count}")
    if label_counts.get(None, 0) > 0:
        print(f"  UNPARSEABLE (review manually): {label_counts[None]}")

    print(f"""
Next steps:
  1. Open data/nba_raw_prelabeled.csv in a spreadsheet or text editor.
  2. Read each row and correct the 'label' column where Groq got it wrong.
     Pay special attention to the analysis/hot_take boundary (see planning.md).
  3. Delete the 'groq_prelabel' and 'source' columns before uploading to Colab
     (the notebook only needs 'text' and 'label').
  4. Aim for ~75 examples per label. If a class is thin after review:
       - analysis:  add more from top/new discussion posts
       - hot_take:  add more from controversial sort
       - reaction:  add more game thread comments
  5. Upload the cleaned CSV to Colab and run Section 1.
""")


if __name__ == "__main__":
    main()
