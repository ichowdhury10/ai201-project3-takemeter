"""
collect_and_prelabel.py
-----------------------
Generates 300 realistic r/nba posts (100 per label) using Groq,
then saves them as a CSV ready for human review and correction.

No Reddit account or HuggingFace credentials needed.

Usage:
    GROQ_API_KEY=your_key python scripts/collect_and_prelabel.py

Output:
    data/nba_raw_prelabeled.csv  — review every row, correct any
                                   that don't fit the label cleanly,
                                   then keep only 'text' + 'label'.
"""

import csv
import json
import os
import time
from groq import Groq

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
assert GROQ_API_KEY, "Set GROQ_API_KEY environment variable."

PER_LABEL   = 100   # generates 100 × 3 = 300 examples
BATCH_SIZE  = 10    # examples per API call
OUTPUT_PATH = "data/nba_raw_prelabeled.csv"

client = Groq(api_key=GROQ_API_KEY)

# ── Generation prompts per label ────────────────────────────────────────────

GENERATION_PROMPTS = {
    "analysis": """\
Generate {n} realistic r/nba Reddit posts that are clear examples of ANALYSIS.

An analysis post makes a structured argument backed by at least one specific, verifiable piece of evidence — a statistic, historical comparison, tactical breakdown, or direct player/team comparison. The evidence does real argumentative work: removing it would weaken the claim. The post reasons toward a conclusion rather than simply asserting one.

Requirements:
- Each post should reference specific stats, win/loss records, historical seasons, or tactical details
- Vary the players, teams, and topics (trades, MVP debates, team building, historical comparisons, playoff performance, etc.)
- Vary the length: some 1-2 sentences, some 3-5 sentences
- Write in authentic Reddit voice — informal but substantive
- Do NOT write generic analysis — use specific numbers, names, and claims

Return a JSON array of {n} strings, each being one Reddit post. No other text.
Example format: ["post 1 text", "post 2 text", ...]""",

    "hot_take": """\
Generate {n} realistic r/nba Reddit posts that are clear examples of HOT TAKES.

A hot take is a bold, confident opinion stated without supporting evidence, or with evidence that is vague, cherry-picked, or decorative. The claim stands on its own assertiveness. Hot takes often use charged language (overrated, trash, the GOAT, not even top 5, criminally underrated) and invite disagreement.

Requirements:
- Bold, assertive opinions with little to no real evidence
- Cover a range: player rankings, trade criticism, team criticism, GOAT debates, coaching hot takes
- Vary the length: some are one punchy sentence, some are 2-3 sentences of escalating opinion
- Write in authentic Reddit voice — confident, slightly combative, no hedging
- Do NOT include solid evidence that would make it cross into analysis

Return a JSON array of {n} strings, each being one Reddit post. No other text.
Example format: ["post 1 text", "post 2 text", ...]""",

    "reaction": """\
Generate {n} realistic r/nba Reddit posts that are clear examples of REACTIONS.

A reaction is an immediate emotional response to a specific recent event — a game result, trade announcement, injury report, buzzer beater, bad call, or in-game moment. The post expresses a feeling about what just happened with little to no argumentative content. The trigger is always a specific event.

Requirements:
- Reference specific events: blown leads, surprise trades, injuries, incredible plays, bad calls, overtime wins
- Pure emotion — shock, joy, frustration, disbelief, heartbreak — with little or no argument
- Vary the style: ALL CAPS outbursts, short stunned reactions, stream of consciousness
- Write in authentic Reddit game-thread voice — raw, in-the-moment, unfiltered
- Do NOT include structured arguments or evidence

Return a JSON array of {n} strings, each being one Reddit post. No other text.
Example format: ["post 1 text", "post 2 text", ...]""",
}


def generate_batch(label: str, n: int) -> list[str]:
    """Generate n examples for a given label using Groq."""
    prompt = GENERATION_PROMPTS[label].format(n=n)
    try:
        resp = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.9,    # high temp for variety
            max_tokens=3000,
        )
        raw = resp.choices[0].message.content.strip()
        # Strip markdown code fences if present
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        posts = json.loads(raw)
        if isinstance(posts, list):
            return [str(p).strip() for p in posts if str(p).strip()]
        return []
    except Exception as e:
        print(f"  [generation error] {e}")
        return []


def main():
    print("=== TakeMeter Data Generation ===\n")
    print(f"Generating {PER_LABEL} examples per label ({PER_LABEL * 3} total)...\n")

    all_examples: list[dict] = []

    for label in ["analysis", "hot_take", "reaction"]:
        print(f"Generating {label} examples...")
        label_examples: list[str] = []

        while len(label_examples) < PER_LABEL:
            needed    = PER_LABEL - len(label_examples)
            batch_n   = min(BATCH_SIZE, needed)
            batch     = generate_batch(label, batch_n)
            label_examples.extend(batch)
            print(f"  {len(label_examples)}/{PER_LABEL} generated")
            time.sleep(0.5)

        for text in label_examples[:PER_LABEL]:
            all_examples.append({
                "text":          text,
                "label":         label,
                "groq_prelabel": label,
                "source":        "groq_generated",
                "notes":         "",
            })

        print(f"  ✅ {label}: done\n")

    os.makedirs("data", exist_ok=True)
    with open(OUTPUT_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["text", "label", "groq_prelabel", "source", "notes"]
        )
        writer.writeheader()
        writer.writerows(all_examples)

    print(f"✅ Saved {len(all_examples)} examples → {OUTPUT_PATH}")
    print("""
Label distribution: 100 analysis / 100 hot_take / 100 reaction

Next steps:
  1. Open data/nba_raw_prelabeled.csv and skim every row.
     Remove any that don't clearly fit their label.
     Flag borderline ones in the 'notes' column.
  2. When done: keep only 'text' and 'label' columns → save as nba_labeled.csv.
  3. Upload nba_labeled.csv to Colab Section 1.
""")


if __name__ == "__main__":
    main()
