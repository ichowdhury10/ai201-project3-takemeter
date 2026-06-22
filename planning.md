# TakeMeter — Planning Document
### AI201 · Project 3 · r/nba Discourse Classifier

> Written before data collection, per project requirements.
> Updated before starting stretch features.

---

## Community

**r/nba** — NBA basketball subreddit, ~3.7M members.

r/nba is a good fit for this task for three reasons. First, discourse quality is visibly stratified: some posts do real analytical work with statistics and historical context; others are pure assertive takes ("this player is trash / the GOAT"); others are immediate fan reactions to games. Second, community members actively argue about discourse quality — calling out "trash takes" or praising "actually good analysis" is its own recurring meta-conversation. The labels will reflect distinctions the community itself cares about, not distinctions imposed from outside. Third, the text is reliably English, sentence-length or longer, and centered on a narrow enough domain (NBA basketball) that label application stays tractable.

---

## Labels

Three labels, designed around the dominant modes of r/nba discourse:

### `analysis`
A post that makes a structured argument supported by at least one specific, verifiable piece of evidence — a statistic, historical comparison, tactical breakdown, or direct player/team comparison. The evidence does real argumentative work: removing it would weaken or eliminate the claim. Analysis posts reason toward a conclusion rather than simply asserting one.

**Example 1 (clearly analysis):**
> "People are sleeping on how dominant Jokic has been in the playoffs. His playoff PER is 28.1 — highest among active players with 10+ postseason games, above Durant's 27.3 and LeBron's 27.9. He's not just good; he's historically good in the moments that matter."

**Example 2 (clearly analysis):**
> "The Celtics' defensive rating in the last 15 games is 104.2 — top 3 in the league. The difference is how they've adjusted to hedge on screens rather than going under. Brown and Tatum are recovering faster on the perimeter and it's showing in opponent FG% on mid-range jumpers."

---

### `hot_take`
A bold, confident opinion stated without supporting evidence, or with evidence that is vague, cherry-picked, or decorative (present to sound credible but not doing real argumentative work). Hot takes often use charged language ("overrated," "trash," "not even top 5," "the GOAT") and invite disagreement. The claim stands on its own assertiveness; removing any evidence present would not meaningfully weaken it.

**Example 1 (clearly hot_take):**
> "LeBron is the most overrated player in NBA history. The man has always had superteams and still chokes when it matters. His fans are delusional."

**Example 2 (clearly hot_take):**
> "Luka is not a top 5 player. I don't care what anyone says. His defense is a joke and he disappears in big games. Overrated by media."

---

### `reaction`
An immediate emotional response to a specific recent event — a game result, trade announcement, injury report, draft pick, or in-game moment. Reaction posts express a feeling about what just happened with little to no argumentative content. The trigger is a specific event, not a general opinion about a player or team.

**Example 1 (clearly reaction):**
> "I CANNOT BELIEVE THEY JUST BLEW A 20-POINT LEAD. Three games in a row. This team is going to give me a heart attack."

**Example 2 (clearly reaction):**
> "No way. No WAY they just traded him. I was not ready for this. What even is this offseason."

---

## Hard Edge Cases

**The hardest anticipated boundary: `analysis` vs. `hot_take` when one stat is present.**

The archetypal ambiguous case:
> "LeBron's playoff record against top-seeded opponents is below .500 so he's overrated."

This post cites a specific statistic but the framing is accusatory and the stat is selected for effect, not as part of a genuine argument. The stat alone doesn't establish the claim — "overrated" requires a benchmark (compared to what? other players in the same matchup situations?) that the post doesn't provide. The post uses the stat as decoration.

**Decision rule:** If the post provides evidence that would support the claim even if you removed the charged opinion language, label it `analysis`. If the evidence is cherry-picked, vague, or doesn't do the argumentative work needed to establish the claim, label it `hot_take`. For the example above: removing "so he's overrated" leaves a bare stat with no claim — the stat is not doing the work. → `hot_take`.

**Secondary edge case: `reaction` vs. `hot_take` for event-triggered opinions.**

> "This trade is terrible and they're going to regret it for 5 years."

This reacts to a specific event (a trade) but the claim is opinion, not emotion. There is no stat, no argument — but also no emotional vocabulary. Apply this rule: if the post could have been written without a triggering event (i.e., the opinion is general enough to stand alone), it is `hot_take`. If the content only makes sense as a response to something that just happened and primarily expresses feeling ("I can't believe," "this hurts," "I'm in shock"), it is `reaction`. The trade example is `hot_take`: "this trade is terrible" is a general evaluative claim, not an emotional reaction to the event.

**Third edge case: long game-thread comment that contains both reaction and analysis.**

A comment that opens with "I can't believe that call" and then spends 4 sentences analyzing the team's defensive scheme should be labeled by dominant mode. If the analytical content is longer and more substantive than the emotional opener, label `analysis`. If the emotional content is dominant and the analytical content is one line tacked on, label `reaction`.

---

## Data Collection Plan

**Source:** r/nba public posts and comments via Reddit's JSON API (unauthenticated, no scraping library required).

**Endpoints:**
- `/r/nba/top.json?t=month` — top posts of the past month, good for all types
- `/r/nba/controversial.json?t=month` — controversial posts, skews toward hot_takes
- `/r/nba/hot.json` — current discussion, mixed types
- Game thread comment sections — best source of `reaction` examples

**Target distribution:** 75 examples per label (225 total, ~12.5% buffer over the 200 minimum). This keeps no label above 33% of the dataset.

**Underrepresentation fallback:** If `analysis` falls below 60 examples after initial collection, search for posts in weekly discussion threads and "film breakdown" / "statistical deep dive" flairs. If `reaction` falls short, pull more comments from game thread sections (those threads reliably produce reaction-mode text). If `hot_take` falls short, pull from controversial sort and post titles.

**Text extraction:** For posts, use title + selftext combined. Posts with no selftext (link posts or image posts with no body) are excluded unless the title alone is substantive (≥ 60 characters). For comments, use the comment body directly.

**Minimum text length:** 40 characters. Posts shorter than this are excluded — too little signal for either the model or the human annotator to make a reliable judgment.

**CSV format:**
| Column | Description |
|--------|-------------|
| `text` | The post/comment text (title + selftext for posts, body for comments) |
| `label` | Final human-reviewed label: `analysis`, `hot_take`, or `reaction` |
| `groq_prelabel` | Groq's pre-assigned label (before human review) |
| `notes` | Free text — record any annotation difficulty here |

---

## Evaluation Metrics

**Metrics I will report:**
1. **Overall accuracy** — for both the baseline and fine-tuned model
2. **Per-class precision, recall, and F1** — for all three labels, both models
3. **Macro-F1** — average F1 across classes, treating each class equally
4. **Confusion matrix** — rows = true label, columns = predicted label; reveals directional errors

**Why accuracy alone is not enough:**

If `hot_take` accounts for 40% of the test set, a model that predicts `hot_take` every time achieves 40% accuracy — which would outperform a random guesser but tells us nothing useful. Macro-F1 catches this: a model with degenerate behavior on one class will have a near-zero F1 for the ignored classes, dragging the macro average down sharply. Per-class F1 is the diagnostic: it shows exactly which label the model fails on and in which direction (high precision/low recall = model is conservative; high recall/low precision = model over-predicts that class).

For this specific task, the `analysis` vs. `hot_take` boundary is the hardest. I expect the model to have lower recall for `analysis` than for `hot_take`, because analysis posts with brief evidence may look like hot_takes at the token level. Per-class recall for `analysis` is the metric I will watch most closely.

---

## Definition of Success

A classifier is useful for this task if a community moderator or regular user could trust its output as a first-pass filter — not perfectly, but reliably enough to prioritize which posts to surface or flag.

**Specific thresholds:**
- Macro F1 ≥ 0.70: model is learning all three distinctions meaningfully
- Fine-tuned model beats the Groq baseline by ≥ 0.08 on macro F1: fine-tuning added real value
- No per-class F1 below 0.55: model isn't completely failing on any single class
- Baseline unparseable responses < 5% of test set: prompt is clean enough to evaluate fairly

If the fine-tuned model does not beat the baseline by ≥ 0.08, I will investigate whether the test labels are consistent with the training labels (annotation drift), whether the training set has imbalance problems, or whether the boundary between `analysis` and `hot_take` is too fuzzy to learn from 200 examples.

---

## AI Tool Plan

**1. Label stress-testing (before annotation)**

I will give Claude my label definitions and edge case description and ask it to generate 10 posts that sit at the boundary between `analysis` and `hot_take`. If it produces posts I cannot cleanly classify using my decision rules, the definitions need tightening before I annotate 200 examples. I will document which boundary posts required rule revisions.

**2. Annotation assistance (during data collection)**

I will use Groq (llama-3.3-70b-versatile) to pre-label all collected posts before human review. The pre-labeling prompt includes my full label definitions and one example per label. I will review and correct every pre-assigned label — the Groq output is a starting point, not a final answer. Every label I override will be recorded in the `notes` column with a brief explanation. I estimate overriding approximately 15–25% of pre-labels based on the edge cases described above.

I will disclose AI pre-labeling in the README's AI usage section, including the proportion of labels I overrode and what patterns triggered those overrides.

**3. Failure analysis (after running the notebook)**

After collecting wrong predictions from Section 4 of the notebook, I will paste them into Claude and ask it to identify systematic patterns — shared lexical features, post length, sarcasm markers, label pairs that keep getting confused. I will then verify each proposed pattern by re-reading the examples myself. Any pattern I cannot verify by reading the data I will not report. The goal is to surface patterns I might miss reviewing 20 wrong predictions one at a time, not to outsource the analysis.

---

## Stretch Features (updated before starting each)

*This section will be updated before starting any stretch feature.*

- [ ] Inter-annotator reliability
- [ ] Confidence calibration
- [ ] Error pattern analysis
- [ ] Deployed interface
