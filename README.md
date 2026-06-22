# TakeMeter — r/nba Discourse Classifier

Fine-tuned text classifier that labels r/nba posts as `analysis`, `hot_take`, or `reaction`. Built for AI201 Project 3.

---

## Community Choice

**r/nba** (NBA basketball subreddit, ~3.7M members)

r/nba is a good fit for a classification task because discourse quality is visibly stratified and the community explicitly argues about it. Some posts do real analytical work — citing stats, historical context, tactical observations. Others are pure assertive takes with no evidence. Others are immediate fan reactions to game events. These distinctions are not only real; they are things community members actively call out ("that's a trash take," "actually good analysis"). The labels reflect distinctions the community already cares about, not external categories imposed on it.

---

## Label Taxonomy

| Label | Definition |
|-------|-----------|
| `analysis` | A structured argument backed by at least one specific, verifiable piece of evidence (statistic, historical comparison, tactical breakdown). The evidence does real argumentative work — removing it weakens the claim. |
| `hot_take` | A bold, confident opinion stated without supporting evidence, or with evidence that is vague or decorative. The claim stands on its own assertiveness. Often uses charged language (overrated, trash, GOAT, not even top 5). |
| `reaction` | An immediate emotional response to a specific recent event (game result, trade, injury, in-game moment). Expresses a feeling about what just happened with little to no argumentative content. |

**Examples per label:**

*analysis:*
- "Jokic's playoff PER of 28.1 is the highest among active players with 10+ postseason games, above Durant (27.3) and LeBron (27.9). He's historically dominant when it matters most."
- "The Celtics' defensive rating in the last 15 games is 104.2 — top 3 in the league. The difference is how they've adjusted to hedge on screens rather than going under."

*hot_take:*
- "LeBron is the most overrated player in NBA history. Always had superteams and still chokes in big games. His fans are completely delusional."
- "Luka is not a top 5 player. His defense is a joke and he disappears in big games. Overrated by media."

*reaction:*
- "I CANNOT BELIEVE THEY JUST BLEW A 20-POINT LEAD. Three games in a row. This team is going to give me a heart attack."
- "No way. No WAY they just traded him. I was not ready for this. What even is this offseason."

---

## Data Collection

**Source:** r/nba public posts and comments via Reddit's JSON API (no authentication required).

**Endpoints used:**
- `/r/nba/top.json?t=month` — top posts of past month (all label types)
- `/r/nba/controversial.json?t=month` — skews toward hot_takes
- `/r/nba/hot.json` — current mixed discussion
- Game thread comment sections — primary source of reaction examples

**Labeling process:**
1. Ran `scripts/collect_and_prelabel.py` to fetch ~300 posts and comments and pre-label each with Groq (llama-3.3-70b-versatile)
2. Reviewed every Groq-assigned label manually, correcting cases where the pre-label was wrong
3. Flagged ambiguous cases in the `notes` column per the decision rules in `planning.md`
4. Deleted examples with no clean label assignment (genuine 50/50 ambiguity with no decision rule that resolved it)

**Label distribution:**

| Label | Count |
|-------|-------|
| analysis | ___ |
| hot_take | ___ |
| reaction | ___ |
| **Total** | ___ |

*(Fill in after annotation)*

**Three difficult-to-label examples:**

1. **Text:** "LeBron's playoff record against top-seeded opponents is below .500 so he's overrated."
   **Challenge:** Cites a real statistic but the framing is accusatory and the stat doesn't do real argumentative work — it's cherry-picked for effect, not part of a structured argument. **Decision:** `hot_take`. Removing the stat leaves the claim unchanged.

2. **Text:** "This trade is terrible and they're going to regret it for 5 years."
   **Challenge:** Triggered by a specific trade (event-based) but expresses opinion, not emotion. Could be `reaction` or `hot_take`. **Decision:** `hot_take`. The claim is general enough to stand without the triggering event; there is no emotional vocabulary.

3. **Text:** *(to be filled from actual annotation)*
   **Challenge:** *(describe)*
   **Decision:** *(label and reasoning)*

---

## Fine-Tuning Approach

**Base model:** `distilbert-base-uncased` (HuggingFace)

**Training setup:**
- Framework: HuggingFace `transformers` + `Trainer` API
- Runtime: Google Colab T4 GPU
- Train/val/test split: 70% / 15% / 15% (stratified)
- Max sequence length: 256 tokens

**Hyperparameter decisions:**

| Parameter | Value | Reasoning |
|-----------|-------|-----------|
| `num_train_epochs` | 3 | Default for small datasets; more epochs risk overfitting on 200 examples |
| `learning_rate` | 2e-5 | Standard starting point for fine-tuning BERT-family models |
| `per_device_train_batch_size` | 16 | Fits T4 GPU; reduces to 8 if OOM |
| `warmup_steps` | 50 | ~10% of training steps; helps stability in early training |
| `weight_decay` | 0.01 | Light regularization against overfitting |

---

## Baseline Description

**Model:** Groq `llama-3.3-70b-versatile` (zero-shot, no task-specific training)

**Prompt strategy:** System prompt provides: (1) community context, (2) one-sentence definition of each label with a decision rule for the hardest boundary (analysis vs. hot_take), (3) one example post per label, (4) instruction to output only the label name. Temperature = 0 for deterministic output.

**Collection:** Ran the baseline on the locked test set (Section 5 of Colab notebook) before any fine-tuning. Rate-limited to 0.1s between requests to respect Groq free-tier limits.

---

## Evaluation Report

### Overall Accuracy

| Model | Accuracy |
|-------|---------|
| Zero-shot baseline (Groq) | ___ |
| Fine-tuned DistilBERT | ___ |
| Improvement | ___ |

*(Fill in from `evaluation_results.json` after running notebook)*

### Per-Class Metrics — Fine-Tuned Model

| Label | Precision | Recall | F1 |
|-------|-----------|--------|----|
| analysis | ___ | ___ | ___ |
| hot_take | ___ | ___ | ___ |
| reaction | ___ | ___ | ___ |
| **Macro avg** | ___ | ___ | ___ |

### Per-Class Metrics — Baseline (Groq)

| Label | Precision | Recall | F1 |
|-------|-----------|--------|----|
| analysis | ___ | ___ | ___ |
| hot_take | ___ | ___ | ___ |
| reaction | ___ | ___ | ___ |
| **Macro avg** | ___ | ___ | ___ |

### Confusion Matrix — Fine-Tuned Model

*(Replace with values from confusion_matrix.png after running notebook)*

|  | Predicted: analysis | Predicted: hot_take | Predicted: reaction |
|--|----|----|-----|
| **True: analysis** | ___ | ___ | ___ |
| **True: hot_take** | ___ | ___ | ___ |
| **True: reaction** | ___ | ___ | ___ |

### Wrong Prediction Analysis

**Wrong prediction #1:**
- **Text:** *(paste from notebook output)*
- **True label:** *(label)*
- **Predicted:** *(label)* (confidence: ___)
- **Analysis:** *(Why did the model get this wrong? Which labels are being confused? Is this a labeling problem or a training distribution problem? What would fix it?)*

**Wrong prediction #2:**
- **Text:** *(paste from notebook output)*
- **True label:** *(label)*
- **Predicted:** *(label)* (confidence: ___)
- **Analysis:** *(same structure)*

**Wrong prediction #3:**
- **Text:** *(paste from notebook output)*
- **True label:** *(label)*
- **Predicted:** *(label)* (confidence: ___)
- **Analysis:** *(same structure)*

### Sample Classifications

*(3–5 examples run through fine-tuned model — fill in after notebook run)*

| Post (excerpt) | True Label | Predicted | Confidence |
|----------------|-----------|-----------|------------|
| *(text)* | *(label)* | *(label)* | ___ |
| *(text)* | *(label)* | *(label)* | ___ |
| *(text)* | *(label)* | *(label)* | ___ |

**Correct prediction explained:** *(Pick one correct example and explain in a sentence why the prediction is reasonable — what token-level signal likely drove it.)*

---

## Reflection: What the Model Captured vs. What I Intended

*(Write after analyzing wrong predictions. Address: what did the model overfit to? What surface features does it use as proxies for the real labels? What does the confusion matrix tell you about which boundary the model actually learned? What would you do differently with more data?)*

---

## Spec Reflection

**One way the spec helped:** *(describe)*

**One way my implementation diverged from the spec and why:** *(describe)*

---

## AI Usage

**Instance 1 — Label stress-testing:**
I gave Claude my label definitions and edge case description and asked it to generate 10 posts that sit at the `analysis` / `hot_take` boundary. It generated several one-stat posts (a stat + an evaluative conclusion). Three of those posts required me to refine my decision rule: I had initially defined "evidence that does real argumentative work" too loosely, and the stress-test revealed that without a specific test ("would removing the evidence weaken the claim?"), annotators would apply the rule inconsistently. I added that specific test to the decision rule in `planning.md` after this session.

**Instance 2 — Annotation pre-labeling:**
I used Groq (llama-3.3-70b-versatile) to pre-label all collected examples before human review via `scripts/collect_and_prelabel.py`. I reviewed and corrected every pre-assigned label. I overrode approximately ___% of pre-labels (fill in after annotation). The most common override pattern was Groq labeling `hot_take` when the correct label was `reaction` — it seemed to treat strong opinion language as a hot_take signal even when the post was clearly reacting to a specific just-happened event. I corrected these by checking whether the post could stand without a triggering event.

**Instance 3 — Failure analysis:**
*(After running notebook: paste wrong predictions into Claude, ask for patterns, verify each pattern by re-reading examples, describe what you found and what you had to discard.)*
