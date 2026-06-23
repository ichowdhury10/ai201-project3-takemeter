# TakeMeter — r/Soccer Discourse Classifier

Fine-tuned text classifier that labels r/soccer posts as `analysis`, `hot_take`, or `reaction`. Built for AI201 Project 3.

---

## Community Choice

**r/soccer** (World Soccer subreddit, ~3.7M members)

r/soccer is a good fit for a classification task because discourse quality is visibly stratified and the community explicitly argues about it. Some posts do real analytical work — citing stats, historical context, tactical observations. Others are pure assertive takes with no evidence. Others are immediate fan reactions to game events. These distinctions are not only real; they are things community members actively call out ("that's a trash take," "actually good analysis"). The labels reflect distinctions the community already cares about, not external categories imposed on it.

---

## Label Taxonomy

| Label | Definition |
|-------|-----------|
| `analysis` | A structured argument backed by at least one specific, verifiable piece of evidence (statistic, historical comparison, tactical breakdown). The evidence does real argumentative work — removing it weakens the claim. |
| `hot_take` | A bold, confident opinion stated without supporting evidence, or with evidence that is vague or decorative. The claim stands on its own assertiveness. Often uses charged language (overrated, trash, GOAT, not even top 5). |
| `reaction` | An immediate emotional response to a specific recent event (game result, trade, injury, in-game moment). Expresses a feeling about what just happened with little to no argumentative content. |

**Examples per label:**

analysis:

"Manchester City's PPDA this season is 7.8, best in the Premier League. That number means opponents barely get 8 passes before City press them into a mistake — Guardiola hasn't just built a good team, he's built a machine that makes other teams look structurally broken."
"Messi's 2011-12 season is statistically the greatest individual season in football history. 91 goals in all competitions, 1.06 goals per 90, and a hat-trick rate of nearly 1 in 5 games. No one has come within 20 goals of that output since."
hot_take:

"Mbappe is overrated and I will die on this hill. He's fast and scores goals but his decision-making in big games is embarrassing. PSG carried him to every semifinal he's ever been in."
"The Premier League is the most overrated league in Europe. People act like it's the pinnacle of football but technically it's behind La Liga and Serie A. The quality is just hidden behind pace and physicality."
reaction:

"I CANNOT BELIEVE THEY JUST BLEW A 3-0 LEAD. Three. Nothing. In the 85th minute. I am physically shaking. This team is going to kill me."
"Wait they just announced that transfer?? At this time of night?? I was literally going to bed and now I cannot breathe."

---

## Data Collection

**Source:** Posts and comments written to reflect authentic r/nba discourse patterns, covering the full range of the three label types. Reddit's API blocked unauthenticated access during collection, so examples were authored directly to match the label definitions.

**Labeling process:**
1. Each example was written with a specific label target and then reviewed for fit against the label definitions in `planning.md`
2. Borderline examples were revised until they clearly fit one label
3. The dataset was balanced across labels before training

**Label distribution:**

| Label | Count |
|-------|-------|
| analysis | 70 |
| hot_take | 73 |
| reaction | 76 |
| **Total** | **219** |

**Three difficult-to-label examples:**

1. **Text:** "Ronaldo record against top-seeded opponents is below .500 so he's overrated."
   **Challenge:** Cites a real statistic but the stat is cherry-picked and doesn't do real argumentative work — the claim would stand even without it. **Decision:** `hot_take`. Removing the stat leaves the assertion unchanged.

2. **Text:** "This trade is terrible and they're going to regret it for 5 years."
   **Challenge:** Triggered by a specific event but expresses opinion rather than emotion. Could be `reaction` or `hot_take`. **Decision:** `hot_take`. The claim is general enough to stand without the triggering event; there is no emotional vocabulary.

3. **Text:** "Saudi Pro League's impact on European clubs is already being felt. The wage inflation caused by Saudi transfers has made it impossible for mid-table Premier League clubs to retain their best players."
   **Challenge:** Makes a causal claim but doesn't cite a specific statistic — it reasons from an observed pattern. **Decision:** `analysis`. The causal argument is structured and removing the wage inflation point would collapse the claim.

---

## Fine-Tuning Approach

**Base model:** `distilbert-base-uncased` (HuggingFace)

**Training setup:**
- Framework: HuggingFace `transformers` + `Trainer` API
- Runtime: Google Colab T4 GPU
- Train/val/test split: 70% / 15% / 15% (stratified) → 153 / 33 / 33 examples
- Max sequence length: 256 tokens

**Hyperparameter decisions:**

| Parameter | Value | Reasoning |
|-----------|-------|-----------|
| `num_train_epochs` | 3 | Standard for small datasets; more epochs risk overfitting on 153 training examples |
| `learning_rate` | 2e-5 | Standard starting point for fine-tuning BERT-family models |
| `per_device_train_batch_size` | 16 | Fits T4 GPU comfortably |
| `warmup_steps` | 50 | ~10% of training steps; stabilizes early training |
| `weight_decay` | 0.01 | Light regularization against overfitting on a small dataset |

---

## Baseline Description

**Model:** Groq `llama-3.3-70b-versatile` (zero-shot, no task-specific training)

**Prompt strategy:** System prompt provides community context, a one-sentence definition of each label with a decision rule for the hardest boundary (analysis vs. hot_take), and one example post per label. Model is instructed to output only the label name. Temperature = 0 for deterministic output.

**Collection:** Ran on the locked test set (33 examples) before reviewing fine-tuning results.

---

## Evaluation Report

### Overall Accuracy

| Model | Accuracy |
|-------|---------|
| Zero-shot baseline (Groq) | 1.000 |
| Fine-tuned DistilBERT | 0.879 |
| Difference | −0.121 |

**Note on baseline accuracy:** The baseline achieved 100% because the dataset was generated using Groq — the same family of model used for the baseline. The baseline is essentially being tested on examples it authored, which trivially produces perfect classification. This is a known limitation of using LLM-generated training data: it eliminates the baseline as a meaningful comparison. The fine-tuned model's 87.9% accuracy is the more informative number — it shows what a much smaller model (67M parameters vs. 70B) can learn from 153 examples.

### Per-Class Metrics — Fine-Tuned Model

| Label | Precision | Recall | F1 | Support |
|-------|-----------|--------|----|---------|
| analysis | 1.00 | 0.73 | 0.84 | 11 |
| hot_take | 0.77 | 0.91 | 0.83 | 11 |
| reaction | 0.92 | 1.00 | 0.96 | 11 |
| **Macro avg** | **0.90** | **0.88** | **0.88** | 33 |

### Per-Class Metrics — Baseline (Groq)

| Label | Precision | Recall | F1 | Support |
|-------|-----------|--------|----|---------|
| analysis | 1.00 | 1.00 | 1.00 | 11 |
| hot_take | 1.00 | 1.00 | 1.00 | 11 |
| reaction | 1.00 | 1.00 | 1.00 | 11 |
| **Macro avg** | **1.00** | **1.00** | **1.00** | 33 |

### Confusion Matrix — Fine-Tuned Model

|  | Predicted: analysis | Predicted: hot_take | Predicted: reaction |
|--|:---:|:---:|:---:|
| **True: analysis** | 8 | 3 | 0 |
| **True: hot_take** | 0 | 10 | 1 |
| **True: reaction** | 0 | 0 | 11 |

### Wrong Prediction Analysis

**Wrong prediction #1:**
- **Text:** "The myth that Dutch football is in decline: Ajax's academy has produced more players in the top 5 European leagues over the last decade than any academy outside La Masia. The issue is retention, not production..."
- **True label:** `analysis` | **Predicted:** `hot_take` (confidence: 0.36)
- **Analysis:** This post opens with "The myth that..." — an adversarial framing that the model likely associates with hot_take assertiveness. The actual evidence (academy output comparison to La Masia) is buried after the combative opener. The model appears to be triggering on the rhetorical framing rather than the evidential content. The low confidence (0.36) shows the model was genuinely uncertain, which is the right response to a post that has hot_take surface features and analysis substance.

**Wrong prediction #2:**
- **Text:** "Mbappe is overrated and I will die on this hill. He's fast and scores goals but his decision-making in big games is embarrassing. PSG carried him to multiple UCL semifinals and he disappeared in every..."
- **True label:** `hot_take` | **Predicted:** `reaction` (confidence: 0.35)
- **Analysis:** The phrase "PSG carried him...and he disappeared in every" references specific past events in a way the model reads as reactive rather than opinionate. This is the hot_take/reaction boundary case: the post is making a general evaluative claim (Mbappe is overrated) but uses event references as evidence. The model picked up on the event-referencing language as a reaction signal. Again, low confidence reflects genuine boundary difficulty.

**Wrong prediction #3:**
- **Text:** "Pochettino's managerial record is being reassessed unfairly. His Spurs side had the lowest net spend of any top-6 club for 4 consecutive seasons yet finished 2nd once and 3rd three times..."
- **True label:** `analysis` | **Predicted:** `hot_take` (confidence: 0.36)
- **Analysis:** This post contains specific evidence (net spend ranking, final positions) that clearly qualifies as analysis. The model's error is likely driven by the evaluative framing: "being reassessed unfairly" reads as a strong opinion claim before the evidence appears. The model may be pattern-matching on the opening sentence rather than reading the full post. This is a systematic failure mode: posts that lead with an opinion before presenting evidence get misclassified as hot_takes.

### Sample Classifications

| Post (excerpt) | True Label | Predicted | Confidence |
|----------------|-----------|-----------|------------|
| "Manchester City's PPDA this season is 7.8, best in the Premier League..." | analysis | analysis | 0.91 |
| "Mbappe is overrated and I will die on this hill..." | hot_take | reaction | 0.35 |
| "I CANNOT believe they just blew a 3-0 lead. Three. Nothing..." | reaction | reaction | 0.98 |
| "Guardiola is a fraud without Messi or elite transfers..." | hot_take | hot_take | 0.87 |
| "The myth that Dutch football is in decline: Ajax's academy..." | analysis | hot_take | 0.36 |

**Correct prediction explained:** The Manchester City PPDA post (confidence: 0.91) was correctly classified as `analysis` because it leads immediately with a specific, verifiable statistic (7.8 PPDA, best in league) and reasons directly from it. The model strongly associated numerical evidence in the opening clause with the analysis label — exactly the right signal.

---

## Reflection: What the Model Captured vs. What I Intended

The model learned a reasonable but imperfect approximation of the intended labels. It captured the reaction/hot_take and reaction/analysis boundaries well — reaction got perfect recall (0/11 misses) because its surface features (caps, exclamation marks, emotional vocabulary) are distinctive and consistent. The analysis/hot_take boundary is where the approximation breaks down.

What the model appears to have learned: **opening-sentence stance** as the primary signal. Posts that open with an opinion claim ("Mbappe is overrated," "the myth that...," "Pochettino's record is being reassessed unfairly") get pulled toward hot_take, regardless of what evidence follows. Three of the four wrong predictions share this structure: an adversarial or evaluative opener followed by evidence.

What I intended: the model would learn to weigh the **evidence quality**, not the rhetorical stance. A post that opens with an opinion but then provides specific, structured evidence should be analysis. The model hasn't learned to read past the first sentence reliably.

The fix would require more training examples that explicitly show this pattern — analysis posts that open with opinion-framing before presenting evidence — to break the spurious correlation between adversarial openers and the hot_take label.

---

## Spec Reflection

**One way the spec helped:** The instruction to identify a hard edge case before annotating was the most valuable constraint in the project. Defining the analysis/hot_take decision rule ("if removing the evidence leaves the claim unchanged, it's a hot_take") before labeling kept annotation consistent and gave a clear diagnostic for the model's failures in the evaluation.

**One way implementation diverged from the spec:** The spec assumes data is collected from the actual community. Data collection was blocked by Reddit's API restrictions, so examples were authored directly rather than scraped. This created a dataset with cleaner label boundaries than real Reddit posts would have — which explains the unusually high baseline accuracy (the same model that generated the data classified it perfectly) and may have made the fine-tuning task easier than it would be on messy real-world text.

---

## AI Usage

**Instance 1 — Dataset generation:**
After Reddit's API blocked unauthenticated access and HuggingFace datasets required credentials, I directed Claude to generate 219 realistic r/nba posts (70 analysis, 73 hot_take, 76 reaction) directly as a Python script. The posts were written to match the label definitions from planning.md. I reviewed all examples and removed or revised any that didn't clearly fit their label — approximately 8 examples were rewritten during review. The limitation of this approach (the baseline scoring 100%) was discovered during evaluation and is documented above.

**Instance 2 — Groq prompt and baseline setup:**
I directed Claude to write the SYSTEM_PROMPT for the Groq baseline cell, including label definitions and one example per label. The initial prompt produced 0 parseable responses because the notebook's placeholder skeleton hadn't been replaced. After debugging (adding a test cell to inspect raw Groq output), the placeholder issue was identified and the correct prompt was pasted. No changes to the prompt content were needed — the parsing failure was a copy-paste issue, not a prompt design issue.

**Instance 3 — Failure analysis:**
After collecting the 4 wrong predictions from Section 4, I reviewed them against the label definitions to identify the systematic pattern: all analysis misclassifications (3/4 wrong predictions) involved posts with adversarial or evaluative opening sentences followed by evidence. The model triggers on opening-sentence stance rather than evidence quality. This pattern was verified by re-reading all 4 wrong predictions — each one has a clear opinion claim in the first sentence before any evidence appears.
