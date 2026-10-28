# Pre-registration: independent-model test

Fixed on 27 October 2026, before any Muse Spark reading was taken. Changes are added below with date and reason;
every result is reported.

## Question
Every reader so far is Muse Glimmer. Do the panel's flags and the fixes hold for a different model that took no part
in building any fix?

## Design
- Messages, options and fixes: the 14 messages of the evaluation, with the readings and final wordings saved in
  `results/eval.json` by version 1.5 (`meaningcheck eval --baseline`).
- Readers: Muse Spark 1.3 through the Meta Model API, given the 8 held-out backgrounds. Muse Spark returns no token
  probabilities, so each reading is asked 3 times in each of two option orders, and the chosen letters are counted.
- Each reader reads the original; for the 8 changed messages also the fix and the plain-rewrite baseline.
- Command: `meaningcheck crossmodel` (output `results/crossmodel.json`). About 1,440 calls.

## Hypotheses and decision rules
- **HM1 (the fix transfers).** Over the changed messages, the share of Muse Spark readers who understand the fix minus
  the share who understand the original is above 0: supported if the lower end of the 95% bootstrap interval
  (resampling messages) is above 0.
- **HM2 (the flags transfer).** On original wordings, the share understood for messages the Glimmer panel passed minus
  the share for messages it flagged is above 0, by the same rule (resampling messages within each group).
- Exploratory only: fix minus plain-rewrite baseline, and the share of answers in which a letter could be found.

## Amendments
- **C1 (27 October 2026, after the first run; disclosed).** The first run (`results/crossmodel_run1_VOID_tie_bias.json`)
  is void. Two defects made it unreliable: (1) when Muse Spark gave no usable letter in any of its answers, all options
  got equal probability and the tie was broken in favour of the intended reading, which counted a non-answer as
  understanding (in 5 of 14 messages at least one reader had no usable answer on the original wording); (2) empty
  replies were not asked again. Fixes: a reading counts as understood only when the intended reading is strictly the
  most probable (a tie never counts), an empty or letterless reply is asked again up to twice, and readers with no
  usable answer are counted and reported. The design, messages, readers and decision rules are unchanged; the test is
  rerun with a fresh cache. Its first-run numbers (HM1 +52 points [25, 79], HM2 +20 [-18, 57]) are kept for the
  record only.
