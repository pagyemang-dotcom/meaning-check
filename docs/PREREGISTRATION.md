# Pre-registration: held-out evaluation of Meaning Check

Fixed on 26 October 2026, before any live reading was taken. Any later change is added below with its date and reason,
and every result is reported, including failures.

## Design

- Messages: the 14 messages in `examples/eval_messages.json` (9 for people, 5 instructions for AI agents), each with
  the sender's meaning. The set was written before any model was run and is not changed afterwards.
- Build panel: 16 readers (`panels.people_panel` or `panels.agent_panel`). Held-out panel: 8 readers from other
  countries (`panels.people_heldout` or `panels.agent_heldout`), never used to build any message.
- Readers: Muse Glimmer 30B, option-letter probabilities in one forward pass, averaged over two option orders.
  Writer: Muse Spark 1.3.
- For each message: the held-out panel reads the original and the final message. A reading counts as understood when
  the intended reading is the reader's most probable one.
- Command: `meaningcheck eval` (output `results/eval.json`).

## Hypotheses and decision rules

- **H1 (main).** The final message is understood by more held-out readers than the original. Supported if the 95%
  bootstrap interval (resampling messages) of the mean gain lies above 10 percentage points.
- **H2 (no harm).** On no more than 2 of the 14 messages does the held-out rate fall.
- **H3 (brevity).** The median number of added statements is at most 2.
- **H4 (reader check).** On at least 90% of readings, at least half of the next-token probability lies on the option
  letters (`letter_mass`); if not, the reader template is reported as a threat to validity.

Messages the writer cannot analyse (invalid output after retries) are reported as skipped, not dropped silently.

## Amendments

- **A1 (27 October 2026, after the run, disclosed).** The registered writer was Muse Spark 1.3. The run used Muse Glimmer
  as the writer, because the Meta Model API account was restricted (a card authorisation failed) and the tool fell back
  to Muse Glimmer as designed. Readers, messages, panels and decision rules were unchanged.
- **A2 (27 October 2026, after the run).** `eval` now records each message's lowest letter mass, so H4 can be tested
  on the next run; H4 is reported as not tested for the first run. An exploratory baseline (`--baseline`) is added;
  it is labelled exploratory wherever reported.
