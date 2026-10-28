# Pre-registration: human study

Fixed on 27 October 2026, before the form existed and before any response. Changes are added below with date and
reason; every result is reported.

## Question
Do people misread the messages the reader panel flags, and does Meaning Check's fix help people, not only AI readers?

## Design
- Messages: the 14 evaluation messages (`examples/eval_messages.json`); options: the readings the writer produced in
  the evaluation run, shown in a fixed shuffled order; fixes: version 1.4's final wordings (`results/eval.json`).
- Each participant sees every message once. For the 8 messages the tool changed, a participant sees either the original
  or the fix; form A shows the fix for half of them and form B for the other half. Participants are routed by birth
  month (odd: A, even: B), so each contributes to both conditions. Unchanged messages appear as written.
- Participants: adults reached through the author's networks (AIMS and colleagues in several countries), anonymous;
  only country of upbringing and first language are asked. Target 20 or more; analysis runs on whatever has arrived
  by the cut-off (Thursday 29 October 2026, 18:00 London), and the number is reported.
- A response counts as understood when the chosen option is the sender's intended reading.

## Hypotheses and decision rules
- **HH1 (the fix helps people).** Within participants, the share understood on fixed messages minus the share on
  original messages is above 0: supported if the lower end of the 95% bootstrap interval (resampling participants)
  is above 0.
- **HH2 (the panel's flags match people).** On original wordings, the share understood on messages the panel passed
  minus the share on messages it flagged is above 0, by the same rule.
- Exploratory only: results by country of upbringing, per-message rates, and agreement with the AI readers.

## Commands
`meaningcheck human-form` writes the form builder (`results/human_form.gs`) and the item file
(`results/human_items.json`); `meaningcheck human-analyse responses.csv` writes `results/human_results.json`.

## Amendments
(none yet)
