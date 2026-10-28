# Changes

- **2.1 (28 October 2026).** Every reader card opens: the reader's setup and exact system prompt, what it read, its
  probability for every reading, and its before and after once fixed (arrow keys step through readers). Checks stream
  stage by stage (`/api/check_stream`). One-click scenarios grouped as workplace, words read differently and AI agent
  prompts, with three new workplace scenarios. Evidence tab from the recorded results (`/api/results`) and a How it
  works tab with the architecture and the theorems. `meaningcheck reproduce` recomputes every reported statistic
  offline. Faster: scenarios replay their recorded live run (about two seconds, labelled, with Run it live now);
  Muse Glimmer and Muse Spark read at the same time; the panel and unseen readers, and all candidate wordings, go to the
  GPU in single requests; the word analysis is computed after the result; the reader service never starts a second
  GPU; the page wakes the readers on opening and shows a timer. README: architecture diagram, models used, prompt templates, theory. `requirements.txt`.
- **2.0 (27 October 2026).** Two-model panel: Muse Spark readers give a second opinion on every check; a message counts
  as misread if either model misreads it, and the fix is cross-checked with both. Spending guard for public links.
  Interface: verdict across both models, disagreement banner, before/after toggle, word-level diff of the fix,
  table of wordings tried with both models' scores, downloadable report, evidence panel, Ctrl+Enter.
- **1.7.** Strict scoring (a tie is never understanding), empty Muse Spark replies asked again, Muse Spark retried on
  every call. Independent-model test rerun (the first run is void and kept on record).
- **1.6.** Independent-model test on Muse Spark (`meaningcheck crossmodel`).
- **1.5.** Human-study tooling (form builder, analysis, pre-registration); not run.
- **1.4.** The kept fix is the shortest wording the panel understands, chosen among four candidates.
- **1.3.** Exploratory baseline, Test this wording, diagnosis-led results.
- **1.0 to 1.2.** Muse Glimmer can write; calibrated readings; public web app on Modal; Colab route.
- **0.1 (26 October 2026, from 16:01).** First build.
