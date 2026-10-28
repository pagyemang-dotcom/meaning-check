# Meaning Check
**Try it live: https://pagyemang-dotcom--meaning-check-web-web.us-east.modal.run/** (scenarios open instantly; a new message wakes the GPU readers, which takes a few minutes after a pause).

**Version 2.1: click any reader to see exactly how it read your message.** Every message is read by two Meta models.** Muse Glimmer readers (16) and Muse Spark readers (8) read
your message; it counts as misread if either model's readers misread it, and the fix is the shortest wording both
understand. The independent test below showed why: one model's readers miss misreadings another model makes.

**See who would misread your message, and what they would take it to mean. Then fix it with the fewest words.**

The same words are understood differently by readers with different backgrounds, and by AI agents with different
settings. "Please table this item" means *discuss it now* in London and *put it off* in New York; "close it if it's
resolved" leaves an agent to decide who judges "resolved". Meaning Check has a panel of readers read your message,
shows who misreads it and how, adds the shortest set of clarifying statements that every reader understands, and
tests the result on readers it never used.

## Result

- **Pre-registered (Muse Glimmer readers never used to build the fixes):** understanding rose from 52.7% to 100%
  across the 14 test messages (gain 47.3 points, 95% interval 21.4 to 71.4); no message got worse.
- **Pre-registered, independent model (Muse Spark, which built nothing):** the fixes transfer, +53.6 points [25.0,
  85.7]. The panel's flags do not fully transfer: Muse Spark misread "biweekly", "revert", "bring a plate" and "delete
  old files", which every Muse Glimmer reader understood. One model's readers miss misreadings another model makes.
- A plain "make it clear" rewrite does about as well on these tests (exploratory). The evidence supports Meaning Check
  as a check before sending: who misreads, what they take it to mean, and whether a wording survives readers it was
  not built for.

Details, caveats, a void run and every amendment: `RESULTS.md`.

![How a check works: your message goes to 16 Muse Glimmer readers and 8 Muse Spark readers; an exact solver finds the fewest facts to add; the writer drafts candidate wordings; both panels re-read every candidate; the shortest wording both understand is sent.](docs/architecture.svg)

## Try it in one click

Open the web app and pick a scenario: a deadline at risk, feedback to a colleague, a status update, "table this item",
a date written 03/04, "biweekly" (where the two Meta models disagree), or an instruction to an AI agent. Readers fill
in as they finish (the check streams). **Click any reader** to open its card: where it lives and what it does, the exact
system prompt it was given, what it read, the probability it gave every possible reading (with "you meant" and "acts on
this" marked), and, after the fix, its before and after side by side. Scenarios open in about two seconds: the first live run of each is recorded and replayed, labelled as a recorded live
run with a **Run it live now** button (Colab step 7b records them all). Your own messages are always checked live: the
page starts waking the GPU as soon as it opens and shows a timer; the two models read at the same time, every set of
wordings goes to the GPU in one request, and the word analysis arrives after the result. The **Evidence** tab shows every recorded test
message, and the wordings the tool tried for it; **How it works** shows the pipeline and the theory.

## Meta models used

| Role | Model | How it is reached | What it returns |
| --- | --- | --- | --- |
| Readers (16 panel + 8 held-out) | **Muse Glimmer 30B** (`meta-models/Muse-Glimmer-30B`, open weights) | Our own service on Modal, one A100 80 GB (`deploy/modal_glimmer.py`) | Next-token probability of each option letter |
| Second-opinion readers (8) | **Muse Spark** | Meta Model API (`https://api.meta.ai/v1`) | Sampled answers, two option orders |
| Writer: readings, statements, candidate wordings, the one question | **Muse Spark**, falling back per call to Muse Glimmer | Meta Model API / the Modal service | JSON |
| Photo and voice input | **Muse Spark** vision, **Muse Voice Transcribe** | Meta Model API | Text |

No other model is used anywhere.

### Prompt templates

Each reader is the same model given a different system prompt (`meaningcheck/core.py`, `Persona.system_prompt`):

```text
People:    You live in Ghana. You work in an office and read a lot of workplace email. You read messages the way
           people around you usually would.
AI agents: You are an AI assistant deployed by a team in the United Kingdom. You handle customer-support tickets.
           You receive instructions from your operator and act on what you believe they want.
```

Panel: the United Kingdom, the United States, Ghana and India; two backgrounds; reasoning "low" (quick) or "high"
(careful). Held-out readers come from other countries and are never used to build a fix (`meaningcheck/panels.py`).
The reading question (`meaningcheck/reader.py`), asked twice with the options in opposite orders:

```text
Message you received:
"""
<the message>
"""

What does the sender most likely want you to understand?

A) <reading 1>
B) <reading 2>
...

Answer with the letter of one option.
```

For AI agents the head is "Instruction from your operator: ... Which of these will you do?". Muse Glimmer's answer is
read after the prefix `Answer: (`, chosen at start-up as the prefix that puts the most probability on the letters. A
reader **understands** only when the intended reading is strictly the most probable; a tie counts as a misreading.

## The theory

From *A Mathematical Theory of Being Understood* (Paul Agyemang, AIMS Ghana; under review at JAIR).

- A message is a set of statements *m* about the intended reading *w*. Stating attribute *a* rules out the readings
  that differ from *w* on *a*: the exclusion set *E_a(w)*.
- A reader π acts on the most plausible reading not ruled out. Its rivals, *Dom_π(w)*, are the readings it finds at
  least as plausible as *w*.
- **Audience theorem.** Every reader in panel *R* understands *m* if and only if
  ⋃_{a∈m} *E_a(w)* ⊇ ⋃_{π∈R} *Dom_π(w)*. The shortest such *m* is a minimum set cover, NP-hard in general and small in
  practice; `meaningcheck/cover.py` solves it exactly as a 0-1 integer programme (HiGHS) and reports optimality.
- **Sampling theorem.** If all *T* sampled readers understand a message built from *A* candidate statements, then with
  95% confidence a new reader from the same population misreads it with probability at most (*A* ln 2 + ln 20) / *T*.
- **Price of context.** When no single message reaches every reader, readers are grouped so each group has one that
  works, and one question tells the groups apart.

## Reproduce every number

```bash
pip install -r requirements.txt && pip install -e .
meaningcheck reproduce        # offline: recomputes every statistic and interval from results/*.json
```

It recomputes the summaries and bootstrap intervals from the per-message rows, checks them against the recorded
summaries, checks the figures quoted in `RESULTS.md`, and checks the web app shows the same files. Re-running the
models: `meaningcheck eval` and `meaningcheck crossmodel` (every reading is cached in `.mc_cache/`).

## What was built during the hackathon (all of this repository)

Everything in this repository was written during the Meta Global AI Developer Hackathon, from 26 October 2026.
The method rests on prior work by the author, *A Mathematical Theory of Being Understood* (under review at the Journal
of Artificial Intelligence Research), and on its pre-registered experiment with small open language models. No code
from that study is reused here.

| Part | File |
| --- | --- |
| Reader panel: option-letter probabilities, averaged over two option orders | `meaningcheck/reader.py`, `panels.py` |
| Muse Glimmer service on Modal: calibrated letter readings, written-answer fallback, writing | `deploy/modal_glimmer.py`, `meaningcheck/glimmer_service.py` |
| Muse Spark writer: readings, statements, rewrite, clarifying question; vision and voice input | `meaningcheck/writer.py`, `backends.py` |
| Exact shortest fix (integer programme, HiGHS) with optimality proof; verify-and-repair loop | `meaningcheck/cover.py`, `engine.py` |
| One question when no wording works (grouping readers) | `meaningcheck/engine.py` |
| Held-out readers and the sampling guarantee | `meaningcheck/engine.py`, `cli.py` |
| Which words cause the misreading | `meaningcheck/engine.py` |
| Web app: streaming reader grid, reader cards you can open, evidence and how-it-works tabs | `meaningcheck/server.py`, `web/index.html` |
| Offline reproduction of every reported statistic | `meaningcheck reproduce`, `meaningcheck/reproduce.py` |
| CI check for policies and prompts | `meaningcheck lint` |
| Tests: simulated readers, the service on a tiny Muse Glimmer-architecture model on CPU, a stand-in Meta API | `tests/` |

## How it works

1. **Readings.** Muse Spark lists what the message could be taken to mean (the first is what you mean) and the
   questions on which those readings differ, each with a one-line statement of your answer.
2. **Panel.** Sixteen readers read the message: Muse Glimmer placed in the UK, the US, Ghana and India, with two
   backgrounds and two reasoning settings (quick and careful). For AI-agent prompts the readers are agents deployed in
   those countries doing support or operations work. Each reader gives a probability for every reading, read from the
   option letters in one forward pass and averaged over two option orders to remove position bias.
3. **Who misreads.** Each reader acts on its most probable reading. The grid shows each misreader's reading.
4. **Say enough.** A message is understood by every reader exactly when its statements rule out every reading some
   reader finds at least as plausible as yours (the audience theorem). The fewest such statements are found by an
   exact 0-1 integer programme; the solver proves no shorter set exists.
5. **Verify and repair.** The panel re-reads the message with the statements. AI readers sometimes misread even a full
   description, so a reader that still misreads adds its new rivals, and the programme is solved again.
6. **Rewrite.** Muse Spark turns the statements into one natural message, which is kept only if the panel understands
   it at least as well as the plain version.
7. **One question.** If no single message reaches every reader, readers are grouped so each group has a message that
   works, and Muse Spark gives the one question that tells the groups apart (the price-of-context theorem).
8. **Unseen readers.** Eight readers from other countries, never used while building, read the result. With 95%
   confidence a fresh reader from the panel's population misreads a message the whole panel understood with probability
   at most (A ln 2 + ln 20) / 16, where A is the number of candidate statements (the sampling theorem).

Why two models: the Muse Spark API does not return token probabilities, which the readers need, so readers run on the
open-weight Muse Glimmer; Muse Spark does the writing. If the Muse Spark API is unavailable (no key, account
restricted, credit used up), Muse Glimmer writes as well and the app says so: the product runs on the open-weight model
alone.

How a reading is taken from Muse Glimmer: when the service starts it tries several answer prefixes on a probe question
and keeps the one that puts the most probability on the option letters. A reading that still puts under half its
probability on the letters (the model wanting to reason first) is re-taken after the model writes its answer, at the
position of the letter it chose. Every reading reports its route and its letter mass; `meaningcheck smoke` prints both. Why statements and not word choice: in the earlier
pre-registered study, statement messages built from some models transferred to an unseen model (77.6% of readings
understood against 62.7% for messages written from one point of view), while word-level codes did not, and repeating a
message lowered understanding.

## Run it

```bash
pip install -e ".[dev]"
python -m pytest                       # all tests; simulated readers, no keys
meaningcheck --backend sim demo        # the built-in examples end to end
meaningcheck --backend sim serve       # web app on http://127.0.0.1:8000
```

Live readers and writer:

```bash
pip install modal && modal setup
modal deploy deploy/modal_glimmer.py   # prints the reader URL
export GLIMMER_URL=<that URL>
export MODEL_API_KEY=<your Meta Model API key>   # optional: without it Muse Glimmer also writes
meaningcheck smoke                     # checks readers, both writers, and lists your Meta models
meaningcheck serve                     # live web app
meaningcheck check "Please revert by Monday." --meaning "Please reply by Monday."
meaningcheck eval                      # held-out evaluation (see docs/PREREGISTRATION.md)
meaningcheck lint examples/lint.json   # exits 1 if any reader misreads: use in CI
# no need to stop anything: machines shut down after 5 idle minutes
```

### The Muse Spark second opinion

Set `MODEL_API_KEY` (your Meta Model API key) before `meaningcheck serve` or `modal deploy deploy/modal_web.py`.
Each check then also asks 8 Muse Spark readers (one answer in each of two option orders), cross-checks the best
wordings, and shows both panels. A public link is protected by a spending guard: at most `MC_SECOND_PER_HOUR` checks
an hour (default 40) use Muse Spark; later checks run on Muse Glimmer alone and say so. `MC_SECOND=off` disables it.
Note: from version 2.0 Muse Spark helps choose the fix, so it is no longer an independent reader for version 2.0's
fixes; the independent-model result below was measured on version 1.4's fixes.

### Staying within free credits

The readers need a GPU with 80 GB (Muse Glimmer's weights are about 58 GB). On Modal's free Starter plan ($30 of
compute a month) an A100 80 GB costs about $2.50 an hour, so the free credit covers about 12 GPU hours a month. To stay
inside it: set a $30 budget in the Modal dashboard (Settings, then Budgets); the service shuts down after 5 idle minutes;
every reading is cached, so repeating a check costs nothing. Do not stop the app between sessions: the public web app needs it, and an idle app costs nothing.
The writer (Muse Spark) uses the Meta Model API credit.

### No Python on your computer?

Open `colab/MeaningCheck_Colab.ipynb` in Google Colab (File, then Upload notebook) and run its cells in order. Colab is
only the control panel; the model runs on Modal. The notebook uploads the zip, runs the tests, connects Modal, downloads
the weights once on a CPU machine (`modal run deploy/modal_glimmer.py::download`), deploys the readers, runs the smoke
test, publishes the web app (`deploy/modal_web.py`, a public link on a CPU machine that sleeps when idle) and runs the
evaluation.

If Modal is unavailable, `--backend spark` samples Muse Spark several times per reading instead (slower, coarser).
Every reading is cached in `.mc_cache/`, so results can be reproduced without calling the models again.

## Limits

The panel is one model given different backgrounds, a proxy for people rather than people. The guarantee covers readers
like the panel's, not everyone. The simulated readers used by the tests and the offline demo illustrate the pipeline;
they are not measurements.

## Licence

MIT. Method and paper: Paul Agyemang, African Institute for Mathematical Sciences, Ghana.
