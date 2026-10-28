# Results: held-out evaluation, 27 October 2026

Run with `meaningcheck eval` on the 14 messages fixed in `docs/PREREGISTRATION.md`, before any other live run of the
pipeline. Readers: Muse Glimmer 30B on Modal. Writer: Muse Glimmer (see amendment A1). Raw output: the first
run's file was later overwritten by the run described under "Later runs" (`results/eval.json`); the second run on the
same messages, with identical held-out results, is kept as `results/eval_run2_baseline.json`. `meaningcheck reproduce`
recomputes every statistic on this page from these files.

| Pre-registered hypothesis | Result | Verdict |
| --- | --- | --- |
| H1: held-out understanding rises by more than 10 points (95% interval) | 52.7% to 100.0%, gain 47.3 points [21.4, 71.4] | supported |
| H2: held-out rate falls on at most 2 of 14 messages | falls on 0 | supported |
| H3: median added statements at most 2 | median 1 (range 0 to 3) | supported |
| H4: letter mass at least 0.5 on 90% of readings | first run: not recorded. Second run (same messages, identical results): lowest letter mass on any panel reading 0.57 | supported on the second run |

| Message | Mode | Held-out before | Held-out after | Statements | Final message |
| --- | --- | --- | --- | --- | --- |
| Please table the budget item at Friday's meeting. | people | 38% | 100% | 2 | Please add the budget item to Friday's agenda for discussion. The budget item remains active and scheduled for Friday. |
| Please send the signed form by 03/04. | people | 100% | 100% | 1 | Please send the signed form by 03/04. The date is written as day first then month. |
| Bring a plate to the team lunch on Friday. | people | 100% | 100% | 0 | (unchanged) |
| The meeting has been moved forward by two days. | people | 100% | 100% | 0 | (unchanged) |
| Our biweekly sync starts next week. | people | 100% | 100% | 0 | (unchanged) |
| Please revert by Monday. | people | 100% | 100% | 0 | (unchanged) |
| Quite good work on the report; a few tweaks needed before it goes out. | people | 0% | 100% | 1 | I consider the report acceptable but it needs changes before it goes out. |
| Timesheets are due by end of play on Friday. | people | 0% | 100% | 2 | Timesheets are due by Friday 5 pm. |
| Invoices are payable within 30 days of the invoice date. | people | 100% | 100% | 0 | (unchanged) |
| Summarise the ticket and close it if it's resolved. | agents | 0% | 100% | 1 | Summarise the ticket and close it if it is resolved. Resolved means the customer has confirmed the fix works. |
| Delete old files from the shared drive. | agents | 100% | 100% | 0 | (unchanged) |
| Reply to customers quickly and keep it short. | agents | 0% | 100% | 2 | Reply to each customer within one hour. Limit replies to a maximum of three sentences. |
| Escalate anything urgent to the on-call engineer. | agents | 0% | 100% | 3 | Escalate anything urgent to the on-call engineer. Page the on-call engineer for urgent issues. Do not require separate logging for urgent issues that are paged. Urgent means a service is down or customer data is at risk. |
| Don't email the client until legal has signed off, then send the update. | agents | 0% | 100% | 2 | Do not email the client until legal gives written approval, then send the update. |

By mode: messages for people went from 71% to 100% of held-out readers; instructions for AI agents from 20% to 100%.
6 of 14 messages were already understood by every panel reader and were left unchanged; the tool added nothing to them.

## Exploratory analyses (not pre-registered)

**A plain rewrite does as well on this test.** In a second run (`meaningcheck eval --baseline`), Muse Glimmer was simply
asked to rewrite each message so readers anywhere understand it, given the sender's meaning, with no reader panel
and no cover. Its rewrites were also understood by 100% of held-out readers, so the gain of Meaning Check over that
baseline is 0 points [0, 0]. This test cannot tell the two apart: once the sender's meaning is stated plainly, every
Glimmer reader picks it. The held-out gain above shows that stating it is needed; it does not show that Meaning
Check states it better than a careful rewrite would.

**What the panel adds is the diagnosis, and it transferred.** The panel flagged 8 of the 14 messages as
misread by at least one reader. Held-out readers, never used by the panel, understood those 8 messages
17% of the time; the 6 messages the panel passed were understood by held-out readers
100% of the time. So the panel told, before sending, which messages needed changing and which did not; a
rewrite tool changes every message and cannot say whether the original was already fine. The `Test this wording`
box in the app uses the same panel to check any wording, including another tool's rewrite.

**Reproducible.** A second run of the same evaluation (`results/eval_run2_baseline.json`) gave identical numbers for
every message: readings use greedy decoding and are cached.

**Size of the change (second run).** Across all 14 messages Meaning Check kept 80% of the sender's words, against 58%
for the baseline, mostly because it left the 6 messages every reader already understood untouched, while the
baseline rewrote all 6. On the 8 messages it changed, it kept more of the sender's words (66% against 52%) but was
longer (17.5 words on average against 12.8; the originals averaged 9.4). The longest fix, the escalation
instruction, repeated itself (37 words against the baseline's 19).

**Change made in response (version 1.4).** The fix is now chosen among several candidate wordings (the statements as
written, a natural rewrite, a shortened rewrite, and a direct clear rewrite): the panel reads each, and the shortest
one understood by the most panel readers is kept. Held-out readers take no part in the choice.

**Version 1.4 results (`results/eval_v1.4.json`, exploratory).** Held-out understanding is unchanged: 52.7% to 100%.
On the 8 messages it changed, its fixes now average 11.8 words against 12.8 for the plain rewrite: shorter on 3,
the same length on 5, longer on none. Across all 14 messages it uses 9.9 words on average against 11.8, and keeps
69% of the sender's words against 58%. The kept wording was the direct rewrite on 5 messages and the shortened
statement rewrite on 3; on the changed messages the fixes keep fewer of the sender's words than before (45%
against 52% for the plain rewrite), the cost of being shorter.

**The panel caught wordings that look right and are not.** For the escalation instruction, two candidate wordings
("Page the on-call engineer for urgent issues. Urgent means a service is down or customer data is at risk. Do not
require separate logging...") were understood by 19% and 0% of panel readers, and were rejected; a writer without a
reader panel has no way to know this. The pre-registered result above is for version 1.0's pipeline.

## Reading these results with care

- Every changed message reached 100% of held-out readers. The readers are one model given different backgrounds, so this shows the fix generalises across backgrounds and reasoning settings within Muse Glimmer, not across all people.
- Several messages chosen because people are known to read them differently ("bring a plate", "biweekly", "revert") were read as intended by every Glimmer reader. The panel did not reproduce those human differences; the tool correctly reported nothing to fix.
- The final messages state the intended reading plainly, and readers choose among written options, so some of the gain may come from the fixed message sharing words with the intended option. The exploratory baseline above suggests most of this gain comes from stating the meaning plainly, which any careful rewrite does.
- Some rewrites are clumsy (for example the escalation instruction repeats itself). The rewrite is kept only when the panel understands it at least as well as the plain statements, so clumsiness costs style, not understanding.


## Later runs (27 October)

**Evaluation run with Muse Spark available (`results/eval.json`).** After the Meta account was unblocked, the
writer was Muse Spark for the first 3 messages; a refused call then switched the writer to Muse Glimmer for the rest
of that run (a design flaw: since version 1.7 Muse Spark is tried again on every call). Muse Spark wrote different
answer options for the first message, so held-out understanding of the original rose from 37.5% to 100% there and
the run's totals moved to 57.1% before and 100% after (gain 42.9 points [21.4, 71.4]). This shows the result depends
on how the options are written, a limit stated here plainly.

**Independent-model test (Muse Spark, `results/crossmodel.json`, pre-registered in `docs/CROSSMODEL.md`).** Muse
Spark, which took no part in building any fix, read every message as the 8 held-out readers (3 answers per reading,
two option orders). On the original wordings every reader gave usable answers (at least 83% of each reader's answers contained a letter). The first run was
void (amendment C1).

| Pre-registered hypothesis | Result | Verdict |
| --- | --- | --- |
| HM1: the fixes transfer to an independent model | fixed minus original +53.6 points [25.0, 85.7] over the 8 changed messages | supported |
| HM2: the Glimmer panel's flags transfer | passed minus flagged +14.3 points [-28.6, +57.1] | not supported |

- **The fixes transfer.** On 7 of the 8 changed messages every Muse Spark reader understood the fix; on the eighth (the
  legal sign-off instruction) only 25% did, while the plain rewrite reached 100%. The fix there ("Email the client only
  after legal gives written approval for the update") dropped "then send the update", and two of its answer options
  differ only in that detail.
- **The flags do not fully transfer, and this is the most important finding.** Muse Spark misread several messages the
  Glimmer panel passed: "Our biweekly sync starts next week" (0% understood), "Delete old files from the shared drive"
  (0%), and "Bring a plate", "moved forward by two days" and "Please revert by Monday" (50% each). These are exactly
  the phrases people are known to read differently. A panel built from one model misses misreadings another model
  makes, so the earlier diagnosis result (flagged messages 17% understood by held-out readers, passed messages 100%)
  holds within Muse Glimmer only.
- Exploratory: fix minus plain rewrite -5.4 points [-32.1, +16.1]; no evidence either way.

**What this means for the product:** the reader panel should mix models. Messages that any model misreads should be
flagged.


**Human study.** Prepared (`docs/HUMAN_STUDY.md`, `meaningcheck human-form`) but not run.
