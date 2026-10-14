# Controlled faults: format drift and infra change

**4 of 4 correct.** Production path: per-call signals, live Page-Hinkley during the run (format), retrospective PELT → permutation → BH, classification rules / incidents / mapper. Correct = a confirmed change with the expected label near the fault and nothing earlier; on a control, no label at all.

- **Format** (live gpt-oss-120b on Groq): during the fault phase half the format requests were sent without the "Return ONLY a JSON ..." instruction. Real replies, same schema check.
- **Infra** (replay): real gpt-oss-120b latencies from driftguard.db, shuffled; after call 600, 40% of calls get +3 s injected on our side.

| part   | stream   | expected     |   calls |   bad_rate_before |   bad_rate_after | verdict   | labels                                                                                                        |
|:-------|:---------|:-------------|--------:|------------------:|-----------------:|:----------|:--------------------------------------------------------------------------------------------------------------|
| format | fault    | format_drift |     240 |         0         |         0.375    | correct   | format_drift (pelt) at call 120; format_drift (sequential) at call 160; infra_change (sequential) at call 183 |
| format | control  | no label     |     240 |       nan         |       nan        | correct   | no label                                                                                                      |
| infra  | fault    | infra_change |    1200 |         0.0716667 |         0.463333 | correct   | infra_change (pelt) at call 600; infra_change (sequential) at call 626                                        |
| infra  | control  | no label     |    1200 |       nan         |       nan        | correct   | no label                                                                                                      |

## Side effect seen in the format fault

The format fault stream also got an **infra_change** early warning (live detector, call 183; not
confirmed by PELT). It is real, not a false alarm: the plain-English requests produced much longer
answers (median 937 characters and 338 tokens, against 74 characters and 100 tokens for JSON), so
those calls were slower (median 1,125 ms against 688 ms; p95 2,172 ms against 1,237 ms). The
classification layer grouped both into one incident -- "2 signals moved together within 4 minutes:
format drift, infra change. First: format drift. Evidence: corroborated", urgency **act now** -- and
suggested ruling out infrastructure first. Here the order of events (format first, latency after)
points the other way: the slowdown is a consequence of the longer answers.

## Run notes

The first live format run sent a bookkeeping field to the provider with the faulted requests; Groq
rejected those 45 calls as invalid, so they surfaced as call failures instead of format failures.
The field was removed and the fault stream re-run; the control stream from the first run was
unaffected and is the one reported.
