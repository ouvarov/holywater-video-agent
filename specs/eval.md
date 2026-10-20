# eval (not a stage)

**Purpose.** Part 3 of the note: see whether the pipeline gets better or worse between versions, with numbers.

**Command.** `uv run eval [--profile dev|premium] [--ideas evals/ideas.yaml] [--out evals/results/<ts>.csv]` and `uv run eval --compare A.csv B.csv`.

**Idea set.** `evals/ideas.yaml`: 4 ideas — Ukrainian line in quotes; English line; no line (LLM writes it); a long line near the duration limit.

**Output CSV, one row per idea:** run_id, profile, models+versions, strategy, final status, failed stage, attempts per stage, final_check raw values, judge scores, cost_usd, seconds, `provider_failure_rate` (see below), plus empty columns for human scores (stress, emotion, overall).

**Compare.** Pass rate of hard gates, provider failure rate, mean cost, mean time, per-check deltas, judge-score deltas. A new version is accepted if gate pass rate is not lower and cost is not more than +25 %.

**Lip-sync columns (v2, see `sync.md`; replace v1's `sync_lag_s`, `sync_peak_r`).** `syncnet_confidence`, `syncnet_offset_frames`, `sync_verdict`, `sync_method` (`global` | `piecewise`), `sync_conf_global`, `sync_conf_piecewise`, `sync_shifts` (the piecewise per-phrase `shift_s` joined with `;`, e.g. `-0.083;0.042;0`), placed right after the `fc_*` columns in that order: shot_1's recorded `sync` (any status — a failed reject still records it), empty when absent or null (a v1 record fills only `sync_verdict`). `--compare` prints `mean syncnet_confidence: A -> B (+d)` when both sides have values (an old CSV without the column → the line is omitted, no error); not part of the accept rule.
- Acceptance: a dev run whose shot_1 record has confidence 8.63, offset −6, method `piecewise`, phrase shifts −0.083/0.042/0 → those columns with those values; a run without a record → empty cells; `--compare` of two CSVs with `syncnet_confidence` prints the line, with an old CSV lacking it does not.

## Outcome of a run

Each row has one outcome, read from the failed stage's recorded error:
- **gate pass** — status `done`.
- **gate fail** — the run ended on a `GateFailed`, or on an `Unexpected …` crash (a bug in our code must never make a version look better).
- **provider failure** — the run ended on a `ProviderError` (moderation, timeout, quota, retries used up), or on shot_2's moderation `GateFailed` (its message says "(moderation)"; the stage wraps the provider rejection to add the one-shot fallback note).

CSV column `provider_failure_rate`: `1` for a provider failure, `0` otherwise, so its mean over rows is the rate (works in Excel as is). It is placed right after `error`.

**Gate pass rate** = gate pass / (gate pass + gate fail). Provider failures are left out of both the numerator and the denominator. With no run that reached a verdict the rate is `n/a`.
**Provider failure rate** = provider failures / all rows. Printed in `--compare`, not part of the accept rule.
Mean cost and mean time are over all rows, provider failures included (the money was spent).

## Writing the CSV

- The header is written and flushed before the first idea runs; each idea's row is appended and flushed as soon as that idea finishes. A crash or Ctrl-C later keeps every row already written.
- Ctrl-C: the eval stops, prints one line to stderr (`interrupted: <n> rows in <path>`), exit code 130. The idea that was running gets no row (its run dir and manifest stay for `--resume`).
- Any other exception propagates as before, but the finished rows are already in the file.
- Usage errors (bad profile, bad ideas file) still create no CSV.
- A cell that starts with `=`, `+`, `-`, `@`, a tab or a CR gets a leading `'` (no formula injection when opened in Excel; a line or an error text is user/provider input). A plain number (`-23.5`, LUFS) is not a formula and is written as is, so `--compare` and Excel still read it as a number. The `attempts_<stage>` columns follow `pipeline.STAGES`.

## Ideas file errors

All are usage errors: exit 2, exactly one line on stderr starting with `error:` and naming the file, no traceback, no CSV, no run dir:
- missing / unreadable file, not UTF-8, YAML syntax error, top level not a mapping with an `ideas:` list;
- an invalid entry (unknown kind / key, empty `idea`, duplicate id);
- `line: ""` (or whitespace only) — a config error, never read as "no line". To let the LLM write the line, leave `line` out.

## Compare details

- Rows are matched by `idea_id`. If A and B cover different ideas, one `warning:` line on stdout names the ideas only in A and only in B, and every number (rates, cost, time, checks, judge) is computed over the common ideas only. No common ideas, or a duplicate `idea_id` within one file → usage error (exit 2, one line on stderr).
- Printed lines: `gate pass rate: A -> B (+d)` (2 decimals; `n/a` for a side with no verdict, then no delta), `provider failure rate: A -> B (+d)`, `mean cost_usd`, `mean seconds`, `fc_<check>`, and for each judge criterion and defects with values on both sides `mean judge_<name>: A -> B (+d)` (2 decimals).
- Accept rule (unchanged): gate pass rate B ≥ A and mean cost B ≤ 1.25 × A. If B's gate pass rate is `n/a` → REJECT ("no run reached a gate verdict"); if only A's is `n/a`, the gate part passes.

**Acceptance criteria.** On the `dev` profile: runs all ideas, writes the CSV with the columns above; `--compare` prints deltas; one-idea dev eval is fast enough to be a regression test in `uv run check`. Plus every rule in the sections above (incremental CSV and Ctrl-C, outcome classification and both rates, ideas-file errors incl. `line: ""`, compare on common ideas, judge-score deltas).
