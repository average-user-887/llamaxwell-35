# M6000 35B orchestrator handbook (from 13 Nov 2026 17:00)

Audience: a fresh **Sonnet** orchestrator. You plan, launch, verify and report. You do not write CUDA,
you do not read big logs, and you do not relay messages between other models.

## 1. Roles and model picker

| Role | Model | Use for | Never for |
|---|---|---|---|
| Orchestrator (you) | sonnet, effort medium | card specs, launching runs, gate checks on RESULT.md tables, git push, owner chat | kernel code, reading >200-line logs |
| `m6k-coder` agent | opus | CUDA/C++ patches, builds, harness code changes, numerics fixes | waiting on GPU runs, log digests |
| `m6k-runner` agent | sonnet | one GPU run under the lock from an existing script, writes RESULT.md | code changes beyond flags/env |
| `m6k-logcheck` agent | haiku | grep/tail a log, count lines, check a counter, privacy grep | judgment calls |
| Fable advisor | fable (peer session "llamaxwell-35b optimisation advisor", or one `model: "fable"` Agent call if the peer is gone) | architecture/design choices, gate rulings, plan review at milestones | status updates, relaying, code review of every diff |

Agent files: `~/.claude/agents/m6k-{coder,runner,logcheck}.md` (copies in `HANDOFF/agents/`). Call them with
`subagent_type: "m6k-coder"` etc.; their model is set in the frontmatter, do not override it.

Start the session: `claude --model sonnet` (settings.json default stays `opus`; do not change it).

## 2. Token rules

1. **Opus writes, Opus reads its own code.** Give the coder a self-contained card (goal, base commit, files,
   accept test, output paths). It returns ≤25 lines: commit SHAs, build sha256, the test it ran, open risks.
   Do not ask Opus to summarise for Fable or for the owner. You do that from its RESULT.md.
2. **One card per agent, and you pass state on through files.** Every card writes `$N2/<card>/RESULT.md` (a table first). The
   next card reads that file, not your paraphrase of it.
3. **Fable is consulted, not cc'd.** At most ~1 message per milestone: a ≤20-line table plus one question
   ("adopt X?", "which of A/B next?"). Its answer is a decision; record it in `$N2/DECISIONS.md`.
   Fable never spawns subagents. Do not send Fable diffs or logs.
4. **Background and notify.** GPU runs are detached scripts with a trap. Wait via a background
   `tail --pid=<pid> -f /dev/null`. No sleep/poll loops, no Monitor on long runs.
5. **Narrow reads.** `grep`, `tail -20`, line ranges. Anything bigger goes to `m6k-logcheck`.
6. **Context cap.** At ~100k, write a ≤30-line handoff to `$N2/HANDOFF.md` and let compaction happen.
7. **No duplicate work.** Before you spawn, check whether a running agent owns the card; continue it with SendMessage.
8. Budget hint: Opus for code (2–4 cards/night), Sonnet runners for GPU, Haiku for checks. The orchestrator
   should spend <20% of night tokens.

## 3. Hard lab rules (verbatim, non-negotiable)

- Hardware/ports/endpoints: only in `~/LAB_FACTS.md`. GPU UUID only via env, never in files.
- GPU: `exec 9>$M/gpu-bench.lock; flock 9` for the whole run; one GPU job at a time. Before use POST
  `.../api/models/unload/qwen36-35b-moe` (see LAB_FACTS for the endpoint). **Never GET unload-all.**
- 250 W, no power/clock changes, do not touch `perf_event_paranoid`. `GGML_CUDA_MAXWELL_GRAPHS` must be ABSENT.
- Test containers: own prefix (`n<card>-`); `docker rm -f` only your own. Never stop m6000-llama-swap,
  hermes, honcho, zabbix or `llamacpp-m6000*` prod/rollback containers. Never remove containers.
- Never delete/stash/reset repos or evidence; move to backup.
- Stop what you started (shells, monitors, containers, agents) before ending.
- Gates (owner, 10–13 Nov): release gate ≥1% gain; reformed gate = operator RMSE vs FP64 ≤ stock,
  model dmean within same-run reorder floor, task parity; stock-relative-only fails are kept and labelled,
  never discarded. Q8/Q4 KV judged against F16. Every gain counts.

## 4. Publication

- Repo: gh-pub clone `$M/occamy35-night-20261111/gh-pub`, branch `evidence-20261112` only. Do not touch
  `day-20261112-interim`. No history rewrites, no main pushes.
- Redact: `<LAB>`, `<MODELS>`, `<HOME>`, `<lab-endpoint>`, `<gpu>`. Privacy grep before every push:
  (pattern list kept in the lab copy of this handbook; it names personal identifiers)
  (llama.cpp's "no API key is set" is benign).
- Git identity: `-c user.name=average-user-887 -c user.email=average-user-887@users.noreply.github.com`; end the
  commit message with the Co-Authored-By / Claude-Session trailers from your system reminder.

## 5. Prod swap / rollback

See `HANDOFF/STATE.md` for the current prod container and the exact rollback commands.

## 6. Night plan

`HANDOFF/NIGHT_PLAN_20261113.md`.
