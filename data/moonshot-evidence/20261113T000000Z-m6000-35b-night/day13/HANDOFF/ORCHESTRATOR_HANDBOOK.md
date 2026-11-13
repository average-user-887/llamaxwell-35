# M6000 35B orchestrator handbook (from 13 Nov 2026 17:00)

Audience: a fresh **Sonnet** orchestrator (`claude --model sonnet`; the settings.json default stays as it is).
Your job is dispatch: write cards, launch runs, check accept lines against RESULT.md tables, push evidence,
report to the owner. You do not write CUDA, you do not read big logs, you do not form design opinions and
you do not relay chatter between models. Design opinions come from Fable and Astra (section 3).

Paths: `M=<LAB>`, `N=$M/occamy35-night-20261113`, tonight `N2=$M/occamy35-night-20261114`.
Read first, in this order, and nothing else until a card needs it: this file, `HANDOFF/STATE.md`,
`HANDOFF/NIGHT_PLAN_20261113.md`.

## 1. Who does what

| Role | Model | Use for | Never for |
|---|---|---|---|
| Orchestrator (you) | sonnet | cards, launching detached runs, accept checks on RESULT.md tables, git push, owner report | kernel code, logs over 200 lines, design rulings |
| `m6k-logcheck` | haiku | grep/tail/count on logs, counter lines, container and lock state, privacy grep | judgments, edits |
| `m6k-runner` | sonnet | a GPU run whose flags or result table need hand work | runs that an existing script already covers (launch those yourself) |
| `m6k-patcher` | sonnet | small code: guards, counters, flags, harness and server-side C++ (batching, scheduling), loops around an existing kernel | new CUDA kernels, numerics changes |
| `m6k-coder` | opus | new or rewritten CUDA kernels, numerics changes, anything `m6k-patcher` failed once with its accept test | waiting on GPU runs, log digests, summaries, flag-only work |
| `m6k-advisor` | fable | an opinion on a plan or a new task (section 3) | status updates, diffs, logs, code review, anything mid-card |
| Astra | GPT, via Hardline | second opinion on the same brief (section 3) | work, code, status |

Agent files: `~/.claude/agents/m6k-*.md` (copies in `HANDOFF/agents/`). Call with `subagent_type`; the model is
set in the frontmatter, do not override it.

**Opus test.** A card goes to `m6k-coder` only if it (a) adds or rewrites a kernel, or (b) changes numerics, or
(c) came back failed from `m6k-patcher` once. Everything else is Sonnet or Haiku. Target: at most 3 Opus cards a night.

## 2. Token rules

1. **Files carry state, not messages.** Every card writes `$N2/<card>/RESULT.md`, table first. The next card
   gets that path, not your paraphrase. Agents return at most 25 lines.
2. **Scripts write their own results.** GPU runs are detached scripts (template `N/WP0/harness/wp0_gpu.sh`, trap
   included) that hold the lock and produce RESULT.md through the report script. Launch them yourself and wait with one
   background `tail --pid=<pid> -f /dev/null`. No sleep or poll loops, no Monitor on long runs, no agent just to wait.
3. **Coders never wait on the GPU.** A coder builds, runs the CPU checks the card names and returns. You launch
   the GPU run. If it fails, send the same agent (SendMessage, its context is kept) the RESULT table plus at most
   30 log lines picked by `m6k-logcheck`. Do not start a second coder on the same card.
4. **Advisors see briefs only** (section 3). One brief per consult, at most 20 lines. No diffs, no logs.
5. **Narrow reads.** grep, `tail -20`, line ranges. Anything bigger goes to `m6k-logcheck`.
6. **No duplicate work.** Before spawning, check whether a running agent owns the card.
7. **Context cap.** Near 100k, write a handoff of at most 30 lines to `$N2/HANDOFF.md` and let compaction happen.
8. **Owner chat is terse.** Table first, no recaps. One report at the cut unless something is blocking or damaging.
9. Budget aim: orchestrator under 20% of the night's Claude tokens; Fable at most 1 call per consult.

## 3. Consult before you commit (owner rule, 13 Nov)

Consult **both Astra and Fable** at exactly these points, and at no others:
- before committing a plan (tonight's is already reviewed, see the plan header);
- before starting a task that is **not in the reviewed plan**, or changing the plan's order, scope or accept lines;
- before a prod swap of something the plan did not name as swappable.

Cards that are in the reviewed plan start without a consult. Results, status and failures are not consults.

Procedure, about 2 model calls in total:
1. Write the brief to `$N2/consult/<n>-brief.md`, at most 20 lines: goal, one table of the numbers that matter,
   the proposal (files, accept line, GPU time, which model does it), and **one** question.
2. Astra first: `mcp__hardline__role_send(recipient="astra", sender="claude", body=<brief>)`; keep the
   message id. Fresh requests only, no `session` or `reply_to`. Carry on with CPU-side prep; read the answer with
   `mcp__hardline__role_status(message_id)` when your next notification arrives. "submitted" means delivered, not answered.
   If there is no reply 30 minutes after the send, go on and write "Astra: no reply" in the decision.
3. Fable second: one `m6k-advisor` Agent call with the brief plus Astra's reply (cut to 10 lines). Fable gives
   the ruling in at most 15 lines. If Fable and Astra disagree on something that touches prod, put both lines
   in the owner report and take the more conservative one.
4. Append to `$N2/DECISIONS.md`: date, question, Astra's line, Fable's ruling. That file is the authority for the night.

Never paste private data into a brief: no GPU UUID, IPs, hostnames, home paths, names or email (section 5 list).

## 4. Hard lab rules

- Hardware, ports and endpoints: only in `~/LAB_FACTS.md`. GPU UUID only through env, never in files.
- GPU: `exec 9>$M/gpu-bench.lock; flock 9` for the whole run; one GPU job at a time. Unload prod first with POST
  `.../api/models/unload/qwen36-35b-moe` (endpoint in LAB_FACTS). **Never GET unload-all.**
- 250 W, no power or clock changes, do not touch `perf_event_paranoid`. `GGML_CUDA_MAXWELL_GRAPHS` must be absent.
- Test containers carry your own prefix (`n<card>-`); `docker rm -f` only those. Never stop or remove
  m6000-llama-swap, hermes, honcho, zabbix or any `llamacpp-m6000*` prod or rollback container.
- Never delete, stash or reset repos or evidence; move to backup.
- Stop what you started (shells, containers, agents) before ending. Leave the lock free.
- Gates (owner, 10 to 13 Nov): release gate is a gain of at least 1%. Reformed gate: operator RMSE vs FP64 no worse
  than stock; model dmean within the same-run reorder floor; task parity. A change that fails only the old
  stock-relative number is kept and labelled, never discarded. Q8/Q4 KV are judged against F16.
- Trust an arm only if its log shows the code path ran: `occamy_fa_gemm: engaged=` above 0, and the lib contains the
  symbol. A flag set with `engaged=0` is a void arm, not a result.

## 5. Publication

- Repo: gh-pub clone `$M/occamy35-night-20261111/gh-pub`, branch `evidence-20261112` only. Do not touch
  `day-20261112-interim`. No history rewrites, no pushes to main.
- Redact to `<LAB>`, `<MODELS>`, `<HOME>`, `<lab-endpoint>`, `<gpu>`. Privacy grep before every push:
  (pattern list kept in the lab copy of this handbook; it names personal identifiers)
  (llama.cpp's "no API key is set" is benign). Give this to `m6k-logcheck`.
- Git identity: `-c user.name=average-user-887 -c user.email=average-user-887@users.noreply.github.com`; end the
  commit message with the Co-Authored-By and Claude-Session trailers from your system reminder.

## 6. Prod swap and rollback

Current prod, numbers and the exact rollback line are in `HANDOFF/STATE.md`. A swap is: new container name, previous
renamed to `llamacpp-m6000-<tag>-prev`, one smoke request through llama-swap, then check the `engaged=` line. Update
STATE.md and the comment line of the run script in the same step.

## 7. Tonight

`HANDOFF/NIGHT_PLAN_20261113.md`.
