# Native R2 paired benchmark smoke results

Two small uncached pilots compared native R2 with serial inference on the same build, model weights and effective chat template within each pilot. Native used `draft-mtp` with one draft token and f16 draft K/V. Serial used `--spec-type none`. Fresh server processes ran each arm.

| Pilot | Native | Serial | Native wins / losses / ties |
| --- | ---: | ---: | ---: |
| BFCL multi-turn cases | 2/4 | 3/4 | 0 / 1 / 3 |
| IFEval prompts, strict and loose each | 4/6 | 5/6 | 0 / 1 / 5 |
| IFEval instructions, strict and loose each | 10/12 | 11/12 | 0 / 1 / 11 |

## BFCL uncached four-case smoke

The four public IDs come from the original disjoint smoke cohort. Both Inspect workers completed the four cases without an infrastructure error or cap. Their 25 native and 27 serial tool responses were fed back to the model. Neither the Inspect tool messages nor the tool execution events carried a typed error. The official grader recorded a response mismatch on `multi_turn_base_135` in native only and a state mismatch on `multi_turn_miss_func_24` in both arms. Each of the other two cases passed in both arms. [Score and counter evidence](evidence/bfcl-uncached-score.json)

Two earlier serial admission attempts refused valid Docker CDI metadata and cleaned up before inference. The first verifier expected a legacy device listing that Docker left empty after accepting the exact CDI selector. The next verified the owned character device but rejected Docker’s canonical typed CDI request for that selector. Revision 2 accepted a single typed CDI request for the admitted selector and matched the owned device. [Launch outcomes](evidence/bfcl-launch-outcomes.json)

The original native wrapper exited 1 after the evaluator exited 0. Its inherited counter parser accepted 29 of 37 guard lines and rejected the eight timestamp-only lines. The corrected observer matched all 37 guard and 37 readback lines using the sealed logs, and the original wrapper status remains visible. Both arm cleanups exited 0. [Counter rows](evidence/bfcl-counter-rows.json) · [Observed syntax](workers/compact_witness.py)

The native sampler guard sent 3,039 proposals through fallback and the down4 counter advanced by 121,560 launches. The compact requested, launched and compact-only deltas were all zero. Serial also recorded zero compact launches. These counters describe the observed path through the build.

The sum of four episode wall times was 455.364 s native and 500.981 s serial. The evaluator elapsed times were 461 s and 506 s. Native made 37 model turns and serial made 40, with different token totals. These wall times describe the completed episodes. The timing file gives separate server prompt-evaluation and decode totals. [Timing](evidence/bfcl-uncached-timing.json)

## IFEval independent six-prompt smoke

The IFEval worker evaluated six public prompts with 12 instructions, using the pinned Google IFEval dataset, Inspect Evals mapper and upstream-derived IFEval scoring implementation. Strict and loose scores matched within each arm. The forbidden-words constraint accounted for the one native-only failure relative to serial. Both arms failed one sentence-count constraint. Every planned prompt was scored in both arms and both wrappers and cleanups exited 0. [Paired scores](evidence/ifeval-smoke-score.json) · [Launch outcomes](evidence/ifeval-launch-outcomes.json) · [Exact benchmark and evaluator pins](evidence/protocol-pins.json)

Native recorded 901 sampler-guard fallback proposals and 36,040 down4 launches. The compact requested, launched and compact-only deltas were all zero. The evaluator elapsed times were 23 s native and 25 s serial. [Timing](evidence/ifeval-smoke-timing.json)

## Scope and verification

The predeclared quality campaign uses a five-percentage-point non-inferiority margin with 160 held-out BFCL pairs and an independent 80-workflow cohort of 40 AgentDojo and 40 Tau2 Retail cases. The BFCL four-case pilot is disjoint from its held-out cohort. The IFEval pilot is a subset of its independent 541-prompt, 834-instruction suite, which would require fresh requests. These smoke results leave general equivalence and the larger quality gate unevaluated.

The evidence files keep public case IDs, outcomes, typed failure classes and numeric observations. The run identities and raw logs stay in the sealed owner archive. The counter-row export replaces the numeric timestamps with a constant while preserving the line forms and counters. Same-build checks report the verified equality of private runtime identity fields as booleans.

From this directory, `sha256sum -c SHA256SUMS` checks the public files and `python3 -B verify-evidence.py` recomputes the paired outcomes and audits the published BFCL counter rows without model requests or external dependencies.

The worker sources retain the original scoring and admission logic. The IFEval worker and compact marker parser are archived byte for byte. The BFCL worker changes its first line to a portable interpreter directive. An owner-side byte reconstruction replaces that directive with the shebang from the sealed source and verifies against the local freeze. An end-to-end benchmark run uses the pinned dependencies, upstream data, regenerated local manifests and an admitted local server.
