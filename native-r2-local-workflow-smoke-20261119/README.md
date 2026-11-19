# Native R2 local workflow smoke

A paired pilot ran two public mock-tool workflows on a Quadro M6000 with the same R2 build, weights, effective chat template, and per-case limits. Both arms passed 1/2 for final state and 1/2 for the joint outcome. The native paired result for both metrics was zero wins, zero losses, and two ties.

| Public case | State / joint, serial | State / joint, native | Serial wall | Native wall |
| --- | ---: | ---: | ---: | ---: |
| `agentdojo-travel-u1` | 1 / 1 | 1 / 1 | 43.408 s | 42.891 s |
| `tau2-retail-23` | 0 / 0 | 0 / 0 | 103.295 s | 104.132 s |

Travel passed both its state and answer checks. Each arm generated one malformed tool argument that the mock adapter rejected and fed back to the model. Each also recorded one per-generation length finish, and the episode continued to a passing result. Retail received eight tool responses per arm with one output truncation each. On the ninth round, the rendered prompt reached 8,542 tokens in serial and 8,526 in native. Both exceeded the 8,192-token logical cap before generation, so both episodes received zero for state and joint outcome. Neither arm had infrastructure or transport errors. The sum of episode wall times was 146.703 s serial and 147.024 s native. Each evaluator took 148 s.

Both arms used all eight selected experts and offloaded 42/42 layers. Native used `draft-mtp` with one draft token and f16 draft K/V. Serial used `--spec-type none`. Native recorded 728 sampler-guard fallbacks, with 29,120 down4 launches and zero compact-only launches. These counts identify the physical path taken by the pilot. The runner, evaluator and cleanup exited zero in each arm. The owner-side audit verified both raw receipt seals and matched the runtime identity and budgets.

## Protocol and scoring

Each arm started a fresh model server. The worker used seed 7, temperature 0, thinking disabled, and uncached, non-streaming chat. It rendered and tokenized the request before generation, reserving up to 256 output tokens per round within the logical 8,192-token envelope. It checked the reported prompt-token count against the tokenizer result. The shared episode limits were 20 rounds, 32 tool attempts, 2,048 total output tokens, 120 seconds, and 2,048 bytes per tool response. All tools ran against fresh mock fixtures.

The AgentDojo adapter checked the expected state changes and used the upstream task utility to grade the final answer for the joint metric. In Retail the joint result equaled the state result. The adapter used a deterministic confirmation turn and compared the resulting full Retail database to the frozen golden state. The preflight verified the golden result against the upstream database scorer. These scores describe the scripted local adaptation. [AgentDojo source](https://github.com/UKGovernmentBEIS/inspect_evals/tree/32b79a244f17ecf54fa1186d30f33dc10143c10f/src/inspect_evals/agentdojo) · [Tau2 source](https://github.com/UKGovernmentBEIS/inspect_evals/tree/32b79a244f17ecf54fa1186d30f33dc10143c10f/src/inspect_evals/tau2)

The frozen local cohort has 80 unique cases, split evenly between AgentDojo and Tau2 Retail. These two cases are members of that cohort and each counts once in a full evaluation. The pilot leaves the remaining 78 cases and cohort-level quality unevaluated. BFCL and IFEval use separate protocols and scores.

[Paired numeric evidence](evidence/agentic-local-r2-smoke-pair.json) includes per-case timing, token usage, tool statistics and outcomes. The [wire and tokenizer](source/workflow-wire.txt), [mock loop](source/mock-loop.txt), and [scoring](source/scoring.txt) excerpts preserve the observed logic. The excerpts require the surrounding runner and pinned public dependencies for execution. The public packet uses numeric observations, public smoke IDs and boolean provenance checks. Raw transcripts, actual state digests and machine-specific admission receipts stay in the private archive.

Run `sha256sum -c SHA256SUMS` to verify the package and `python3 -B verify-evidence.py` to recompute the public comparisons without a model service.
