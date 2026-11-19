"""Isolate exact within-episode R2 prompt-cache reuse at the wire boundary."""
import json
from math import isfinite
from time import monotonic

import httpx

POLICY = {
    "first_request_cache_prompt": False,
    "later_requests_cache_prompt": True,
    "slot": 0,
    "cache_ram_mib": 0,
    "n_cache_reuse": 0,
    "thinking": "unchanged inherited template and reasoning-summary policy",
}


def require(goal, message):
    if not goal:
        raise RuntimeError(message)


def timing_counts(body, cache, previous):
    timing = body.get("timings")
    usage = body.get("usage")
    require(isinstance(timing, dict) and isinstance(usage, dict), "R2 timing and usage are required")
    names = ("cache_n", "prompt_n", "predicted_n")
    require(all(type(timing.get(name)) is int and timing[name] >= 0 for name in names),
            "R2 provided invalid token counts")
    cached, evaluated, generated = (timing[name] for name in names)
    require(evaluated >= 1 and cached + evaluated == usage.get("prompt_tokens")
            and generated == usage.get("completion_tokens")
            and cached == (usage.get("prompt_tokens_details") or {}).get("cached_tokens"),
            "R2 prompt/cache/generation counts do not reconcile")
    require(cache or cached == 0, "Cold episode entry reused prompt tokens")
    require(not cache or (previous is not None and cached <= previous),
            "Reported prefix exceeds immediately preceding slot contents")
    clocks = ("prompt_ms", "predicted_ms")
    require(all(type(timing.get(name)) in {int, float} and isfinite(timing[name])
                and timing[name] >= 0 for name in clocks), "R2 provided invalid timing")
    return {key: timing[key] for key in (*names, *clocks)}


class SampleCacheTransport(httpx.AsyncBaseTransport):
    def __init__(self, next_transport, request_guard, ids, receipt=None):
        self.next_transport = next_transport
        self.request_guard = request_guard
        self.ids = frozenset(ids)
        self.receipt = receipt
        self.current = None
        self.seen = set()
        self.accepted = []
        self.armed = False
        self.previous = None
        self.busy = False
        self.request_index = 0
        self.failed = set()
        self.identity = {}

    def open_episode(self, state):
        require(state is not None and state.epoch == 1 and state.sample_id in self.ids
                and isinstance(state.uuid, str) and bool(state.uuid), "No admitted Inspect sample identity")
        key = (state.sample_id, state.uuid)
        if self.current != key:
            require(state.sample_id not in self.seen, "A completed Inspect sample cannot re-enter the slot")
            self.current = key
            self.seen.add(state.sample_id)
            self.identity[state.sample_id] = state.uuid
            self.armed = False
            self.previous = None
            self.request_index = 0
        require(state.sample_id not in self.failed, "A failed episode cannot reuse the slot")
        return state.sample_id

    async def handle_async_request(self, request):
        from inspect_ai.solver._task_state import sample_state

        self.request_guard(request)
        require(not self.busy, "Only one owned BFCL wire request may be in flight")
        sample = self.open_episode(sample_state())
        self.busy = True
        try:
            cache = self.armed
            body = json.loads(request.content)
            require(not body.get("stream") and not any(key in body for key in ("id_slot", "n_cache_reuse")),
                    "Unexpected stream or injected slot/cache override")
            body.update(cache_prompt=cache, id_slot=POLICY["slot"], n_cache_reuse=POLICY["n_cache_reuse"])
            headers = [(k, v) for k, v in request.headers.multi_items() if k != "content-length"]
            wire = httpx.Request(request.method, request.url, headers=headers,
                                 content=json.dumps(body).encode(), extensions=request.extensions)
            self.request_index += 1
            started = monotonic()
            response = await self.next_transport.handle_async_request(wire)
            await response.aread()
            require(response.status_code == 200, f"R2 completion returned HTTP {response.status_code}")
            counts = timing_counts(response.json(), cache, self.previous)
            self.previous = counts["cache_n"] + counts["prompt_n"] + counts["predicted_n"]
            row = {"id": sample, "request": self.request_index, "cache_prompt": cache,
                   "epoch": 1, "model_wire_s": monotonic() - started, **counts}
            self.accepted.append(row)
            if self.receipt is not None:
                with self.receipt.open("a") as handle:
                    handle.write(json.dumps(row, allow_nan=False, sort_keys=True) + "\n")
            self.armed = True
            return response
        except BaseException:
            self.failed.add(sample)
            raise
        finally:
            self.busy = False

    async def aclose(self):
        await self.next_transport.aclose()

    def summary(self, samples):
        rows = {}
        for item in self.accepted:
            episode = rows.setdefault(item["id"], {"request_count": 0, "cache_n": 0, "prompt_n": 0, "predicted_n": 0,
                    "prompt_ms": 0.0, "predicted_ms": 0.0, "model_wire_s": 0.0})
            require(item["request"] == episode["request_count"] + 1, "Non-contiguous request witness")
            require(item["request"] != 1 or (not item["cache_prompt"] and item["cache_n"] == 0),
                    "Leaking first request witness")
            episode["request_count"] += 1
            for key in episode:
                if key != "request_count":
                    episode[key] += item[key]
        observed = {sample.id: sample for sample in samples}
        require(set(rows) <= observed.keys(), "Wire receipt includes a sample missing from Inspect")
        for sample_id, episode in rows.items():
            sample = observed[sample_id]
            require(sample.uuid == self.identity[sample_id], "Inspect sample UUID does not match the wire witness")
            if sample.turn_count is not None and sample.error is None and sample.limit is None:
                require(sample.turn_count == episode["request_count"], "Inspect turns disagree with wire count")
        require(all(sample.id in rows or sample.error is not None or sample.limit is not None for sample in samples),
                "Completed Inspect sample has no wire receipt")
        return {"policy": POLICY, "episodes": rows, "failed_wire_episodes": sorted(self.failed)}
