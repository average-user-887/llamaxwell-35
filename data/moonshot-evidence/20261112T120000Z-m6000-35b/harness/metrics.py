# Recomputable metrics from raw events (run dir events.json). Pure functions.
from evlib import pctl
def itl_ms(ts):  return [(b-a)/1e6 for a,b in zip(ts,ts[1:])]
def window_stats(streams):
    """streams: {name: ts_ns list}. common window = [max first token, min last token] (all streams actively decoding).
    aggregate = events with ts in (start,end] across streams / window; per-stream rates over same window. None if no overlap."""
    if len(streams)<2: return None
    s=max(v[0] for v in streams.values()); e=min(v[-1] for v in streams.values())
    if e<=s: return {'window_ns':0,'tokens':0,'agg_tps':None,'per_stream_tps':{k:None for k in streams}}
    per={k:sum(1 for t in v if s<t<=e) for k,v in streams.items()}
    d=(e-s)/1e9; return {'window_ns':e-s,'tokens':sum(per.values()),'agg_tps':sum(per.values())/d,'per_stream_tps':{k:per[k]/d for k in per}}
def tps_span(ts,a=None,b=None):
    v=[t for t in ts if (a is None or t>=a) and (b is None or t<=b)]
    return (len(v)-1)/((v[-1]-v[0])/1e9) if len(v)>1 else None
def deep_metrics(deep_ts,arr_ns=None,first16_ns=None):
    il=itl_ms(deep_ts); m={'n_tokens':len(deep_ts),'tps_own_span':tps_span(deep_ts),'itl_p50_ms':pctl(il,50),'itl_p95_ms':pctl(il,95),'itl_max_ms':max(il) if il else None}
    if arr_ns is not None and first16_ns is not None:
        a,b=arr_ns,first16_ns
        sub=[(t1-t0)/1e6 for t0,t1 in zip(deep_ts,deep_ts[1:]) if t1>=a and t0<=b]
        m.update(prefill_window_ms=(b-a)/1e6,deep_tokens_during_prefill=sum(1 for t in deep_ts if a<t<=b),
                 tps_during_prefill=sum(1 for t in deep_ts if a<t<=b)/((b-a)/1e9) if b>a else None,
                 itl_p50_during_prefill_ms=pctl(sub,50),itl_p95_during_prefill_ms=pctl(sub,95),itl_max_during_prefill_ms=max(sub) if sub else None,
                 itl_n_during_prefill=len(sub))
    return m
