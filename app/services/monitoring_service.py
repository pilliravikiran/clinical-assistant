"""
app/services/monitoring_service.py
==================================

Simple in-process metrics for monitoring.

Tracks counters and latency so a /metrics endpoint can expose them. In
production a tool like Prometheus scrapes /metrics and Grafana graphs it,
with alerts on latency spikes, error rate, or refusal rate.
"""

import threading

_lock = threading.Lock()
_m = {
    "requests": 0,
    "refusals": 0,
    "cache_hits": 0,
    "errors": 0,
    "latency_sum": 0.0,
    "latency_count": 0,
}


def reset():
    """Zero all metrics (used in tests)."""
    with _lock:
        for k in _m:
            _m[k] = 0 if k not in ("latency_sum",) else 0.0


def record_request(latency_seconds, refused=False, cache_hit=False):
    """Record one answered request (its latency + whether it refused/cache-hit)."""
    with _lock:
        _m["requests"] += 1
        _m["latency_sum"] += latency_seconds
        _m["latency_count"] += 1
        if refused:
            _m["refusals"] += 1
        if cache_hit:
            _m["cache_hits"] += 1


def record_error():
    """Record one failed request (an exception reached the endpoint)."""
    with _lock:
        _m["errors"] += 1


def get_metrics():
    """Return a snapshot of derived metrics (counts, rates, avg latency)."""
    with _lock:
        m = dict(_m)
    n = m["requests"]
    avg_ms = (m["latency_sum"] / m["latency_count"] * 1000) if m["latency_count"] else 0.0
    return {
        "requests_total": n,
        "refusals_total": m["refusals"],
        "cache_hits_total": m["cache_hits"],
        "errors_total": m["errors"],
        "avg_latency_ms": round(avg_ms, 1),
        "refusal_rate": round(m["refusals"] / n, 3) if n else 0.0,
        "cache_hit_rate": round(m["cache_hits"] / n, 3) if n else 0.0,
    }
