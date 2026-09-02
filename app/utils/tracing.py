"""
app/utils/tracing.py
====================

Observability with LangSmith (optional).

We expose a `traceable` decorator. If LangSmith is installed and enabled
(env vars below), decorated functions send a trace to LangSmith so you can see
every step, its inputs/outputs, and latency. If not, it's a harmless no-op.

Enable LangSmith by setting these environment variables (in .env):
    LANGCHAIN_TRACING_V2=true
    LANGCHAIN_API_KEY=your-langsmith-key
    LANGCHAIN_PROJECT=tenet-clinical-ai
"""

try:
    from langsmith import traceable as _traceable

    def traceable(*args, **kwargs):
        """Pass-through to LangSmith's traceable (traces only when enabled)."""
        return _traceable(*args, **kwargs)

except Exception:  # langsmith not installed -> no-op decorator
    def traceable(*args, **kwargs):
        # Support both @traceable and @traceable(name=...)
        if args and callable(args[0]) and not kwargs:
            return args[0]

        def decorator(func):
            return func

        return decorator
