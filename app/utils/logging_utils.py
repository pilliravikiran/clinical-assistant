"""
app/utils/logging_utils.py
==========================

One place to set up logging for the whole app.

Why logging: in production you can't watch print() statements. Logs record what
the app did (events, timing, counts, errors) so you can debug and audit later.

Healthcare rule: NEVER log raw patient data (PHI). We log events and numbers,
and when we must include user text we redact it first (guardrails_service).
"""

import logging


def get_logger(name):
    """
    Return a configured logger.

    Input:  name -> usually the module name, e.g. "rag_service"
    Output: a logging.Logger you can call .info(), .warning(), .error() on.

    We configure it only once per name (avoids duplicate log lines).
    """
    logger = logging.getLogger(name)

    # Only add a handler the first time - otherwise logs print multiple times.
    if not logger.handlers:
        handler = logging.StreamHandler()   # print logs to the console
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)       # show INFO and above

    return logger
