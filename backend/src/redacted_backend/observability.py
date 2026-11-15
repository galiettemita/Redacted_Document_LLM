from __future__ import annotations

import logging
import re
import threading
import traceback
from collections.abc import Mapping
from types import MethodType
from typing import Any

_REDACTED = "[REDACTED]"
_SENSITIVE_KEY_FRAGMENTS = (
    "api_key",
    "authorization",
    "credential",
    "document_body",
    "document_text",
    "evidence",
    "excerpt",
    "pdf_text",
    "prompt",
    "retrieved",
    "secret",
)
_BEARER_PATTERN = re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~+/=-]+")
_OPENAI_KEY_PATTERN = re.compile(r"\bsk-[A-Za-z0-9_-]{8,}\b")
_INSTALL_LOCK = threading.Lock()
_INSTALLED = False


def install_log_redaction() -> None:
    """Redact sensitive structured values before any logging handler sees them."""

    global _INSTALLED
    if _INSTALLED:
        return
    with _INSTALL_LOCK:
        if _INSTALLED:
            return
        base_factory = logging.getLogRecordFactory()

        def redacting_factory(*args: Any, **kwargs: Any) -> logging.LogRecord:
            record = base_factory(*args, **kwargs)
            original_get_message = record.getMessage

            def redacting_get_message(self: logging.LogRecord) -> str:
                _redact_record(self)
                return original_get_message()

            record.getMessage = MethodType(redacting_get_message, record)
            return record

        logging.setLogRecordFactory(redacting_factory)
        _INSTALLED = True


def _redact_record(record: logging.LogRecord) -> None:
    for key, item in tuple(record.__dict__.items()):
        if key == "getMessage":
            continue
        normalized = key.casefold().replace("-", "_")
        if any(fragment in normalized for fragment in _SENSITIVE_KEY_FRAGMENTS):
            record.__dict__[key] = _REDACTED
        else:
            record.__dict__[key] = _redact_value(item)
    record.msg = _redact_value(record.msg)
    record.args = _redact_log_args(record.args)
    if record.exc_info is not None:
        exception_type, _exception, exception_traceback = record.exc_info
        rendered = "Traceback (most recent call last):\n"
        rendered += "".join(traceback.format_tb(exception_traceback))
        rendered += f"{exception_type.__name__}: {_REDACTED}\n"
        record.exc_text = str(_redact_value(rendered))
        record.exc_info = None


def _redact_log_args(args: object) -> object:
    if isinstance(args, Mapping):
        return _redact_mapping(args)
    if isinstance(args, tuple):
        return tuple(_redact_value(item) for item in args)
    return _redact_value(args)


def _redact_value(value: object) -> object:
    if isinstance(value, Mapping):
        return _redact_mapping(value)
    if isinstance(value, list):
        return [_redact_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_redact_value(item) for item in value)
    if isinstance(value, str):
        return _OPENAI_KEY_PATTERN.sub(
            _REDACTED,
            _BEARER_PATTERN.sub(_REDACTED, value),
        )
    return value


def _redact_mapping(value: Mapping[object, object]) -> dict[object, object]:
    redacted: dict[object, object] = {}
    for key, item in value.items():
        normalized = str(key).casefold().replace("-", "_")
        if any(fragment in normalized for fragment in _SENSITIVE_KEY_FRAGMENTS):
            redacted[key] = _REDACTED
        else:
            redacted[key] = _redact_value(item)
    return redacted
