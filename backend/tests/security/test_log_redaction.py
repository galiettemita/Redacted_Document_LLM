import logging

from redacted_backend.config import Settings
from redacted_backend.main import create_app


def test_normal_logs_redact_document_prompt_evidence_and_credentials(caplog) -> None:
    create_app(Settings(_env_file=None))
    logger = logging.getLogger("redacted_backend.security_test")

    with caplog.at_level(logging.INFO):
        logger.info(
            "analysis payload=%s",
            {
                "pdf_text": "PRIVATE PDF CONTENT",
                "prompt": "PRIVATE PROMPT CONTENT",
                "retrieved_excerpts": ["PRIVATE EVIDENCE CONTENT"],
                "authorization": "Bearer private-token-value",
                "event": "analysis_failed",
            },
        )

    assert "analysis_failed" in caplog.text
    assert "[REDACTED]" in caplog.text
    assert "PRIVATE PDF CONTENT" not in caplog.text
    assert "PRIVATE PROMPT CONTENT" not in caplog.text
    assert "PRIVATE EVIDENCE CONTENT" not in caplog.text
    assert "private-token-value" not in caplog.text


def test_normal_logs_redact_extra_fields_and_exception_credentials(caplog) -> None:
    create_app(Settings(_env_file=None))
    logger = logging.getLogger("redacted_backend.security_exception_test")

    with caplog.at_level(logging.ERROR):
        try:
            raise RuntimeError(
                "PRIVATE EXCEPTION BODY provider rejected sk-private12345"
            )
        except RuntimeError:
            logger.exception(
                "analysis failed with Bearer private-bearer-token",
                extra={
                    "prompt": "PRIVATE EXTRA PROMPT",
                    "metadata": {"retrieved_excerpt": "PRIVATE EXTRA EVIDENCE"},
                },
            )

    assert "analysis failed" in caplog.text
    assert "[REDACTED]" in caplog.text
    assert "PRIVATE EXTRA PROMPT" not in caplog.text
    assert "PRIVATE EXTRA EVIDENCE" not in caplog.text
    assert "private-bearer-token" not in caplog.text
    assert "sk-private12345" not in caplog.text
    assert "PRIVATE EXCEPTION BODY" not in caplog.text
    assert caplog.records[0].prompt == "[REDACTED]"
    assert caplog.records[0].metadata == {"retrieved_excerpt": "[REDACTED]"}
