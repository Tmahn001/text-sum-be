"""API route for verification: where did this sentence come from, and does it hold up?

Deliberately a separate endpoint from /summarize: NLI is the slow part, so the
summary can render immediately and the verification arrives behind it.
"""
from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException

from ..schemas import VerifyRequest, VerifyResponse
from ..services import verification as verification_service

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/verify", response_model=VerifyResponse)
def verify(payload: VerifyRequest):
    """Link each summary sentence to its evidence and flag unsupported ones."""
    try:
        return verification_service.verify_summary(
            source=payload.source,
            summary=payload.summary,
            context=payload.context,
            top_k=payload.top_k,
            threshold=payload.threshold,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        # Most likely a model that isn't downloaded or doesn't fit in memory.
        # The frontend degrades to the plain summary, so a 503 is enough here.
        logger.exception("Verification failed")
        raise HTTPException(
            status_code=503, detail="Verification is unavailable right now."
        ) from exc
