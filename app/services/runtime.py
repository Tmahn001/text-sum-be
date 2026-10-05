"""CPU and memory tuning shared by every model this service loads.

The deployment target is a 1 vCPU / 2GB server with no GPU, which fp32 weights
do not fit: BART alone is ~1.2GB, and the verification models add ~800MB. Three
measures bring that back under budget:

* **Dynamic int8 quantization.** Linear layers — the bulk of every transformer —
  are stored as int8 and dequantized per operation. Roughly a 3-4x cut in weight
  memory and usually *faster* on CPU, because int8 matmuls need less bandwidth.
  Embedding tables stay fp32 (quantizing them needs a calibration pass and costs
  more accuracy than it saves here).
* **One inference at a time.** Transient activations, not weights, cause the
  spikes that get the container OOM-killed. A single global slot means two
  requests can never generate at once.
* **Single-threaded torch.** On one core, extra threads add per-thread arenas and
  contention for no throughput.
"""
from __future__ import annotations

import contextlib
import logging
import threading

from ..config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

_configured = False
_configure_lock = threading.Lock()
# One model inference at a time, process-wide. Weights are shared; activations
# are not, so this is what keeps peak memory predictable.
_inference_slot = threading.BoundedSemaphore(1)


def configure_torch() -> None:
    """Apply thread limits once per process. Safe to call repeatedly."""
    global _configured
    with _configure_lock:
        if _configured:
            return
        _configured = True
        threads = settings.torch_num_threads
        if threads <= 0:
            return
        try:
            import torch

            torch.set_num_threads(threads)
            logger.info("torch limited to %d thread(s)", threads)
        except Exception as exc:  # torch missing or refuses — not fatal
            logger.warning("Could not set torch thread count: %s", exc)


def quantize(model):
    """Return `model` with int8 Linear layers, or unchanged if that isn't possible.

    Quantization is a memory/accuracy trade, so it is controlled by
    QUANTIZE_MODELS and skipped on GPU, where there is no reason for it.
    """
    if not settings.quantize_models:
        return model
    try:
        import torch

        if next(model.parameters()).device.type != "cpu":
            return model
        before = _param_bytes(model)
        quantized = torch.ao.quantization.quantize_dynamic(
            model, {torch.nn.Linear}, dtype=torch.qint8
        )
        logger.info(
            "Quantized %s: %.0fMB -> ~%.0fMB of weights",
            type(model).__name__, before / 1e6, _param_bytes(quantized) / 1e6,
        )
        return quantized
    except Exception as exc:  # fall back to fp32 rather than fail the request
        logger.warning("Dynamic quantization unavailable (%s); using fp32", exc)
        return model


def _param_bytes(model) -> int:
    """Approximate resident size of a module's parameters and buffers."""
    total = 0
    for p in model.parameters():
        total += p.numel() * p.element_size()
    for b in model.buffers():
        total += b.numel() * b.element_size()
    # Quantized Linear weights live in packed params, invisible to .parameters().
    for module in model.modules():
        weight = getattr(module, "_packed_params", None)
        if weight is not None:
            try:
                total += module.weight().numel()  # int8: 1 byte per element
            except Exception:
                pass
    return total


def log_peak_memory(label: str) -> None:
    """Log peak resident memory, so the effect of these settings is visible.

    Compare the numbers after each model load against the container limit; this
    is the quickest way to tell whether a config change actually helped.
    """
    try:
        import resource
        import sys

        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # Linux reports kilobytes, macOS bytes.
        megabytes = peak / 1024 if sys.platform != "darwin" else peak / (1024 * 1024)
        logger.info("Peak memory after %s: %.0f MB", label, megabytes)
    except Exception as exc:  # never fail a request over a log line
        logger.debug("Could not read peak memory: %s", exc)


@contextlib.contextmanager
def inference_slot(what: str = "inference"):
    """Hold the single inference slot, or raise RuntimeError if the wait is too long."""
    timeout = settings.inference_timeout_seconds
    acquired = _inference_slot.acquire(timeout=timeout) if timeout > 0 else _inference_slot.acquire()
    if not acquired:
        raise RuntimeError(
            f"Server busy: could not start {what} within {timeout}s — "
            "another model run is still holding the inference slot."
        )
    try:
        yield
    finally:
        _inference_slot.release()
