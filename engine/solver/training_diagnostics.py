"""Training diagnostics: validation and sanitization functions for loss/gradient sanity.

Functions report anomalies via exceptions or logged warnings.  Some functions
may mutate gradients in place when instructed (see ``inspect_gradients``).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator

import torch


@dataclass
class NanGradientTracker:
    """Tracks NaN-gradient zeroing events with a configurable upper limit.

    Each call to :meth:`record` increments the event counter.  When
    *max_events* is reached, ``record`` returns ``False`` — the caller
    should raise instead of continuing to zero grads.

    The tracker also exposes :meth:`pop_firsts` for each batch of
    ``record`` calls — callers can use this to save a full diagnostic
    snapshot the first time each pattern produces a NaN.
    """

    max_events: int
    _event_count: int = field(default=0, init=False)
    _first_seen: dict[str, int] = field(default_factory=dict, init=False)
    _pending_firsts: list[str] = field(default_factory=list, init=False)

    def record(self, name: str, global_step: int) -> bool:
        """Register one NaN-zeroing event for *name*.  Returns ``False`` when the
        total number of events exceeds *max_events*.
        """
        self._event_count += 1
        if name not in self._first_seen:
            self._first_seen[name] = global_step
            self._pending_firsts.append(name)
        return self._event_count <= self.max_events

    @property
    def count(self) -> int:
        """Total number of NaN-zeroing events recorded so far."""
        return self._event_count

    def pop_firsts(self) -> list[str]:
        """Return (and clear) param names that had their **first** NaN event in
        the most recent batch of ``record`` calls.
        """
        result = self._pending_firsts
        self._pending_firsts = []
        return result


def validate_max_optimizer_steps(value) -> int | None:
    """Return validated int or None. Raise ValueError for invalid types/values.

    Accepts:
        None          → None (unlimited)
        positive int  → int (cap)

    Rejects:
        0, -1, bool, float, str, list, tuple, etc.
    """
    if value is None:
        return None

    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(
            f"max_optimizer_steps must be None or int, got {type(value).__name__}: {value!r}"
        )
    if value < 1:
        raise ValueError(
            f"max_optimizer_steps must be >= 1 or None, got {value}"
        )
    return value


def validate_gradient_accumulation_steps(value) -> int:
    """Return validated positive int (1 = no accumulation).

    Accepts None (→ 1) or positive int. Rejects 0, negatives, bool, float,
    str, etc. — same strictness as :func:`validate_max_optimizer_steps`.
    """
    if value is None:
        return 1

    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(
            "gradient_accumulation_steps must be None or int, got "
            f"{type(value).__name__}: {value!r}"
        )
    if value < 1:
        raise ValueError(
            f"gradient_accumulation_steps must be >= 1 or None, got {value}"
        )
    return value


def raise_for_nonfinite_losses(
    loss_dict: dict[str, torch.Tensor],
    *,
    epoch: int,
    step: int,
    global_step: int,
) -> None:
    """Raise FloatingPointError if any loss component is non-finite.

    Error message includes: key, value dtype, device, NaN/+Inf/-Inf counts.
    Called BEFORE loss aggregation and backward, so optimizer never mutates.
    """
    for key, tensor in loss_dict.items():
        if torch.isfinite(tensor).all():
            continue

        flat = tensor.detach().float().flatten()
        nan_count = torch.isnan(flat).sum().item()
        pos_inf_count = torch.isposinf(flat).sum().item()
        neg_inf_count = torch.isneginf(flat).sum().item()
        raise FloatingPointError(
            f"Non-finite loss component at epoch={epoch} step={step} global_step={global_step}: "
            f"key='{key}' dtype={tensor.dtype} device={tensor.device} "
            f"shape={tuple(tensor.shape)} "
            f"NaN={int(nan_count)} +Inf={int(pos_inf_count)} -Inf={int(neg_inf_count)}"
        )


def raise_for_nonfinite_total(
    total,
    *,
    epoch: int,
    step: int,
    global_step: int,
) -> None:
    """Raise FloatingPointError if total loss is non-finite; TypeError if non-scalar.

    Called BEFORE .backward(), so the model grads are not yet changed.
    """
    if not isinstance(total, torch.Tensor):
        raise TypeError(
            f"Loss total must be a torch.Tensor, got {type(total).__name__}"
        )
    if total.ndim != 0:
        raise TypeError(
            f"Loss total must be scalar (0-d tensor), got ndim={total.ndim} shape={tuple(total.shape)}"
        )

    if torch.isfinite(total):
        return

    raise FloatingPointError(
        f"Non-finite loss total at epoch={epoch} step={step} global_step={global_step}: "
        f"value={total.item()} dtype={total.dtype} device={total.device}"
    )


def inspect_gradients(
    named_parameters: Iterator[tuple[str, torch.nn.Parameter]],
    *,
    fail_on_zero_grad: bool = False,
    epoch: int,
    step: int,
    global_step: int,
    nan_zero_patterns: frozenset[str] | None = None,
    nan_tracker: NanGradientTracker | None = None,
) -> tuple[float, list[str]]:
    """Check gradient finiteness; optionally zero NaN/Inf grads for matched params.

    Returns ``(aggregate_norm, zeroed_names)`` where *zeroed_names* lists the
    param names whose non-finite gradients were zeroed (excluding those that
    would have raised).

    Parameters named in *nan_zero_patterns* (substring match) have their NaN/Inf
    gradients zeroed with a logged warning instead of raising.  Only the
    non-finite elements are zeroed — finite elements in the same tensor are
    preserved.  All other parameters still raise on non-finite gradients.

    If *nan_tracker* is provided, each zeroed gradient is recorded via
    :meth:`NanGradientTracker.record`.  When the tracker's event limit is
    exceeded, a ``FloatingPointError`` is raised instead of zeroing.

    Raises:
        FloatingPointError: NaN or Inf gradient detected on a non-exempted param,
            or the *nan_tracker* event limit has been exceeded.
        RuntimeError: ``fail_on_zero_grad=True`` and the aggregate gradient norm is 0.
    """
    aggregate_sq = 0.0
    nan_zero_patterns = nan_zero_patterns or frozenset()
    zeroed_params: list[str] = []

    all_none = True
    for name, param in named_parameters:
        if param.grad is None:
            continue
        all_none = False

        grad = param.grad
        if grad.is_sparse:
            grad = grad.coalesce()
            values = grad.values()
        else:
            values = grad

        if not torch.isfinite(values).all():
            nan_count = torch.isnan(values).sum().item()
            inf_count = torch.isinf(values).sum().item()

            exempted = any(pat in name for pat in nan_zero_patterns)
            if exempted:
                if nan_tracker is not None and not nan_tracker.record(name, global_step):
                    raise FloatingPointError(
                        f"NaN gradient zero-event limit ({nan_tracker.max_events}) exceeded "
                        f"at epoch={epoch} step={step} global_step={global_step}: "
                        f"param='{name}' dtype={values.dtype} device={values.device} "
                        f"shape={tuple(values.shape)} "
                        f"NaN={int(nan_count)} Inf={int(inf_count)}"
                    )
                with torch.no_grad():
                    values.masked_fill_(~torch.isfinite(values), 0.0)
                zeroed_params.append(
                    f"  {name}  shape={tuple(values.shape)}  "
                    f"NaN={int(nan_count)}  Inf={int(inf_count)}"
                )
                continue

            raise FloatingPointError(
                f"Non-finite gradient at epoch={epoch} step={step} global_step={global_step}: "
                f"param='{name}' dtype={values.dtype} device={values.device} "
                f"shape={tuple(values.shape)} "
                f"NaN={int(nan_count)} Inf={int(inf_count)}"
            )

        aggregate_sq += values.float().pow(2).sum().item()

    if zeroed_params:
        print(
            f"[WARN] Zeroed non-finite gradients at "
            f"epoch={epoch} step={step} global_step={global_step}:\n"
            + "\n".join(zeroed_params)
        )

    aggregate_norm = aggregate_sq ** 0.5

    if fail_on_zero_grad and aggregate_norm == 0.0:
        if all_none:
            msg = (
                f"All parameters have no gradients (aggregate zero grad) "
                f"at epoch={epoch} step={step} global_step={global_step}"
            )
        else:
            msg = (
                f"Aggregate gradient norm is zero "
                f"at epoch={epoch} step={step} global_step={global_step}"
            )
        raise RuntimeError(msg)

    return float(aggregate_norm), zeroed_params
