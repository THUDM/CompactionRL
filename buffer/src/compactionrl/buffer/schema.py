"""Strict wire contract; token counts are derived from the effective training mask."""

import math
from copy import deepcopy


def validate_tokens(info):
    ids, mask, logp = (info[k] for k in ("input_ids", "loss_mask", "logprobs"))
    if not ids or len(ids) != len(mask) or len(ids) != len(logp):
        raise ValueError(
            "Token IDs, masks and log probabilities must have equal nonzero lengths"
        )
    if any(type(t) is not int or t < 0 for t in ids):
        raise ValueError("Invalid token ID")
    if any(m not in (0, 1) for m in mask) or not any(mask):
        raise ValueError("Expected a binary mask with at least one optimized token")
    if any(not math.isfinite(float(x)) for x in logp):
        raise ValueError("Non-finite rollout log probability")
    first = next(i for i, m in enumerate(mask) if m)
    if first == 0:
        raise ValueError("A generated token needs at least one conditioning token")
    return first, sum(mask)


def finalize_rollout(records, rollout_id):
    """Assign one terminal reward and compute N_after before segments enter batches.

    Each record must be a non-overlapping trainable context snapshot. Standalone
    summary snapshots duplicating a full main snapshot must be removed upstream.
    Never infer optimized lengths by re-tokenizing assistant text.
    """
    if not records:
        raise ValueError("Empty rollout")
    result = deepcopy(records)
    uids = [r["uid"] for r in result]
    if len(uids) != len(set(uids)):
        raise ValueError("Duplicate segment UID in one rollout")
    reward = float(result[-1]["reward"])
    if not math.isfinite(reward) or reward < 0 or reward > 1:
        raise ValueError(
            "Expected verifier reward in [0, 1], not an infrastructure sentinel"
        )
    remaining = 0
    for index in reversed(range(len(result))):
        r = result[index]
        if float(r["reward"]) != reward:
            raise ValueError("Segments must share their rollout's terminal reward")
        meta = {**(r.pop("meta_data", None) or {}), **r.get("metadata", {})}
        r["metadata"] = dict(meta)
        if meta.get("buffer_filter_force_invalid"):
            raise ValueError("Invalid continuation segment")
        first, count = validate_tokens(r["token_info"])
        r["metadata"].update(
            rollout_id=rollout_id,
            segment_index=index,
            segment_count=len(result),
            optimized_tokens=count,
            later_assistant_tokens=remaining,
            response_start=first,
        )
        r["metadata"]["summary_optimized_tokens"] = sum(
            sum(r["token_info"]["loss_mask"][span["start"] : span["end"]])
            for span in r["token_info"].get("generated_spans", [])
            if span.get("summary")
        )
        r["metadata"]["summary_count"] = sum(
            bool(span.get("summary"))
            for span in r["token_info"].get("generated_spans", [])
        )
        remaining += count
    return result
