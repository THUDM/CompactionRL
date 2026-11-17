"""slime adapter with exact token transport and explicit train_metadata."""

import os
import time
import httpx
from compactionrl.buffer.schema import validate_tokens


def record_to_sample(record, index):
    from slime.utils.types import Sample

    info = record["token_info"]
    first, count = validate_tokens(info)
    meta = dict(record["metadata"])
    if meta["optimized_tokens"] != count:
        raise ValueError("Optimized token metadata disagrees with the loss mask")
    return Sample(
        index=index,
        prompt=record["uid"],
        tokens=info["input_ids"],
        response_length=len(info["input_ids"]) - first,
        rollout_log_probs=info["logprobs"][first:],
        loss_mask=info["loss_mask"][first:],
        reward=float(record["reward"]),
        status=Sample.Status.COMPLETED,
        metadata=meta,
        train_metadata=meta,
        weight_versions=info.get("weight_versions", []),
        messages=record["messages"],
    )


def generate_rollout(args, rollout_id, data_source, evaluation=False):
    from slime.rollout.base_types import RolloutFnTrainOutput

    if evaluation:
        raise ValueError("Use the standalone Harbor evaluation entry point")
    if args.n_samples_per_prompt != 1:
        raise ValueError(
            "Batching uses one segment per sample; rollout groups are formed by buffer group_size"
        )
    deadline = time.monotonic() + float(
        os.environ.get("COMPACTIONRL_BATCH_TIMEOUT", "1800")
    )
    with httpx.Client(timeout=30) as client:
        base = args.rollout_buffer_url.rstrip("/")
        client.post(
            base + "/start_rollout", json={"hf_checkpoint": args.hf_checkpoint}
        ).raise_for_status()
        while time.monotonic() < deadline:
            response = client.post(
                base + "/get_rollout_data", json={"batch_size": args.global_batch_size}
            )
            response.raise_for_status()
            batch = response.json()
            if batch["success"]:
                samples = [
                    record_to_sample(r, rollout_id * args.global_batch_size + i)
                    for i, r in enumerate(batch["data"])
                ]
                client.post(
                    base + "/buffer/ack", json={"lease": batch["lease"]}
                ).raise_for_status()
                return RolloutFnTrainOutput(
                    samples=[[s] for s in samples],
                    metrics={
                        "segments": len(samples),
                        "optimized_tokens": sum(sum(s.loss_mask) for s in samples),
                        "summary_optimized_tokens": sum(
                            s.metadata.get("summary_optimized_tokens", 0)
                            for s in samples
                        ),
                        "mean_summary_length": sum(
                            s.metadata.get("summary_optimized_tokens", 0)
                            for s in samples
                        )
                        / max(
                            1, sum(s.metadata.get("summary_count", 0) for s in samples)
                        ),
                        "compacted_segments": sum(
                            s.metadata["segment_count"] > 1 for s in samples
                        ),
                    },
                )
            if batch.get("collector", {}).get("failed", 0) >= 5 and not batch[
                "collector"
            ].get("completed"):
                raise RuntimeError(
                    f"Collector failed: {batch['collector']['last_error']}"
                )
            time.sleep(0.5)
    raise TimeoutError("Training batch deadline exceeded; inspect collector health")
