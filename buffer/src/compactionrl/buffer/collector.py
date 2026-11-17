"""Harbor stream collection and optional rollout-relative baseline grouping."""

import json
import math
import uuid

from .schema import finalize_rollout


async def collect(client, config, model, task_id):
    uid = str(uuid.uuid4())
    url = config["generation_url"]
    request = dict(
        config["agent"],
        instance_id=task_id,
        uid=uid,
        model=model,
        tokenizer_path=model,
        base_url=url,
        timeout=config.get("timeout_seconds", 1200) / 60,
        max_retries=0,
        use_generate=True,
        enable_tito=True,
    )
    result = None
    keys = []
    try:
        async with client.stream(
            "POST", config["harbor_url"] + "/run_worker", json=request
        ) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line.strip():
                    continue
                message = json.loads(line)
                if message.get("type") == "error":
                    raise RuntimeError(str(message.get("data")))
                if message.get("type") == "result":
                    result = message["data"]
        if result is None:
            raise ValueError("Harbor stream ended without a result")
        reward = result.get("reward", result.get("raw_reward"))
        if (
            reward is None
            or not math.isfinite(float(reward))
            or not 0 <= float(reward) <= 1
        ):
            raise RuntimeError(
                f"Harbor did not return a verifier reward: {result.get('rollout_status_detail', result.get('rollout_status'))}"
            )
        records = (
            result.get("split_rollout_results")
            or result.get("metadata", {}).get("split_rollout_results")
            or [result]
        )
        # In the retained Harbor full mode, summary snapshots repeat the final
        # generated summary already present in the corresponding main snapshot.
        records = [r for r in records if r.get("messages_list_tag") != "summary"]
        if not records:
            raise ValueError("No execution snapshots")
        for r in records:
            r["reward"] = reward
            r["instance_id"] = task_id
            keys.append(r["uid"])
            response = await client.post(url + "/tokens", json={"main_key": r["uid"]})
            response.raise_for_status()
            r["token_info"] = response.json()
            r.setdefault("metadata", {})["summary_training"] = config["agent"].get(
                "summary_training", "full"
            )
        return finalize_rollout(records, uid)
    finally:
        # Token data has been copied into records before returning. Release only
        # this request's keys; errors are non-fatal because TiTO also expires them.
        for key in set(keys):
            try:
                await client.post(url + "/release", json={"main_key": key})
            except Exception:
                pass


def assign_group_advantages(rollouts):
    """Normalize independent rollout rewards, then broadcast to their segments."""
    if len(rollouts) < 2:
        raise ValueError("The group-relative baseline needs >=2 rollouts")
    rewards = [float(rows[0]["reward"]) for rows in rollouts]
    mean = sum(rewards) / len(rewards)
    std = math.sqrt(sum((r - mean) ** 2 for r in rewards) / len(rewards))
    group_id = str(uuid.uuid4())
    for rows, reward in zip(rollouts, rewards):
        for record in rows:
            record["metadata"].update(
                group_id=group_id,
                group_size=len(rollouts),
                group_advantage=(reward - mean) / (std + 1e-6),
            )
    return [record for rows in rollouts for record in rows]
