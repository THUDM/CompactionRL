# Buffer and token contract

The collector calls Harbor `/run_worker` and consumes newline-delimited JSON
until the final `result` event. Only complete valid results with verifier rewards
in [0,1] enter training. Infrastructure sentinels are rejected. A full training
snapshot may contain execution tokens followed by summary tokens; the historical
standalone summary export duplicates those tokens and is excluded in full mode.

Each stored segment contains `uid`, `instance_id`, `messages`, `reward`,
`token_info` and `metadata`. Token info carries aligned `input_ids`, `loss_mask`,
`logprobs` and generation weight versions. Metadata carries rollout ID, segment
order/count, optimized-token count, response start and `later_assistant_tokens`.
Token positions copied into a resumed prompt have zero mask; newly generated
summary tokens retain their generation mask. No text-tokenization fallback exists.

The complete rollout receives one reward. N_after is computed in reverse segment
order, before any training batch split. One prompt produces one rollout; there is
no wait for unrelated same-prompt samples on the PPO path.

SQLite persists records and delivery states:

- `/buffer/write`: atomically inserts a complete validated rollout. Repeating an
  identical UID is idempotent; changing its contents raises an error.
- `/get_rollout_data`: leases exactly the requested number of ready segments, or
  returns `success=false` without consuming a partial batch.
- `/buffer/ack`: marks an assembled batch delivered. Expired unacknowledged leases
  become available again. This is delivery acknowledgement, not an optimizer
  transaction; a crash after acknowledgement can require collecting fresh data.
- `/start_rollout`: starts bounded background collection once.
- `/buffer/stats`: reports queue counts and the latest collector error.

Completed rows are retained for audit and UID deduplication. Rotate the database
between experiments. The collector applies backpressure to active rows, but disk
storage is not bounded automatically. The default service binds to loopback; use
an appropriate private network arrangement when moving to multiple hosts.

The bundled TiTO executable serializes requests by main_key, preserves raw generated IDs and
rejects prefix divergence. It does not promise distributed routing affinity for
multiple SGLang replicas; the smoke configuration uses one rollout engine.
The executable uses the project's Python 3.12 environment; see
[binary requirements](../../terminus/bin/README.md). The historical Rust implementation is
not required or distributed.
