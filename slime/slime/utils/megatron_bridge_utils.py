from contextlib import contextmanager

try:
    from megatron.core.utils import unwrap_model
except ImportError:
    try:
        from megatron.training.utils import unwrap_model
    except ImportError:
        unwrap_model = None


def patch_hf_config_for_megatron_bridge(hf_config):
    configs = []
    seen_config_ids = set()

    def add_config(config):
        if config is None or id(config) in seen_config_ids:
            return
        seen_config_ids.add(id(config))
        configs.append(config)

    add_config(hf_config)
    add_config(getattr(hf_config, "config", None))

    for config in list(configs):
        add_config(getattr(config, "text_config", None))

    for config in configs:
        rope_params = getattr(config, "rope_parameters", None) or getattr(config, "rope_scaling", None)
        if isinstance(rope_params, dict) and "rope_theta" in rope_params and not hasattr(config, "rope_theta"):
            config.rope_theta = rope_params["rope_theta"]

    return hf_config


def patch_auto_bridge_hf_config(bridge):
    hf_pretrained = getattr(bridge, "hf_pretrained", None)
    if hf_pretrained is not None:
        patch_hf_config_for_megatron_bridge(hf_pretrained)

    return bridge


@contextmanager
def preserve_critic_value_head(model):
    """Load a causal-LM backbone without mapping its vocabulary head to a scalar head."""
    if getattr(model[0], "role", "actor") != "critic":
        yield
        return
    saved = []
    try:
        for module in unwrap_model(model):
            head = getattr(module, "output_layer", None)
            if head is not None:
                saved.append((module, head))
                # Bridge builds import tasks from the target's named parameters.
                # The already initialized value head has no HF counterpart.
                module.output_layer = None
        yield
    finally:
        for module, head in saved:
            module.output_layer = head


@contextmanager
def patch_megatron_model(model):
    unwrapped_model = unwrap_model(model)[0]
    model_config = unwrapped_model.config
    attribute_was_added = False
    if not hasattr(model_config, "share_embeddings_and_output_weights"):
        model_config.share_embeddings_and_output_weights = unwrapped_model.share_embeddings_and_output_weights
        attribute_was_added = True

    try:
        yield
    finally:
        if attribute_was_added:
            delattr(model_config, "share_embeddings_and_output_weights")


def configure_bridge_recomputation(provider, args):
    """Apply explicit memory settings when Bridge supplies the model config."""
    for name in ("recompute_granularity", "recompute_method", "recompute_num_layers"):
        value = getattr(args, name, None)
        if value is not None:
            setattr(provider, name, value)
