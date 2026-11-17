import pytest
import torch

from slime.utils.ppo_utils import stepwise_segment_gae, stepwise_value_mask


@pytest.mark.parametrize(
    "value_type,expected",
    [
        ("last_token", [0.25, 0.25, 0.0, 0.7, 0.7]),
        ("step_average", [0.3, 0.3, 0.0, 0.8, 0.8]),
    ],
)
def test_stepwise_values_skip_observations_and_apply_cross_segment_discount(
    value_type, expected
):
    values = torch.tensor([0.2, 0.4, 999.0, 0.1, 0.3])
    mask = torch.tensor([1, 1, 0, 1, 1])
    adv, ret = stepwise_segment_gae(
        values, mask, 1.0, 2, lambd=0.5, value_type=value_type
    )
    torch.testing.assert_close(adv, torch.tensor(expected) / 4)
    assert ret[2] == 0
    torch.testing.assert_close(ret[mask.bool()], (adv + values)[mask.bool()])


def test_stepwise_critic_mask_selects_run_ends_and_respects_chunk_limit():
    mask = torch.tensor([1, 1, 1, 0, 1, 1])
    assert stepwise_value_mask(mask).tolist() == [0, 0, 1, 0, 0, 1]
    assert stepwise_value_mask(mask, 2).tolist() == [0, 1, 1, 0, 0, 1]
