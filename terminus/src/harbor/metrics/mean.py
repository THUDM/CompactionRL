from harbor.metrics.base import BaseMetric


class Mean(BaseMetric[dict[str, float | int]]):
    @staticmethod
    def _clip_reward(value: float | int) -> float:
        return max(0.0, min(1.0, float(value)))

    def compute(
        self, rewards: list[dict[str, float | int] | None]
    ) -> dict[str, float | int]:
        values = []

        for reward in rewards:
            if reward is None:
                values.append(0)
            elif len(reward) != 1:
                raise ValueError(
                    f"Expected exactly one key in reward dictionary, got {len(reward)}"
                )
            else:
                values.extend(self._clip_reward(value) for value in reward.values())

        return {"mean": sum(values) / len(values)}
