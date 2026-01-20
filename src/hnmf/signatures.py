from typing import Any, TypedDict


class DiscrimSample(TypedDict):
    sample: Any
    node: int
    node_value: float
    others_value: float
