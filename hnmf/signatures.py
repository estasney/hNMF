from typing import Literal, TypedDict, Any

Sign = Literal['positive', 'negative', 'abs']

class DiscrimSample(TypedDict):
    sample: Any
    node: int
    node_value: float
    others_value: float
