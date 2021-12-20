from typing import TypedDict, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Any, Literal

    Sign = Literal["positive", "negative", "abs"]


class DiscrimSample(TypedDict):
    sample: Any
    node: int
    node_value: float
    others_value: float
