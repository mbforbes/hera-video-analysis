from dataclasses import dataclass
from typing import Optional


@dataclass
class FrameResult:
    wood: Optional[int]


f = FrameResult(wood="hi")  # this works at runtime
print(f)  # prints "FrameResult(wood='hi')"
