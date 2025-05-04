"""Exploring pydantic models with generics

It's very cool. The validation does work properly, even at type checker-time.
"""

import code
from typing import Generic, TypeVar, Optional, Literal
from pydantic import BaseModel

# generic type variable for caches
C = TypeVar("C")

AgeNumeral = Literal["I", "II", "III", "IV"]


class GenericCache(BaseModel, Generic[C]):
    frame_data: dict[int, C] = {}


class AgeNumeralCache(GenericCache[Optional[AgeNumeral]]):
    pass


a = AgeNumeralCache(frame_data={42: "I"})
b = AgeNumeralCache(frame_data={42: None})
# c = AgeNumeralCache(frame_data={42: "foo"}) # errors (w/ type checker!)
d = AgeNumeralCache.model_validate_json('{"frame_data": {"42": "I"} }')
# e = AgeNumeralCache.model_validate_json('{"frame_data": {"42": "hi"} }')  # errors
f = AgeNumeralCache.model_validate_json('{"blah": "blargh"}')  # no error! just ignores
code.interact(local=dict(globals(), **locals()))
