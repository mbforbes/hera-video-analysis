"""Converts from one-file-per-frame cache to single-file-per-type cache."""

import code
import glob
import os
from typing import Optional

from mbforbes_python_utils import read, write
from pydantic import BaseModel

from main import AgeNumeral, AgeText, AgeNumeralCache, AgeTextCache, FrameResult, FrameResultCache


# old classes


class AgeNumeralResult(BaseModel):
    """Somewhat redundant class just so we have a top-level pyandtic type for the cache format."""

    age: Optional[AgeNumeral]


class AgeTextResult(BaseModel):
    """Somewhat redundant class just so we have a top-level pyandtic type for the cache format."""

    age: Optional[AgeText]


def convert_agenumeralresult(out_dir: str):
    frame_result_files = glob.glob(os.path.join(out_dir, "frame_*_agenumeralresult.json"))
    cache = AgeNumeralCache()
    for f in frame_result_files:
        anr = AgeNumeralResult.model_validate_json(read(f))
        frame = int(os.path.basename(f).split("_")[1])
        cache.frame2data[frame] = anr.age
    write(os.path.join(out_dir, "age_numeral_cache.json"), cache.model_dump_json())


def main() -> None:
    convert_agenumeralresult("output/_5CjPfg93SE")


if __name__ == "__main__":
    main()
