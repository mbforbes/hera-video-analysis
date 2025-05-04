"""Converts from one-file-per-frame cache to single-file-per-type cache.

As each video is run either before or after, this doesn't look for existing caches.
"""

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
    if len(frame_result_files) == 0:
        print("No age numeral files")
        return

    for f in frame_result_files:
        anr = AgeNumeralResult.model_validate_json(read(f))
        frame = int(os.path.basename(f).split("_")[1])
        cache.frame2data[frame] = anr.age

    print(f"{len(frame_result_files)} files, {len(cache.frame2data)} cache")
    if len(frame_result_files) != len(cache.frame2data):
        print("unequal")
        return

    write(os.path.join(out_dir, "age_numeral_cache.json"), cache.model_dump_json())
    for f in frame_result_files:
        os.remove(f)


def convert_agetextresult(out_dir: str):
    frame_result_files = glob.glob(os.path.join(out_dir, "frame_*_agetextresult.json"))
    cache = AgeTextCache()
    if len(frame_result_files) == 0:
        print("No age text files")
        return

    for f in frame_result_files:
        atr = AgeTextResult.model_validate_json(read(f))
        frame = int(os.path.basename(f).split("_")[1])
        cache.frame2data[frame] = atr.age

    print(f"{len(frame_result_files)} files, {len(cache.frame2data)} cache")
    if len(frame_result_files) != len(cache.frame2data):
        print("unequal")
        return

    write(os.path.join(out_dir, "age_text_cache.json"), cache.model_dump_json())
    for f in frame_result_files:
        os.remove(f)


def convert_frameresult(out_dir: str):
    frame_result_files = glob.glob(os.path.join(out_dir, "frame_*_results.json"))
    cache = FrameResultCache()
    if len(frame_result_files) == 0:
        print("No frame result files")
        return

    for f in frame_result_files:
        fr = FrameResult.model_validate_json(read(f))
        frame = int(os.path.basename(f).split("_")[1])
        cache.frame2data[frame] = fr

    print(f"{len(frame_result_files)} files, {len(cache.frame2data)} cache")
    if len(frame_result_files) != len(cache.frame2data):
        print("unequal")
        return

    write(os.path.join(out_dir, "frame_result_cache.json"), cache.model_dump_json())
    for f in frame_result_files:
        os.remove(f)


def main() -> None:
    subdirs = [
        "output/1g85tbr2bfg",
        "output/2DJE13bTCZw",
        "output/8YP2Jm9YZzU",
        # "output/IpVxkZvNOmo",  # ran with new format
        "output/UCR6eDChhe0",
        "output/_5CjPfg93SE",
        "output/gaONBkqLryA",
        "output/naNjOZYbAE4",
        "output/rR2nleqmyuM",
        "output/x0v21gl5y2E",
        "output/y5D5FhyDRK4",
    ]
    for subdir in subdirs:
        print(subdir)
        convert_agenumeralresult(subdir)
        convert_agetextresult(subdir)
        convert_frameresult(subdir)
        print()


if __name__ == "__main__":
    main()
