import argparse
import code
import json
import os
import sys
from typing import Callable, Literal, Optional, TypeVar, Type, Any

import cv2
from dotenv import load_dotenv
from google import genai
from google.genai import types
from imgcat import imgcat
from mbforbes_python_utils import read, write
import numpy as np
from pydantic import BaseModel
import yt_dlp

# Create a generic type variable for the Pydantic model
T = TypeVar("T", bound=BaseModel)

# gemini client
load_dotenv()
api_key = os.getenv("GOOGLE_GEMINI_API_KEY")
if not api_key:
    raise RuntimeError("GOOGLE_GEMINI_API_KEY not found in environment.")
client = genai.Client(api_key=api_key)

# gross mutable global sorryyy
RUNNING_COST = 0.0


class Rectangle(BaseModel):
    """Represents a rectangular region in a 1920 x 1080 image, incl. name and OCR function."""

    name: str
    """Key in the intermediate results dict."""
    x: int
    y: int
    w: int
    h: int
    ocr_fn: Callable[[np.ndarray], Any]
    """Function from crop to value in the intermediate results dict."""


AgeNumeral = Literal["I", "II", "III", "IV"]
AgeText = Literal["Dark Age", "Feudal Age", "Castle Age", "Imperial Age"]

ORDERED_AGE_TEXTS: list[AgeText] = [
    "Dark Age",
    "Feudal Age",
    "Castle Age",
    "Imperial Age",
]

# constituent classes for FrameResult:


class Resources(BaseModel):
    wood: Optional[int]
    food: Optional[int]
    gold: Optional[int]
    stone: Optional[int]


class Villagers(BaseModel):
    wood: Optional[int]
    food: Optional[int]
    gold: Optional[int]
    stone: Optional[int]
    total: Optional[int]
    idle: Optional[int]


class Population(BaseModel):
    current: Optional[int]
    maximum: Optional[int]


class Age(BaseModel):
    numeral: Optional[AgeNumeral]
    """Shows the current age"""
    text: Optional[AgeText]
    """Shows the current age, OR if advancing, shows the next age."""


class FrameMetadata(BaseModel):
    home: Optional[str]
    away: Optional[str]
    desc: Optional[str]
    metadate: Optional[str]
    wins: Optional[int]
    losses: Optional[int]


class Gametime(BaseModel):
    hours: Optional[int]
    minutes: Optional[int]
    seconds: Optional[int]


class PlayerInfo(BaseModel):
    number: Optional[int]
    name: Optional[str]
    score_individual: Optional[int]
    score_team: Optional[int]
    age_numeral: Optional[AgeNumeral]


class SelectedInfo(BaseModel):
    player_name: Optional[str]
    civilization: Optional[str]


class FrameResult(BaseModel):
    resources: Resources
    villagers: Villagers
    population: Population
    age: Age
    metadata: FrameMetadata
    gametime: Gametime
    top_player: PlayerInfo
    bottom_player: PlayerInfo
    selected_info: SelectedInfo


class AgeClickIndex(BaseModel):
    """Saved to point results of binary search towards FrameResults (on disk)"""

    ageclick2frame: dict[AgeText, int]


def display(frame_bgr: cv2.typing.MatLike) -> None:
    imgcat(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))


def download_video(
    url: str, output_dir: str = "./videos"
) -> tuple[Optional[str], Optional[dict[str, Any]]]:
    """Download a YouTube video and return its (local path, metadata dict)."""
    os.makedirs(output_dir, exist_ok=True)

    ydl_opts = {
        "format": "bestvideo[height<=1080][ext=mp4]/best[height<=1080]",  # untested
        "outtmpl": os.path.join(output_dir, "%(id)s.%(ext)s"),
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        print(f"Downloading video: {url}")
        info = ydl.extract_info(url, download=True)
        filepath = ydl.prepare_filename(info)

        if os.path.exists(filepath):
            print(f"Successfully downloaded to {filepath}")
            return filepath, info
        else:
            # Check if the extension changed (e.g., from .webm to .mp4)
            base, _ = os.path.splitext(filepath)
            for ext in [".mp4", ".webm", ".mkv"]:
                test_path = base + ext
                if os.path.exists(test_path):
                    print(f"Successfully downloaded to {test_path}")
                    return test_path, info

            print(f"Failed to find downloaded video file for {url}")
            return None, info


def extract_frame_from_position(video_path: str, position=0.5):
    """Extract a frame at the specified position (0.0 to 1.0) in the video."""
    cap = cv2.VideoCapture(video_path)

    # Get video properties
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = frame_count / fps

    # Calculate the frame number to extract
    frame_number = int(position * frame_count)

    # Set position and read frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        print(f"Failed to extract frame at position {position} (frame {frame_number})")
        return None, None

    # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    timestamp = frame_number / fps
    print(
        f"Extracted frame {frame_number} at {timestamp:.2f}s ({position * 100:.1f}% through video)"
    )
    print(f"Video info: {frame_count} frames, {fps:.2f} fps, {duration:.2f}s duration")

    return frame, frame_number


def get_n_frames(video_path: str):
    cap = cv2.VideoCapture(video_path)
    return int(cap.get(cv2.CAP_PROP_FRAME_COUNT))


def get_frame(video_path: str, frame_number: int):
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        print(f"Error: Failed to extract frame {frame_number} for video {video_path}")
        sys.exit(1)
    if frame.shape[0] != 1080 or frame.shape[1] != 1920 or frame.shape[2] != 3:
        print("Error: Frame isn't (1080, 1920, 3), instead", frame.shape)
        sys.exit(1)
    return frame


def gemini_cost(
    usage_metadata: Optional[types.GenerateContentResponseUsageMetadata],
    model_version: Optional[str],
) -> float:
    """
    Calculates the total cost of a Gemini API call based on usage metadata and model pricing.

    Args:
        usage_metadata: The dictionary containing token usage information.
        model_name: The name of the Gemini model used. Defaults to "gemini-2.0-flash-lite-001".

    Returns:
        The total cost of the API call in USD as a double.
    """
    # Pricing per 1 million tokens in USD
    # https://ai.google.dev/gemini-api/docs/pricing
    pricing = {
        "gemini-2.0-flash-lite-001": {
            "input": 0.075,
            "output": 0.30,
        },
        # NOTE: Other models may use thinking tokens, which are charged differently.
    }
    if usage_metadata is None or model_version is None:
        raise ValueError("Got None for one of usage, model:", usage_metadata, model_version)

    if model_version not in pricing:
        raise ValueError(f"Pricing for model '{model_version}' not found.")

    input_tokens = (
        usage_metadata.prompt_token_count if usage_metadata.prompt_token_count is not None else 0
    )
    output_tokens = (
        usage_metadata.candidates_token_count
        if usage_metadata.candidates_token_count is not None
        else 0
    )

    input_cost = (input_tokens / 1_000_000) * pricing[model_version]["input"]
    output_cost = (output_tokens / 1_000_000) * pricing[model_version]["output"]

    total_cost = input_cost + output_cost

    return total_cost


def ocr_gemini(crop: np.ndarray, prompt: str, model_class: Type[T]) -> T:
    """
    Generic OCR function using Gemini to extract structured data from images.

    Args:
        crop: The image crop as numpy array
        prompt: Custom prompt instructions for the model
        model_class: Pydantic model class to validate and parse the response

    Returns:
        An instance of the provided model_class with extracted data
    """
    _, buffer = cv2.imencode(".png", crop)
    image_bytes = buffer.tobytes()

    response = client.models.generate_content(
        model="gemini-2.0-flash-lite-001",
        contents=[
            types.Part.from_text(text=prompt),
            types.Part.from_bytes(data=image_bytes, mime_type="image/png"),
        ],
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=model_class,
        ),
    )

    global RUNNING_COST
    RUNNING_COST += gemini_cost(response.usage_metadata, response.model_version)

    # Parse the response using the provided model
    return model_class.model_validate_json(
        response.candidates[0].content.parts[0].text  # type: ignore
    )


class OcrNumberResult(BaseModel):
    number: Optional[int]


def ocr_gemini_int(crop: np.ndarray) -> Optional[int]:
    result = ocr_gemini(
        crop=crop,
        prompt="Extract the number from this image. Only if no number is visible, return None.",
        model_class=OcrNumberResult,
    )
    return result.number


class OcrPopulationResult(BaseModel):
    numerator: Optional[int]
    denominator: Optional[int]


def ocr_gemini_pop(crop: np.ndarray) -> tuple[Optional[int], Optional[int]]:
    result = ocr_gemini(
        crop=crop,
        prompt="Extract in-order the two numbers separated by a slash in this image, i.e., numerator/denominator. Only if no numbers are visible, return None for each.",
        model_class=OcrPopulationResult,
    )
    return result.numerator, result.denominator


class OcrAgeTextResult(BaseModel):
    age: Optional[AgeText]


def ocr_gemini_age_text(crop: np.ndarray) -> Optional[AgeText]:
    result = ocr_gemini(
        crop=crop,
        prompt="Extract the text from this image. It must exactly match one of the four options: Dark Age, Feudal Age, Castle Age, Imperial Age. Pick the closest match. Only if there is no text that matches any option, return None.",
        model_class=OcrAgeTextResult,
    )
    return result.age


class OcrAgeNumeralResult(BaseModel):
    roman_numeral: Optional[AgeNumeral]


def ocr_gemini_age_numeral(crop: np.ndarray) -> Optional[AgeNumeral]:
    result = ocr_gemini(
        crop=crop,
        prompt="Extract the text from this image. It will be a roman numeral on top of a crest background. It must exactly match one of the roman numerals from 1 through 4: I, II, III, IV. Pick the closest match. Only if there is no text, return None.",
        model_class=OcrAgeNumeralResult,
    )
    return result.roman_numeral


class OcrTextResult(BaseModel):
    text: Optional[str]


def ocr_gemini_text(crop: np.ndarray) -> Optional[str]:
    result = ocr_gemini(
        crop=crop,
        prompt="Extract the text from this image. Only if there is no text, return None.",
        model_class=OcrTextResult,
    )
    return result.text


class OcrTimeResult(BaseModel):
    hh: Optional[str]
    mm: Optional[str]
    ss: Optional[str]


def ocr_gemini_time(
    crop: np.ndarray,
) -> tuple[None, None, None] | tuple[int, int, int]:
    result = ocr_gemini(
        crop=crop,
        prompt="Extract the time displayed in this image. Is formatted hh:mm:ss. Only if there is no text, return None for each value.",
        model_class=OcrTimeResult,
    )
    return (
        (None, None, None)
        if result.hh is None or result.mm is None or result.ss is None
        else (int(result.hh), int(result.mm), int(result.ss))
    )


class OcrNumberPlayerScoreResult(BaseModel):
    number: Optional[int]
    player: Optional[str]
    numerator: Optional[int]
    denominator: Optional[int]


def ocr_gemini_number_player_score(
    crop: np.ndarray,
) -> tuple[Optional[int], Optional[str], Optional[int], Optional[int]]:
    result = ocr_gemini(
        crop=crop,
        prompt="Extract the number and then player name displayed in this image. From left to right, the narrow image will have a series of icons: first a clock, then a globe. Ignore those. Then, there will be a number inside a square between 1 and 8. Extract that number. Then, there will be a player's name, like 'GL.Hera' or 'TAG_MbL_'. Extract that player name. There, there will be a colon ':', ignore that. Finally, there will be two identical numbers formatted like numerator/denominator. Extract each of these identical numbers as numerator and denominator. Only if there is no text formatted like this, instead return None for each component.",
        model_class=OcrNumberPlayerScoreResult,
    )
    return result.number, result.player, result.numerator, result.denominator


class OcrPlayerCivResult(BaseModel):
    player: Optional[str]
    civilization: Optional[str]


def ocr_gemini_player_civ(crop: np.ndarray) -> tuple[Optional[str], Optional[str]]:
    result = ocr_gemini(
        crop=crop,
        prompt="Extract the player name and civilization name displayed in this image. If present, it will be formatted like 'Player (Civilization)'. For example, 'GL.Hera (Britons)' or 'TAG_MbL_ (Armenians)'. Only if no text is present in this format, instead return None for each component.",
        model_class=OcrPlayerCivResult,
    )
    return result.player, result.civilization


AGE_TEXT_RECT = Rectangle(name="age_text", x=625, y=14, w=180, h=28, ocr_fn=ocr_gemini_age_text)

RECTANGLES: list[Rectangle] = [
    Rectangle(name="wood", x=49, y=17, w=56, h=23, ocr_fn=ocr_gemini_int),
    Rectangle(name="wood_villagers", x=8, y=30, w=40, h=17, ocr_fn=ocr_gemini_int),
    Rectangle(name="food", x=148, y=17, w=56, h=23, ocr_fn=ocr_gemini_int),
    Rectangle(name="food_villagers", x=107, y=30, w=40, h=17, ocr_fn=ocr_gemini_int),
    Rectangle(name="gold", x=247, y=17, w=56, h=23, ocr_fn=ocr_gemini_int),
    Rectangle(name="gold_villagers", x=206, y=30, w=40, h=17, ocr_fn=ocr_gemini_int),
    Rectangle(name="stone", x=346, y=17, w=56, h=23, ocr_fn=ocr_gemini_int),
    Rectangle(name="stone_villagers", x=305, y=30, w=40, h=17, ocr_fn=ocr_gemini_int),
    Rectangle(name="population", x=446, y=15, w=75, h=27, ocr_fn=ocr_gemini_pop),
    Rectangle(name="villagers", x=403, y=29, w=41, h=18, ocr_fn=ocr_gemini_int),
    Rectangle(name="idles", x=526, y=15, w=33, h=27, ocr_fn=ocr_gemini_int),
    Rectangle(name="age_numeral", x=577, y=10, w=33, h=32, ocr_fn=ocr_gemini_age_numeral),
    AGE_TEXT_RECT,
    Rectangle(name="home", x=828, y=0, w=311, h=37, ocr_fn=ocr_gemini_text),  # Hera
    Rectangle(name="desc", x=848, y=39, w=291, h=24, ocr_fn=ocr_gemini_text),  # Ranked
    Rectangle(name="wins", x=1144, y=0, w=63, h=63, ocr_fn=ocr_gemini_int),
    Rectangle(name="losses", x=1209, y=0, w=63, h=63, ocr_fn=ocr_gemini_int),
    Rectangle(name="away", x=1277, y=0, w=306, h=37, ocr_fn=ocr_gemini_text),  # Opponents
    Rectangle(name="metadate", x=1277, y=39, w=284, h=24, ocr_fn=ocr_gemini_text),
    Rectangle(name="gametime", x=1596, y=54, w=72, h=22, ocr_fn=ocr_gemini_time),
    Rectangle(
        name="top_number_player_score",
        x=1544,
        y=836,
        w=319,
        h=27,
        ocr_fn=ocr_gemini_number_player_score,
    ),
    Rectangle(
        name="top_player_age_numeral", x=1885, y=839, w=26, h=24, ocr_fn=ocr_gemini_age_numeral
    ),
    Rectangle(
        name="bottom_number_player_score",
        x=1544,
        y=862,
        w=319,
        h=27,
        ocr_fn=ocr_gemini_number_player_score,
    ),
    Rectangle(
        name="bottom_player_age_numeral",
        x=1885,
        y=862,
        w=26,
        h=26,
        ocr_fn=ocr_gemini_age_numeral,
    ),
    Rectangle(name="selected_player_civ", x=599, y=918, w=250, h=26, ocr_fn=ocr_gemini_player_civ),
]


def get_frame_result(out_dir: str, video_path: str, frame_number: int):
    """Gets full results (comprehensive OCR)"""
    print("Getting FrameResult for frame", frame_number)
    output_file = os.path.join(out_dir, f"frame_{frame_number}_results.json")

    # return results if they already exist
    if os.path.exists(output_file):
        print("Skipping: Frame results exist at", output_file)
        return FrameResult.model_validate_json(read(output_file))

    # else, do OCR and cache result before returning
    results = analyze_frame(get_frame(video_path, frame_number))
    write(output_file, results.model_dump_json())
    return results


class AgeTextResult(BaseModel):
    """Somewhat redundant class just so we have a top-level pyandtic type for the cache format."""

    age: Optional[AgeText]


def get_frame_age_text(out_dir: str, video_path: str, frame_number: int):
    """Gets age text only from frame"""
    print("Getting AgeTextResult for frame", frame_number)
    output_file = os.path.join(out_dir, f"frame_{frame_number}_agetextresult.json")

    # return age if already found
    if os.path.exists(output_file):
        print("Skipping: AgeTextResult exists at", output_file)
        return AgeTextResult.model_validate_json(read(output_file))

    # else, do OCR on age only and cache result before returning
    atr = analyze_frame_age_text(get_frame(video_path, frame_number))
    write(output_file, atr.model_dump_json())
    return atr


def analyze_frame_age_text(
    frame: cv2.typing.MatLike,
    display_frame: bool = False,
    display_crops: bool = False,
    print_results: bool = True,
    print_cost: bool = False,
) -> AgeTextResult:
    """Extract text from defined regions in a frame and show results."""
    # Display the full frame first
    if display_frame:
        display(frame)

    rect = AGE_TEXT_RECT
    # Crop the image according to the rectangle
    crop = frame[rect.y : rect.y + rect.h, rect.x : rect.x + rect.w]
    if print_results:
        print("Performing OCR for", rect.name)
    if display_crops:
        display(crop)  # , height=(rect.h))
    detected = rect.ocr_fn(crop)
    if print_results:
        print("Detected:", detected)
        print()

    if print_cost:
        print(f"Total cost so far: ${RUNNING_COST:.8f}")

    return AgeTextResult(age=detected)


def analyze_frame(
    frame: cv2.typing.MatLike,
    display_frame: bool = True,
    display_crops: bool = False,
    print_results: bool = True,
    print_cost: bool = True,
) -> FrameResult:
    """Extract text from defined regions in a frame and show results."""
    # Display the full frame first
    if display_frame:
        display(frame)

    results: dict[str, Any] = {}

    for rect in RECTANGLES:
        # Crop the image according to the rectangle
        crop = frame[rect.y : rect.y + rect.h, rect.x : rect.x + rect.w]
        if print_results:
            print("Performing OCR for", rect.name)
        if display_crops:
            display(crop)  # , height=(rect.h))
        detected = rect.ocr_fn(crop)
        if print_results:
            print("Detected:", detected)
            print()

        results[rect.name] = detected

    if print_cost:
        print(f"Total cost so far: ${RUNNING_COST:.8f}")

    return FrameResult(
        resources=Resources(
            wood=results["wood"],
            food=results["food"],
            gold=results["gold"],
            stone=results["stone"],
        ),
        villagers=Villagers(
            wood=results["wood_villagers"],
            food=results["food_villagers"],
            gold=results["gold_villagers"],
            stone=results["stone_villagers"],
            total=results["villagers"],
            idle=results["idles"],
        ),
        population=Population(
            current=results["population"][0],
            maximum=results["population"][1],
        ),
        age=Age(
            numeral=results["age_numeral"],
            text=results["age_text"],
        ),
        metadata=FrameMetadata(
            home=results["home"],
            away=results["away"],
            desc=results["desc"],
            metadate=results["metadate"],
            wins=results["wins"],
            losses=results["losses"],
        ),
        gametime=Gametime(
            hours=results["gametime"][0],
            minutes=results["gametime"][1],
            seconds=results["gametime"][2],
        ),
        top_player=PlayerInfo(
            number=results["top_number_player_score"][0],
            name=results["top_number_player_score"][1],
            score_individual=results["top_number_player_score"][2],
            score_team=results["top_number_player_score"][3],
            age_numeral=results["top_player_age_numeral"],
        ),
        bottom_player=PlayerInfo(
            number=results["bottom_number_player_score"][0],
            name=results["bottom_number_player_score"][1],
            score_individual=results["bottom_number_player_score"][2],
            score_team=results["bottom_number_player_score"][3],
            age_numeral=results["bottom_player_age_numeral"],
        ),
        selected_info=SelectedInfo(
            player_name=results["selected_player_civ"][0],
            civilization=results["selected_player_civ"][1],
        ),
    )


def _binary_search_age_click(
    out_dir: str, video_path: str, age: AgeText, start_frame: int, end_frame: int
) -> tuple[FrameResult, int]:
    """Searches start_frame - end_frame inclusive as the viable range for the first frame with age
    `age`. Returns (FrameResult, frame number)."""
    print(f"Considering {start_frame} - {end_frame} ({(end_frame - start_frame) + 1} frames)")
    age_idx = ORDERED_AGE_TEXTS.index(age)

    if start_frame > end_frame:
        print(f"Error: binary search got start_frame={start_frame} > end_frame={end_frame}")
        sys.exit(1)
    elif start_frame == end_frame:
        fr = get_frame_result(out_dir, video_path, start_frame)
        assert fr.age.text == age
        return fr, start_frame
    else:
        mid_frame = (start_frame + end_frame) // 2  # round down
        atr = get_frame_age_text(out_dir, video_path, mid_frame)
        # TODO: heuristic should incorporate position in video! right now assuming None for text =
        # before video starts (i.e., before 'dark age')
        mid_age_idx = -1 if atr.age is None else ORDERED_AGE_TEXTS.index(atr.age)
        if mid_age_idx >= age_idx:
            return _binary_search_age_click(out_dir, video_path, age, start_frame, mid_frame)
        else:
            return _binary_search_age_click(out_dir, video_path, age, mid_frame + 1, end_frame)


def binary_search_age_click(out_dir: str, video_path: str, age: AgeText):
    n_frames = get_n_frames(video_path)
    return _binary_search_age_click(out_dir, video_path, age, 0, n_frames - 1)


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="AoE2 Frame OCR Development Tool")
    parser.add_argument("--video", required=True, help="YouTube video URL or local video path")
    # parser.add_argument(
    #     "--position", type=float, default=0.5, help="Position in video (0.0 to 1.0)"
    # )
    parser.add_argument("--output", default="./output", help="Output directory for results")
    args = parser.parse_args()

    # Check if input is a URL or local file
    video_path = args.video
    video_info = None
    if args.video.startswith(("http://", "https://", "www.")):
        # Download the video if it's a URL
        video_path, video_info = download_video(args.video)
        if video_path is None:
            print("Error: Failed to download video. Exiting.")
            sys.exit(1)

    # Locate out dir
    video_uid = ".".join(os.path.basename(video_path).split(".")[:1])
    out_dir = os.path.join(args.output, video_uid)
    os.makedirs(out_dir, exist_ok=True)

    # Save metadata if it was retrieved and we don't have it saved.
    metadata_path = os.path.join(out_dir, "metadata.json")
    if video_info is not None and not os.path.exists(metadata_path):
        write(metadata_path, json.dumps(video_info))

    aci_path = os.path.join(out_dir, "age_click_index.json")
    if os.path.exists(aci_path):
        print(f"Skipping {video_uid} as age clicks already found at", aci_path)
    else:
        ageclick2frame: dict[AgeText, int] = {}
        for age in ORDERED_AGE_TEXTS[1:]:
            _, frame = binary_search_age_click(out_dir, video_path, age)
            ageclick2frame[age] = frame

        aci = AgeClickIndex(ageclick2frame=ageclick2frame)
        write(aci_path, aci.model_dump_json())


if __name__ == "__main__":
    main()
