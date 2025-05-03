import argparse
import code
from dataclasses import dataclass
import json
import os
import sys
from typing import Callable, Literal, Optional, TypeVar, Type, Generic, Any

import cv2
from dotenv import load_dotenv
import easyocr
from google import genai
from google.genai import types
from imgcat import imgcat
import numpy as np
from pydantic import BaseModel
import pytesseract
import yt_dlp

# Create a generic type variable for the Pydantic model
T = TypeVar("T", bound=BaseModel)


# heavy, make once
# reader = easyocr.Reader(["en"])


# gemini client
load_dotenv()
api_key = os.getenv("GOOGLE_GEMINI_API_KEY")
if not api_key:
    raise RuntimeError("GOOGLE_GEMINI_API_KEY not found in environment.")
client = genai.Client(api_key=api_key)


@dataclass
class Rectangle:
    """Represents a rectangular region in an image."""

    name: str
    x: int
    y: int
    w: int
    h: int
    ocr_fn: Callable[[np.ndarray], Any]


@dataclass
class ProportionRectangle:
    """Represents a proportion (in 0 - 1) rectangular region in an image."""

    name: str
    xP: int
    yP: int
    wP: int
    hP: int
    ocr_fn: Callable[[np.ndarray], Any]


def display(frame_bgr: cv2.typing.MatLike) -> None:
    imgcat(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))


def download_video(url, output_dir="./videos"):
    """Download a YouTube video and return its local path."""
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


def extract_frame(video_path, position=0.5):
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


# def clean_age_text(image):
#     """Custom post-processor for age text in AoE2."""
#     # Convert to grayscale
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#
#     # Apply thresholding to isolate text
#     _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
#
#     # OCR with custom config for better accuracy with game fonts
#     text = str(pytesseract.image_to_string(thresh, config="--psm 7")).strip()
#
#     # Clean up common OCR errors in age names
#     if "castle" in text.lower():
#         return "Castle Age"
#     elif "imperial" in text.lower() or "imper" in text.lower():
#         return "Imperial Age"
#     elif "feudal" in text.lower() or "feud" in text.lower():
#         return "Feudal Age"
#     elif "dark" in text.lower():
#         return "Dark Age"
#     else:
#         return text


# def clean_number_text(image):
#     """Custom post-processor for numeric values in AoE2."""
#     # Convert to grayscale
#     # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

#     # Apply thresholding
#     _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)

#     # OCR with config optimized for digits
#     text = str(
#         pytesseract.image_to_string(
#             thresh, config="--psm 7 -c tessedit_char_whitelist=0123456789"
#         )
#     ).strip()

#     # Try to convert to integer, return original if not possible
#     try:
#         return str(int(text))
#     except ValueError:
#         return text


# def clean_number_text(image):
#     """OCR post-processing tuned to avoid blocky artifact hallucinations."""
#     # Convert to grayscale
#     gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
#
#     # Light upscale with smooth interpolation
#     # gray = cv2.resize(gray, None, fx=1., fy=1.5, interpolation=cv2.INTER_CUBIC)
#
#     # Very light blur
#     gray = cv2.GaussianBlur(gray, (3, 3), 0)
#
#     # Dynamic threshold (Otsu)
#     _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
#
#     # **Tiny blur after threshold to smooth jagged edges**
#     thresh = cv2.GaussianBlur(thresh, (1, 1), 0)
#
#     # Optional: Tiny morphology if needed (comment out if over-smoothing)
#     # kernel = np.ones((2, 2), np.uint8)
#     # thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
#
#     # Show processed image
#     display(thresh)
#
#     # OCR config
#     text = str(
#         pytesseract.image_to_string(
#             thresh, config="--psm 7 -c tessedit_char_whitelist=0123456789"
#         )
#     ).strip()
#
#     try:
#         return str(int(text))
#     except ValueError:
#         return text

# def ocr_easyocr(crop: np.ndarray):
#     # Apply OCR
#     # raw_text = str(pytesseract.image_to_string(crop)).strip()
#     gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
#     gray = cv2.resize(gray, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
#     _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
#     display(thresh)  # , height=(rect.h))
#     res = reader.readtext(thresh)
#     print("Res len:", len(res))
#     # code.interact(local=dict(globals(), **locals()))
#     _bb, raw_text, _conf = res[0]
#     print(f"OCR results: {raw_text}")

#     # Apply custom post-processing if specified
#     processed_text = raw_text
#     # if rect.post_process:
#     #     # For post-processing, also show the preprocessed image
#     #     processed_text = rect.post_process(crop)
#     #     print(f"Processed OCR: {processed_text}")

#     return processed_text


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
    numerator: int
    denominator: int


def ocr_gemini_pop(crop: np.ndarray) -> tuple[int, int]:
    result = ocr_gemini(
        crop=crop,
        prompt="Extract in-order the two numbers separated by a slash in this image, i.e., numerator/denominator.",
        model_class=OcrPopulationResult,
    )
    return result.numerator, result.denominator


class OcrAgeResult(BaseModel):
    age: Literal["Dark Age", "Feudal Age", "Castle Age", "Imperial Age"]


def ocr_gemini_age(crop: np.ndarray) -> str:
    result = ocr_gemini(
        crop=crop,
        prompt="Extract the text from this image. It must exactly match one of the four options: Dark Age, Feudal Age, Castle Age, Imperial Age. Pick the closest match.",
        model_class=OcrAgeResult,
    )
    return result.age


class OcrAgeNumeralResult(BaseModel):
    roman_numeral: Literal["I", "II", "III", "IV"]


def ocr_gemini_age_numeral(crop: np.ndarray) -> str:
    result = ocr_gemini(
        crop=crop,
        prompt="Extract the text from this image. It will be a roman numeral on top of a crest background. It must exactly match one of the roman numerals from 1 through 4: I, II, III, IV. Pick the closest match.",
        model_class=OcrAgeNumeralResult,
    )
    return result.roman_numeral


class OcrTextResult(BaseModel):
    text: str


def ocr_gemini_text(crop: np.ndarray) -> str:
    result = ocr_gemini(
        crop=crop,
        prompt="Extract the text from this image",
        model_class=OcrTextResult,
    )
    return result.text


class OcrTimeResult(BaseModel):
    hh: str
    mm: str
    ss: str


def ocr_gemini_time(crop: np.ndarray) -> tuple[int, int, int]:
    result = ocr_gemini(
        crop=crop,
        prompt="Extract the time displayed in this image. Is formatted hh:mm:ss",
        model_class=OcrTimeResult,
    )
    return int(result.hh), int(result.mm), int(result.ss)


class OcrNumberPlayerScoreResult(BaseModel):
    number: int
    player: str
    numerator: int
    denominator: int


def ocr_gemini_number_player_score(crop: np.ndarray) -> tuple[int, str, int, int]:
    result = ocr_gemini(
        crop=crop,
        prompt="Extract the number and then player name displayed in this image. From left to right, the narrow image will have a series of icons: first a clock, then a globe. Ignore those. Then, there will be a number inside a square between 1 and 8. Extract that number. Then, there will be a player's name, like 'GL.Hera' or 'TAG_MbL_'. Extract that player name. There, there will be a colon ':', ignore that. Finally, there will be two identical numbers formatted like numerator/denominator. Extract each of these identical numbers.",
        model_class=OcrNumberPlayerScoreResult,
    )
    return result.number, result.player, result.numerator, result.denominator


class OcrPlayerCivResult(BaseModel):
    player: Optional[str]
    civilization: Optional[str]


def ocr_gemini_player_civ(crop: np.ndarray) -> tuple[Optional[str], Optional[str]]:
    result = ocr_gemini(
        crop=crop,
        prompt="Extract the player name and civilization name displayed in this image. If present, it will be formatted like 'Player (Civilization)'. For example, 'GL.Hera (Britons)' or 'TAG_MbL_ (Armenians)'. If text is not present in this format, instead return None for each.",
        model_class=OcrPlayerCivResult,
    )
    return result.player, result.civilization


def analyze_frame(
    frame: cv2.typing.MatLike, rectangles: list[Rectangle | ProportionRectangle]
):
    """Extract text from defined regions in a frame and show results."""
    # Display the full frame first
    display(frame)  # , height=100)

    results = {}

    for r in rectangles:
        if isinstance(r, Rectangle):
            rect = r
        else:
            h, w, _ = frame.shape
            rect = Rectangle(
                x=r.xP * w,
                y=r.yP * h,
                w=r.wP * w,
                h=r.hP * w,
                name=r.name,
                ocr_fn=r.ocr_fn,
            )

        # Crop the image according to the rectangle
        crop = frame[rect.y : rect.y + rect.h, rect.x : rect.x + rect.w]

        # Display the original crop
        print("Performing OCR for", rect.name)
        display(crop)  # , height=(rect.h))

        # detected = ocr_easyocr(crop)
        detected = rect.ocr_fn(crop)
        print("Detected:", detected)

        results[rect.name] = detected
        print()

    return results


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="AoE2 Frame OCR Development Tool")
    parser.add_argument(
        "--video", required=True, help="YouTube video URL or local video path"
    )
    parser.add_argument(
        "--position", type=float, default=0.5, help="Position in video (0.0 to 1.0)"
    )
    parser.add_argument(
        "--output", default="./output", help="Output directory for results"
    )
    args = parser.parse_args()

    # Define regions of interest specific to AoE2
    rectangles: list[Rectangle | ProportionRectangle] = [
        Rectangle("wood", x=49, y=17, w=56, h=23, ocr_fn=ocr_gemini_int),
        Rectangle("wood_villagers", x=8, y=30, w=40, h=17, ocr_fn=ocr_gemini_int),
        Rectangle("food", x=148, y=17, w=56, h=23, ocr_fn=ocr_gemini_int),
        Rectangle("food_villagers", x=107, y=30, w=40, h=17, ocr_fn=ocr_gemini_int),
        Rectangle("gold", x=247, y=17, w=56, h=23, ocr_fn=ocr_gemini_int),
        Rectangle("gold_villagers", x=206, y=30, w=40, h=17, ocr_fn=ocr_gemini_int),
        Rectangle("stone", x=346, y=17, w=56, h=23, ocr_fn=ocr_gemini_int),
        Rectangle("stone_villagers", x=305, y=30, w=40, h=17, ocr_fn=ocr_gemini_int),
        Rectangle("population", x=446, y=15, w=75, h=27, ocr_fn=ocr_gemini_pop),
        Rectangle("villagers", x=403, y=29, w=41, h=18, ocr_fn=ocr_gemini_int),
        Rectangle("idles", x=526, y=15, w=33, h=27, ocr_fn=ocr_gemini_int),
        Rectangle("age", x=625, y=14, w=180, h=28, ocr_fn=ocr_gemini_age),
        Rectangle("home", x=828, y=0, w=311, h=37, ocr_fn=ocr_gemini_text),  # Hera
        Rectangle("desc", x=848, y=39, w=291, h=24, ocr_fn=ocr_gemini_text),  # Ranked
        Rectangle("wins", x=1144, y=0, w=63, h=63, ocr_fn=ocr_gemini_int),
        Rectangle("losses", x=1209, y=0, w=63, h=63, ocr_fn=ocr_gemini_int),
        Rectangle("away", x=1277, y=0, w=306, h=37, ocr_fn=ocr_gemini_text),  # Opp.
        Rectangle("metadate", x=1277, y=39, w=284, h=24, ocr_fn=ocr_gemini_text),
        Rectangle("gametime", x=1596, y=54, w=72, h=22, ocr_fn=ocr_gemini_time),
        Rectangle(
            "top_number_player_score",
            x=1544,
            y=836,
            w=319,
            h=27,
            ocr_fn=ocr_gemini_number_player_score,
        ),
        Rectangle(
            "top_player_age_numeral",
            x=1885,
            y=839,
            w=26,
            h=24,
            ocr_fn=ocr_gemini_age_numeral,
        ),
        Rectangle(
            "bottom_number_player_score",
            x=1544,
            y=862,
            w=319,
            h=27,
            ocr_fn=ocr_gemini_number_player_score,
        ),
        Rectangle(
            "bottom_player_age_numeral",
            x=1885,
            y=862,
            w=26,
            h=26,
            ocr_fn=ocr_gemini_age_numeral,
        ),
        Rectangle(
            "selected_player_civ",
            x=599,
            y=918,
            w=250,
            h=26,
            ocr_fn=ocr_gemini_player_civ,
        ),
    ]

    # Check if input is a URL or local file
    video_path = args.video
    if args.video.startswith(("http://", "https://", "www.")):
        # Download the video if it's a URL
        video_path, video_info = download_video(args.video)
        if not video_path:
            print("Failed to download video. Exiting.")
            sys.exit(1)

    # code.interact(local=dict(globals(), **locals()))

    # Extract a frame at the specified position
    frame, frame_number = extract_frame(video_path, args.position)
    if frame is None:
        print("Failed to extract frame. Exiting.")
        sys.exit(1)

    # check whether results exist
    video_uid = ".".join(os.path.basename(video_path).split(".")[:1])
    out_dir = os.path.join(args.output, video_uid)
    os.makedirs(out_dir, exist_ok=True)
    output_file = os.path.join(out_dir, f"frame_{frame_number}_results.json")
    if os.path.exists(output_file):
        print("Skipping: Frame results exist at", output_file)
    elif frame.shape[0] != 1080 or frame.shape[1] != 1920 or frame.shape[2] != 3:
        print("Error: Frame isn't (1080, 1920, 3), instead", frame.shape)
    else:
        # Analyze the frame and display results
        results = analyze_frame(frame, rectangles)

        # Save the results
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {output_file}")


if __name__ == "__main__":
    main()
