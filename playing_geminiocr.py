import argparse
import code
import os

from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import BaseModel

# Try imgcat if available
try:
    from imgcat import imgcat
except ImportError:
    imgcat = None
from PIL import Image


def load_client():
    load_dotenv()
    api_key = os.getenv("GOOGLE_GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GOOGLE_GEMINI_API_KEY not found in environment.")
    return genai.Client(api_key=api_key)


def list_models(client: genai.Client) -> None:
    for model in client.models.list():
        # Each `model` object has various attributes
        print(f"  Model Name: {model.name}")
        print(f"  Display Name: {model.display_name}")
        print(f"  Description: {model.description}")
        print(f"  Supported actions: {model.supported_actions}")
        print(f"  Input Token Limit: {model.input_token_limit}")
        print(f"  Output Token Limit: {model.output_token_limit}")
        print("-" * 20)


# model="gemini-2.5-flash-preview-04-17",
# model="gemini-1.5-pro-latest",
# model="gemini-2.0-flash-001",


class OcrResult(BaseModel):
    number: int


def ocr_image(client: genai.Client, image_path: str) -> int:
    with open(image_path, "rb") as f:
        image_bytes = f.read()

    response = client.models.generate_content(
        model="gemini-2.0-flash-lite-001",
        contents=[
            types.Part.from_text(text="Extract the number from this image."),
            types.Part.from_bytes(data=image_bytes, mime_type="image/png"),
        ],
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=OcrResult,
        ),
    )

    # code.interact(local=dict(globals(), **locals()))

    return OcrResult.model_validate_json(
        response.candidates[0].content.parts[0].text  # type: ignore
    ).number


def main():
    parser = argparse.ArgumentParser()
    # parser.add_argument("image_path", type=str, help="Path to the image file")
    args = parser.parse_args()

    # if imgcat:
    #     with open(args.image_path, "rb") as f:
    #         imgcat(f.read())
    # else:
    #     img = Image.open(args.image_path)
    #     img.show()

    client = load_client()
    # list_models(client)

    tests = [
        ("img/ocrtest1.png", 16),
        ("img/ocrtest2.png", 75),
        ("img/ocrtest3.png", 44),
        ("img/ocrtest4.png", 0),
    ]
    print("want", "\t", "got")
    for path, want in tests:
        # path = args.image_path
        got = ocr_image(client, path)
        print(want, "\t", got)


if __name__ == "__main__":
    main()
