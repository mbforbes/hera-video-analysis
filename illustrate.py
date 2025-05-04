"""Illustrate process"""

import cv2

from main import ALL_RECTANGLES, Rectangle


def visualize_rectangles(image_path: str, rectangles: list[Rectangle], output_path: str):
    """
    Load an image, draw rectangles on it, and save the result.

    Args:
        image_path: Path to the input image
        rectangles: List of Rectangle objects
        output_path: Path to save the output image
    """
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not load image at {image_path}")

    # Define color and text properties
    rect_color = (255, 0, 255)  # pink
    text_color = (255, 0, 255)
    line_thickness = 2
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.75
    font_thickness = 2

    # Draw each rectangle and its name
    for rect in rectangles:
        # Draw the rectangle
        cv2.rectangle(
            image, (rect.x, rect.y), (rect.x + rect.w, rect.y + rect.h), rect_color, line_thickness
        )

        # Prepare text position (below the rectangle)
        text_x = rect.x
        text_y = rect.y + rect.h + 20  # 20 pixels below the rectangle

        # Draw the name
        cv2.putText(
            image, rect.name, (text_x, text_y), font, font_scale, text_color, font_thickness
        )

    # Save the result
    cv2.imwrite(output_path, image)
    print(f"Visualization saved to {output_path}")


if __name__ == "__main__":
    # Example usage
    image_path = "example_frame.png"
    output_path = "visualized_rectangles.png"

    # Call the function with ALL_RECTANGLES
    visualize_rectangles(image_path, ALL_RECTANGLES, output_path)
