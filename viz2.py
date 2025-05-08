"""Mostly AI-generated, thanks AI"""

import code
import os
import glob
import json
import logging

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from main import ORDERED_AGE_NUMERALS, ORDERED_AGE_TEXTS
from main import AgeText

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_json_file(file_path: str) -> dict:
    """Load a JSON file and return its contents as a dictionary."""
    try:
        with open(file_path, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        return {}
    except json.JSONDecodeError:
        logger.error(f"Error decoding JSON from {file_path}")
        return {}


def process_video_directory(video_dir: str) -> pd.DataFrame:
    """Process a single video directory and return a dataframe of its frames."""
    video_id = os.path.basename(video_dir)
    logger.info(f"Processing video ID: {video_id}")

    # Load index files
    age_click_index = load_json_file(os.path.join(video_dir, "age_click_index.json"))
    age_start_index = load_json_file(os.path.join(video_dir, "age_start_index.json"))
    final_frame_index = load_json_file(
        os.path.join(video_dir, "final_frame_index.json")
    )

    # Load frame results cache
    frame_cache_path = os.path.join(video_dir, "frame_result_cache.json")
    frame_cache = load_json_file(frame_cache_path)

    if not frame_cache or "frame2data" not in frame_cache:
        logger.warning(f"No valid frame cache found for {video_id}")
        return pd.DataFrame()

    frame_data = frame_cache.get("frame2data", {})

    # Create lists to store data for dataframe
    rows = []

    # Process age click frames
    for age_text, frame_num in age_click_index.get("ageclick2frame", {}).items():
        if frame_num is None:
            continue

        if str(frame_num) not in frame_data:
            logger.error(
                f"Frame {frame_num} referenced in age_click_index but not found in frame cache for {video_id}"
            )
            continue

        frame_result = frame_data[str(frame_num)]
        row = {
            "video_id": video_id,
            "frame_num": frame_num,
            "type": "click",
            "age": age_text,
        }
        # Add all frame result data
        row.update(flatten_dict(frame_result, prefix=""))
        rows.append(row)

    # Process age start frames
    for age_numeral, frame_num in age_start_index.get("agestart2frame", {}).items():
        if frame_num is None:
            continue

        if str(frame_num) not in frame_data:
            logger.error(
                f"Frame {frame_num} referenced in age_start_index but not found in frame cache for {video_id}"
            )
            continue

        # Map age numeral to age text
        age_text = (
            ORDERED_AGE_TEXTS[ORDERED_AGE_NUMERALS.index(age_numeral)]
            if age_numeral in ORDERED_AGE_NUMERALS
            else "Unknown"
        )

        frame_result = frame_data[str(frame_num)]
        row = {
            "video_id": video_id,
            "frame_num": frame_num,
            "type": "start",
            "age": age_text,
        }
        # Add all frame result data
        row.update(flatten_dict(frame_result, prefix=""))
        rows.append(row)

    # Process final frame
    final_frame = final_frame_index.get("final_frame_number")
    if final_frame is not None:
        if str(final_frame) not in frame_data:
            logger.error(
                f"Final frame {final_frame} not found in frame cache for {video_id}"
            )
        else:
            frame_result = frame_data[str(final_frame)]
            row = {
                "video_id": video_id,
                "frame_num": final_frame,
                "type": "final",
                "age": "n/a",  # As specified in requirements
            }
            # Add all frame result data
            row.update(flatten_dict(frame_result, prefix=""))
            rows.append(row)

    return pd.DataFrame(rows)


def flatten_dict(d: dict, prefix: str = "") -> dict:
    """
    Flatten a nested dictionary into a single level dictionary.
    Keys are joined with underscores.
    """
    result = {}
    for k, v in d.items():
        key = f"{prefix}_{k}" if prefix else k

        if isinstance(v, dict):
            result.update(flatten_dict(v, key))
        else:
            result[key] = v

    return result


def build_combined_df():
    # Get all video directories
    output_dir = "output"
    video_dirs = glob.glob(os.path.join(output_dir, "*"))
    video_dirs = [d for d in video_dirs if os.path.isdir(d)]

    if not video_dirs:
        logger.warning(f"No video directories found in {output_dir}")
        return

    logger.info(f"Found {len(video_dirs)} video directories")

    # Process each directory and collect dataframes
    dfs = []
    for video_dir in video_dirs:
        df = process_video_directory(video_dir)
        if not df.empty:
            dfs.append(df)

    if not dfs:
        logger.warning("No valid data found in any video directory")
        return

    # Combine all dataframes
    combined_df = pd.concat(dfs, ignore_index=True)

    # Can save the combined dataframe to CSV for inspection
    # combined_df.to_csv("combined_frames_data.csv", index=False)
    # logger.info(f"Combined dataframe saved with {len(combined_df)} rows")

    return combined_df


def _add_x_bar_labels(ax):
    for bar in ax.patches:
        if bar.get_height() > 0:
            x_value = int(bar.get_x())
            x_position = bar.get_x() + bar.get_width() / 2
            height = bar.get_height()
            ax.text(
                x_position,  # x position (center of bar)
                height
                + (0.1 if x_value % 2 == 0 else 0.5),  # even/odd jitter for no overlap
                f"{x_value}",  # text (x-value)
                ha="center",  # horizontal alignment
                fontsize=8,  # smaller font size to avoid overcrowding
                # rotation=90,  # vertical text to save space
            )


def create_villagers_by_age_plot(df: pd.DataFrame):
    """
    Create a seaborn histogram showing the distribution of total villagers at each age click.
    Age is used as the hue parameter.
    """

    # Filter to only include click events and relevant ages
    ages = ORDERED_AGE_TEXTS
    # ages: list[AgeText] = ["Feudal Age"]
    plot_df = df[(df["type"] == "click") & (df["age"].isin(ages))]

    if plot_df.empty:
        logger.warning("No data available for villagers by age plot")
        return

    plt.figure(figsize=(12, 6))
    sns.set_theme(style="darkgrid")
    sns.histplot(
        data=plot_df,
        x="villagers_total",
        hue="age",
        multiple="stack",
        palette="Set2",
        # kde=True,
        # bins=150,
        binwidth=1,
    )
    _add_x_bar_labels(plt.gca())
    plt.title("Villagers when Clicking to Next Age")
    plt.xlabel("Total Villagers")
    plt.ylabel("Games")
    plt.tight_layout()
    # plt.show()
    plt.savefig("img/villagers-at-age-click.png")
    plt.close()

    logger.info("Created villagers by age histogram")


def create_resources_by_age_plots(df: pd.DataFrame):
    """
    Create a set of histograms showing the distribution of each resource (wood, food, gold, stone)
    when arriving at each age (start events). Age is used as the hue parameter.
    """
    import seaborn as sns
    import matplotlib.pyplot as plt

    # Filter to only include start events and relevant ages
    ages = ["Feudal Age"]
    # ages = ORDERED_AGE_TEXTS[1:]
    plot_df = df[(df["type"] == "start") & (df["age"].isin(ages))]

    if plot_df.empty:
        logger.warning("No data available for resources by age plots")
        return

    # Resources to plot
    resources = ["wood", "food", "gold", "stone"]

    # Create a figure with subplots for each resource
    fig, axes = plt.subplots(2, 2, figsize=(20, 12))
    axes = axes.flatten()

    # Create a histogram for each resource
    for i, resource in enumerate(resources):
        ax = axes[i]
        resource_col = f"resources_{resource}"

        # Skip if column doesn't exist
        if resource_col not in plot_df.columns:
            logger.warning(f"Column {resource_col} not found in dataframe")
            continue

        # Filter out rows with None/NaN values for this resource
        resource_df = plot_df[plot_df[resource_col].notna()]

        if resource_df.empty:
            logger.warning(f"No valid data for {resource}")
            continue

        sns.histplot(
            data=resource_df,
            x=resource_col,
            hue="age",
            multiple="stack",
            palette="Set2",
            binwidth=25,
            kde=True,
            ax=ax,
        )

        # _add_x_bar_labels(ax)

        ax.set_title(f"{resource.capitalize()} When Landing in Feudal Age")
        ax.set_xlabel(f"Total {resource.capitalize()}")
        ax.set_ylabel("Games")

    plt.tight_layout()
    # plt.show()
    plt.savefig("img/resources-at-feudal-land.png")
    plt.close()

    logger.info("Created resources by age histograms")


def main() -> None:
    combined_df = build_combined_df()

    if combined_df is None or combined_df.empty:
        logger.error("Combined DF invalid:", combined_df)
        return

    create_villagers_by_age_plot(combined_df)
    create_resources_by_age_plots(combined_df)


if __name__ == "__main__":
    main()
