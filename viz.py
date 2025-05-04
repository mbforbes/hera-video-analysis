import os
from typing import Optional

import matplotlib.pyplot as plt
from mbforbes_python_utils import read
import seaborn as sns

from main import AgeClickIndex, AgeText, FrameResultCache

# load data
output_dir = "./output"
sds = [d for d in os.listdir(output_dir) if os.path.isdir(os.path.join(output_dir, d))]
villager_counts: list[tuple[AgeText, Optional[int]]] = []
for sd in sds:
    subdir = os.path.join(output_dir, sd)
    aci_path = os.path.join(subdir, "age_click_index.json")
    frc_path = os.path.join(subdir, "frame_result_cache.json")
    if os.path.exists(aci_path) and os.path.exists(frc_path):
        aci = AgeClickIndex.model_validate_json(read(aci_path))
        frc = FrameResultCache.model_validate_json(read(frc_path))
        for age_text, frame_number in aci.ageclick2frame.items():
            if frame_number is None:
                continue
            frame_result = frc.frame2data[frame_number]
            villager_counts.append((age_text, frame_result.villagers.total))

# plot
sns.set_theme(style="darkgrid", rc={"figure.figsize": (10, 6)})
sns.histplot(
    x=[v[1] for v in villager_counts],
    hue=[v[0] for v in villager_counts],
    bins=100,
    binrange=(0, 101),
)
plt.title("Villagers when clicking to next age")
plt.xlabel("Villagers")
plt.ylabel("Games")
plt.show()
