import os

from mbforbes_python_utils import read

from main import AgeClickIndex

output_dir = "./output"
sds = [d for d in os.listdir(output_dir) if os.path.isdir(os.path.join(output_dir, d))]
for sd in sds:
    subdir = os.path.join(output_dir, sd)
    aci_path = os.path.join(subdir, "age_click_index.json")
    if os.path.exists(aci_path):
        aci = AgeClickIndex.model_validate_json(read(aci_path))
        aci.ageclick2frame  # TODO: curspot
