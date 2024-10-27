import sys
from pathlib import Path
from pprint import pformat
from itertools import product

try:
    exp_type = sys.argv[1]
except:
    exp_type = "cv"
    exp_type = "nlp"

noise_types = ["gaussian", "lmo"]
if exp_type == "cv":
    datasets, models = ["fmnist", "p100"], ["lr", "2f"]
elif exp_type == "nlp":
    datasets, models = ["qnli", "sst2"], ["r", "b"]

for dataset, model, noise_type in product(datasets, models, noise_types):
    args_path = f"{noise_type}/{dataset}_{model}.py"
    path = Path(noise_type)
    path.mkdir(parents=True, exist_ok=True)
    
    pois_ct = [1]
    clip_norm = [1]
    trials_start = 0
    trials_end = 1
    trials = (trials_start, trials_end)
    if noise_type == "gaussian":
        noise_params = [22.73, 9.74, 3.41, 2.27]
    elif noise_type == "lmo":
        noise_params = [0.3, 0.7, 2, 3]
    save_dir = f"../../save_results/auditing/{exp_type}/{dataset}_{model}/"
    
    args = {
        "pois_ct": pois_ct,
        "clip_norm": clip_norm,
        "trials": trials,
        "dataset": dataset,
        "model": model,
        "save_dir": save_dir,
        "data_dir": "datasets/",
        "noise_type": noise_type,
        "noise_params": noise_params,
    }
    
    formatted_string = "args = " + pformat(args, indent=4)
    formatted_string = formatted_string.replace("'", '"')
    with open(args_path, "w") as file:
        file.write(formatted_string)
