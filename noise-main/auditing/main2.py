import os, sys
import multiprocessing as mp
from collections import defaultdict

from init import init
from utils import exp_run, exp_run_0, exp_run_1, get_cfg

BATCH_SIZE = 50


if __name__ == '__main__':
    dataset = sys.argv[1] if len(sys.argv) > 1 else "fmnist"
    model = sys.argv[2] if len(sys.argv) > 2 else "lr"
    run_if = True if len(sys.argv) > 3 else False
    
    exp_type, data_dir, dataset, model, pois_cts, clip_norms, _, _, save_dir, bkd_start, bkd_trials = init(['gaussian', 'lmo'], dataset, model)
    
    suffix=".h5" if exp_type=="cv" else ".safetensors"
    ms = [fname.split(suffix)[0] for fname in os.listdir(save_dir) if fname.endswith(suffix)]
    
    cfg_map = defaultdict(list)
    for m in ms:
        cfg_map[get_cfg(m)].append(m)
    args = {d: len(cfg_map[d]) for d in cfg_map if len(cfg_map[d]) > 0}
    
    print(f"there are {len(ms)} models for {dataset}_{model}.")
    print(f"there are {len(cfg_map)} cfgs for {dataset}_{model} with both gaussian and lmo.")
    print(f"there are {int(len(cfg_map)/2)} cfgs for {dataset}_{model} for each noise.")
    
    
    all_exp, all_exp_0, all_exp_1 = [], [], []
    fmt_cmd = "python infer_{}.py {} {} {} {} {} {} {} {}"
    for arg in args:
        for start in range(0, args[arg], BATCH_SIZE):
            cmd = fmt_cmd.format(exp_type, start, start + BATCH_SIZE, arg[0], arg[1], arg[2], arg[3], dataset, model)
            print(cmd)
            
            if "cv" in exp_type:
                all_exp.append(cmd)
            elif "nlp" in exp_type:
                if "gaussian" in arg[2]:
                    all_exp_0.append(cmd)
                elif "lmo" in arg[2]:
                    all_exp_1.append(cmd)
    
    print(all_exp_0)
    print(f"\nExperiment numbers (cv): {len(all_exp)}")
    print(f"Experiment0 numbers (nlp.gaussian): {len(all_exp_0)}")
    print(f"Experiment1 numbers (nlp.lmo): {len(all_exp_1)}")
    
    print(f"import {dataset}_{model}")
    if run_if:
        psize = 16 if "cv" in exp_type else 1
        pool = mp.Pool(processes=psize)
        if "cv" in exp_type:
            print('exp_run')
            pool.map(exp_run, all_exp)
        elif "nlp" in exp_type:
            print('exp_run_0')
            pool.map(exp_run_0, all_exp_0)
            print('exp_run_1')
            pool.map(exp_run_1, all_exp_1)
        pool.close()
        pool.join()
    else:
        psize = 16 if "cv" in exp_type else 1
        print(f"psize={psize}")
        if "cv" in exp_type:
            print('exp_run')
        elif "nlp" in exp_type:
            print('exp_run_0')
            print('exp_run_1')
