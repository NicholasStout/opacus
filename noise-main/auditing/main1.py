import sys
import multiprocessing as mp
from itertools import product

from init import init
from utils import exp_run, exp_run_0, exp_run_1

differ_each = False


if __name__ == '__main__':
    noise_type = sys.argv[1] if len(sys.argv) > 1 else "gaussian"
    dataset = sys.argv[2] if len(sys.argv) > 2 else "fmnist"
    model = sys.argv[3] if len(sys.argv) > 3 else "lr"
    run_if = True if len(sys.argv) > 4 else False
    _, data_dir, dataset, model, pois_cts, clip_norms, noise_params, _, _, bkd_start, bkd_trials = init(noise_type, dataset, model)
    
    all_exp, all_exp_0, all_exp_1 = [], [], []
    bkd_ifs = ["nobackdoor", "backdoor"]
    for pois_ct, clip_norm, noise_param, trial, bkd_if in product(pois_cts, clip_norms, noise_params, range(bkd_start, bkd_trials), bkd_ifs):
        cmd = "python audit_{}.py --dataset={} --model={} --n_pois={} --l2_norm_clip={} --exp_name={} --noise_type={} --noise_params={} --{} > /dev/null"
        exp_type = "cv" if "fmnist" in dataset or "p100" in dataset else "nlp"
        cur_exp_name_ = "{}_{}-nbkd-{}-{}-{}-{}" if "no" in bkd_if else "{}_{}-bkd-{}-{}-{}-{}"
        
        if not differ_each:
            if "no" in bkd_if and float(pois_ct) > 1:
                continue
        
        cur_exp_name = cur_exp_name_.format(dataset, model, noise_type, noise_param, pois_ct, trial)
        exp_name = cmd.format(exp_type, dataset, model, pois_ct, clip_norm, cur_exp_name, noise_type, noise_param, bkd_if)
        if "cv" in exp_type:
            all_exp.append(exp_name)
        elif "nlp" in exp_type:
            if "gaussian" in noise_type:
                all_exp_0.append(exp_name)
            elif "lmo" in noise_type:
                all_exp_1.append(exp_name)
    
    print(f"\nExperiment numbers (cv): {len(all_exp)}")
    print(f"Experiment0 numbers (nlp.gaussian): {len(all_exp_0)}")
    print(f"Experiment1 numbers (nlp.lmo): {len(all_exp_1)}")
    
    print(f"import args.{noise_type}.{dataset}_{model}")
    if run_if:
        psize = 16 if "cv" in exp_type else 1
        pool = mp.Pool(processes=psize)
        if "cv" in exp_type:
            print('exp_run')
            pool.map(exp_run, all_exp)
        elif "nlp" in exp_type:
            if "gaussian" in noise_type:
                print('exp_run_0')
                pool.map(exp_run_0, all_exp_0)
            elif "lmo" in noise_type:
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
            if "gaussian" in noise_type:
                print('exp_run_0')
            elif "lmo" in noise_type:
                print('exp_run_1')
    