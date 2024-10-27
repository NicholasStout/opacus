import os, sys
import numpy as np
from collections import defaultdict

from init import init
from utils import parse_name, get_cfg_2, get_cfg_3, bkd_find_thresh, bkd_get_eps

differ_each = False
release_poist_ct = ['1', '2', '4', '8']


if __name__ == '__main__':
    noise_type = sys.argv[1] if len(sys.argv) > 1 else "gaussian"
    dataset = sys.argv[2] if len(sys.argv) > 2 else "fmnist"
    model = sys.argv[3] if len(sys.argv) > 3 else "lr"
    search_ct = int(sys.argv[4]) if len(sys.argv) > 4 else 0
    run_if = True if len(sys.argv) > 4 else False
    
    exp_type, data_dir, dataset, model, pois_cts, clip_norms, _, _, save_dir, bkd_start, bkd_trials = init(noise_type, dataset, model)
    
    
    res_dir = os.path.join(save_dir, "results")
    all_nps = [f for f in os.listdir(res_dir) if f.endswith('.npy') and f.startswith('batch')]
    
    new_all_nps = []
    for item in all_nps:
        if parse_name(item) is not None:
            new_all_nps.append(item)
    print(new_all_nps)
    
    combined = defaultdict(list)
    for arr_f in new_all_nps:
        arr = np.load(os.path.join(res_dir, arr_f), allow_pickle=True)
        # print(arr_f, parse_name(arr_f))
        combined[parse_name(arr_f)].append(arr)

    all_files = []
    for name in combined:
        if np.concatenate(combined[name]).ravel().shape != (0,):
            print(name, np.concatenate(combined[name]).ravel().shape)
            filename = os.path.join(res_dir, '-'.join(['all'] + list(name)))
            np.save(filename, np.concatenate(combined[name]).ravel())
            print("{}.npy is saved!".format(os.path.join(res_dir, '-'.join(['all'] + list(name)))))
            all_files.append(filename)
        else:
            print(name, " ignored!")
    
    if len(all_files) == len(combined) and search_ct > 0:
        nos = [f for f in all_nps if "nbkd" in f and noise_type in f]
        yess = [f for f in all_nps if "bkd" in f and noise_type in f]
        print(len(nos), len(yess))
        # print(nos, yess)

        no_d, yes_d = {}, {}
        if not differ_each:
            for f in nos:
                no_d[get_cfg_3(f)] = np.load(os.path.join(res_dir, f))
            for f in yess:
                yes_d[get_cfg_2(f)] = np.load(os.path.join(res_dir, f))
        else:
            print("not implemented")
        
        print('no_d', no_d)
        print('yes_d', yes_d)
        
        valid_cfg = set(no_d) and set(yes_d)
        valid_cfg = sorted(valid_cfg, key = lambda tup: (tup[0], tup[1], tup[2]))
        
        vals = {}
        for cfg in valid_cfg[:1]:
            cfg_no = tuple((*list(cfg)[:-1], '.')) if not differ_each else cfg
            print("check", cfg_no)
            print("cfg", cfg)
            nobkd_li = no_d[cfg_no]
            print(nobkd_li)
            bkd_li = yes_d[cfg]
            print(bkd_li)
            
            nobkd_search = nobkd_li[:search_ct]
            bkd_search = bkd_li[:search_ct]
            nobkd_val = nobkd_li[search_ct:]
            bkd_val = bkd_li[search_ct:]
            
            assert len(nobkd_val) == len(bkd_val)
            n_repeat = len(nobkd_val)
            
            best_thresh = bkd_find_thresh(nobkd_search, bkd_search, use_dkw=True)
            bkd_lb = bkd_get_eps(n_repeat, cfg, nobkd_val, bkd_val, best_thresh, use_dkw=False)
            vals[cfg] = [bkd_lb[0], bkd_lb[2], bkd_lb[3]]
            print(vals[cfg])
            print("noise_type: {}, noise: {}, pois_ct: {}, clip norm: 1, epslb: {}, p0: {}, p1: {}".format(*cfg, bkd_lb[0], bkd_lb[2], bkd_lb[3]))
        
        np.save(res_dir, vals)
        
        os.makedirs("final", exist_ok=True)
        filename = "results."+dataset+"_"+model+"."+noise_type
        np.save(os.path.join("final", filename), vals)

    else:
        print('check the setting before next step; or check if you input search_ct>0...')
        print(len(all_files))
        print(len(combined))
        print(search_ct)
        
    
    