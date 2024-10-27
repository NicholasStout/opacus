import importlib
from pathlib import Path
CACHE_DIR = "/scr/home/qiy22005/PLAYGROUND/auditing/auditing-dpsgd/cache_dir"



def init_1(noise="gaussian", dataset="fmnist", model="lr"):
    try:
        auditing_args = importlib.import_module(f'args.{noise}.{dataset}_{model}')
        print(f"import args.{noise}.{dataset}_{model}")
        data_dir    = auditing_args.args.get('data_dir', "datasets/")
        dataset     = auditing_args.args.get('dataset', "fmnist")
        model       = auditing_args.args.get('model', "lr")
        pois_cts    = auditing_args.args.get('pois_ct', [1,2,4,8])
        clip_norms  = auditing_args.args.get('clip_norm', [1])
        noise_params= auditing_args.args.get('noise_params', [22.73, 9.74, 3.41, 2.27])
        noise_type  = auditing_args.args.get('noise_type', 'gaussian')
        save_dir    = auditing_args.args.get('save_dir', "debug")
        bkd_start,_ = auditing_args.args.get('trials', [0, 1])
        _,bkd_trials= auditing_args.args.get('trials', [0, 1])
        
        exp_type = "cv" if dataset.startswith("p100") or dataset.startswith("fmnist") else "nlp"
        
        path = Path(save_dir)
        path.mkdir(parents=True, exist_ok=True)
    
    except ModuleNotFoundError:
        print('Setting cannot be found ...')
        print(f"please check if the spell is wrong: args.{noise}.{dataset}_{model}")
        exit(1)
    
    return exp_type, data_dir, dataset, model, pois_cts, clip_norms, noise_params, noise_type, save_dir, bkd_start, bkd_trials


def init_2(dataset="fmnist", model="lr"):
    exp_type_l, data_dir_l, dataset_l, model_l, pois_cts_l, clip_norms_l, noise_params_l, noise_type_l, save_dir_l, bkd_start_l, bkd_trials_l = init_1("lmo", dataset, model)
    exp_type_g, data_dir_g, dataset_g, model_g, pois_cts_g, clip_norms_g, noise_params_g, noise_type_g, save_dir_g, bkd_start_g, bkd_trials_g = init_1("gaussian", dataset, model)
    
    assert save_dir_l == save_dir_g
    return exp_type_g, data_dir_g, dataset_g, model_g, pois_cts_g, clip_norms_g, noise_params_g, noise_type_g, save_dir_g, bkd_start_g, bkd_trials_g



def init(noise="gaussian", dataset="fmnist", model="lr"):
    if noise == ["gaussian", "lmo"]:
        return init_2(dataset, model)
    else:
        return init_1(noise, dataset, model)
