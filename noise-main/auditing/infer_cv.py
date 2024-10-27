import os, sys
import numpy as np
np.random.seed(None)
from collections import defaultdict

from absl import app
from absl import flags

BATCH_SIZE = 50
differ_each = False  # TODO True is needed to be implemented


flags.DEFINE_string('dataset', 'fmnist', 'fmnist, p100, sst2, qnli.')
flags.DEFINE_string('model', '2f', '[fmnist, p100:] 2f, lr; [sst2, qnli:] r, b.')
flags.DEFINE_integer('n_pois', 8, '[number of clusters:] 1, 2, 4, 8.')
flags.DEFINE_float('l2_norm_clip', 1.0, '[Clipping norm] 1')
flags.DEFINE_string('exp_name', None, '[name of experiment] dataset, model, n/bkd, clip_norm, noise_type, noise_param, trial')
flags.DEFINE_string('noise_type', 'gaussian', '[type of noise] gaussian, lmo')
flags.DEFINE_float('noise_params', 1.1, '[For gaussian: ratio of the standard deviation to the clipping norm; For lmo: lmo params index]')
flags.DEFINE_boolean('backdoor', False, '[whether to backdoor] False, True.')
flags.DEFINE_float('learning_rate', 0.15, 'Learning rate for training')
flags.DEFINE_integer('microbatches', 1, 'Number of microbatches (must evenly divide batch_size)')
flags.DEFINE_integer('epochs', 24, 'Number of epochs')
flags.DEFINE_integer('batch_size', 250, 'Batch size')
FLAGS = flags.FLAGS



import tensorflow as tf
from distutils.version import LooseVersion
if LooseVersion(tf.__version__) < LooseVersion('2.0.0'):
    GradientDescentOptimizer = tf.train.GradientDescentOptimizer
    AdamOptimizer = tf.train.AdamOptimizer
else:
    GradientDescentOptimizer = tf.optimizers.SGD
    AdamOptimizer = tf.optimizers.Adam

from init import init
from utils import get_cfg
from auditor import auditor


def main(_):
    # python infer_cv.py 0 50 bkd gaussian 22.73 2 fmnist lr
    start = sys.argv[1]
    end = sys.argv[2]
    bkd_if = sys.argv[3]
    noise_type = sys.argv[4]
    noise_param = str(sys.argv[5])
    pois_ct = sys.argv[6]
    dataset = sys.argv[7]
    model = sys.argv[8]
    
    if not differ_each:
        if bkd_if=="nbkd" and int(pois_ct) > 1:
            exit(0)
    
    exp_type, data_dir, dataset, model, _, _, _, noise_type, save_dir, _, _ = init(noise_type, dataset, model)
    res_dir = os.path.join(save_dir, "results")
    os.makedirs(res_dir, exist_ok=True)
    
    cfg_map = defaultdict(list)
    h5s = [fname for fname in os.listdir(save_dir) if fname.endswith('.h5')]
    for h5 in h5s:
        cfg_map[get_cfg(h5)].append(h5)
    for val in cfg_map:
        cfg_map[val] = sorted(cfg_map[val], key=lambda h: int(h.split('-')[-1].split(".")[0]))
        print(val, cfg_map[val]) # ('bkd', 'lmo', '3', '8') ['fmnist_lr-bkd-lmo-3-8-0.h5', 'fmnist_lr-bkd-lmo-3-8-1.h5']
    
    saved_name = '-'.join([bkd_if, noise_type, noise_param, pois_ct, start, end])
    
    cfg_key = (bkd_if, noise_type, noise_param, pois_ct)
    print(saved_name, len(cfg_map[cfg_key]))
    
    
    assert exp_type == "cv"
    if "fmnist" in dataset:
        all_bkds = {"p": np.load(os.path.join(data_dir, "fmnist/clipbkd-new-1.npy"), allow_pickle=True)[2]}
        all_bkds["p"] = all_bkds["p"][0].reshape((-1, 28, 28, 1)), np.eye(2)[all_bkds["p"][1]][None, :]
    elif "p100" in dataset:
        all_bkds = {"p": np.load(data_dir + "/p100/p100_1.npy", allow_pickle=True)[2]}
        all_bkds["p"] = all_bkds["p"][0].reshape((-1, 100)), all_bkds["p"][1]
    
    
    alls = []
    auditing = auditor()
    for h5 in cfg_map[cfg_key][int(start):int(end)]:
        x, y = all_bkds['p']
        nob_vals = auditing.backdoor_cv(h5, x,  y, save_dir)
        alls.append(nob_vals)
    
    if alls == []:
        print(f'check this setting {cfg_key}')
    else:
        np.save(os.path.join(res_dir, '-'.join(["batch", saved_name])), np.array(alls))
        print(f"the results are saved in {res_dir}!")
    
    
if __name__ == '__main__':
    app.run(main)
    