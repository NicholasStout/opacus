from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import subprocess
import numpy as np
from scipy import stats


def exp_run(cmd):
    cmd = f"CUDA_VISIBLE_DEVICES= " + cmd
    print(f"{cmd}")
    subprocess.run(cmd, shell=True)


def exp_run_0(cmd):
    cmd = f"CUDA_VISIBLE_DEVICES=0 " + cmd
    print(f"{cmd}")
    subprocess.run(cmd, shell=True)


def exp_run_1(cmd):
    cmd = f"CUDA_VISIBLE_DEVICES=1 " + cmd
    print(f"{cmd}")
    subprocess.run(cmd, shell=True)


def get_cfg(m):
    # m: bkd_if poist_ct noise_type noise_val
    splt = m.split('-')
    return (splt[1], splt[2], splt[3], splt[4])


def parse_name(fname):
    splt = fname.split('-')
    # ['batch', 'bkd', 'lmo', '8', '2.0', '0', '50.npy']
    splt[-1] = splt[-1].split(".")[0]
    return tuple([splt[v] for v in [1, 2, 3, 4]])


def get_cfg_2(m):
    splt = m.split('-')
    splt[-1] = splt[-1].split(".")[0]
    return (splt[2], splt[3], splt[4])


def get_cfg_3(m):
    splt = m.split('-')
    splt[-1] = splt[-1].split(".")[0]
    return (splt[2], splt[3], ".")


def clopper_pearson(count, trials, conf):
    count, trials, conf = np.array(count), np.array(trials), np.array(conf)
    q = count / trials

    ci_low = stats.beta.ppf(conf / 2., count, trials - count + 1)
    ci_upp = stats.beta.isf(conf / 2., count + 1, trials - count)

    if np.ndim(ci_low) > 0:
        ci_low[q == 0] = 0
        ci_upp[q == 1] = 1
    else:
        ci_low = ci_low if (q != 0) else 0
        ci_upp = ci_upp if (q != 1) else 1
    return ci_low, ci_upp



def bkd_find_thresh(nobkd_li, bkd_li, use_dkw=False):
    # find the biggest ratio
    best_threshs = {}
    nobkd_arr = nobkd_li
    bkd_arr = bkd_li
    all_arr = np.concatenate((nobkd_arr, bkd_arr)).ravel()
    all_threshs = np.unique(all_arr)
    best_plain_thresh = -np.inf, all_threshs[0]
    best_corr_thresh = -np.inf, all_threshs[0]
    for thresh in all_threshs:
        nobkd_ct = (nobkd_arr >= thresh).sum()
        bkd_ct = (bkd_arr >= thresh).sum()
        bkd_p = bkd_ct/bkd_arr.shape[0]
        nobkd_p = nobkd_ct/nobkd_arr.shape[0]
        
        if use_dkw:
            nobkd_ub = nobkd_p + np.sqrt(np.log(2/.05)/nobkd_arr.shape[0])
            bkd_lb = bkd_p - np.sqrt(np.log(2/.05)/bkd_arr.shape[0])
        else:
            _, nobkd_ub = clopper_pearson(nobkd_ct, nobkd_arr.shape[0], .01)
            bkd_lb, _ = clopper_pearson(bkd_ct, bkd_arr.shape[0], .01)

        if bkd_ct in [bkd_arr.shape[0], 0] or nobkd_ct in [nobkd_arr.shape[0], 0]:
            plain_ratio = 1
        elif bkd_p + nobkd_p > 1:  # this makes ratio bigger
            plain_ratio = (1-nobkd_p)/(1-bkd_p)
        else:
            plain_ratio = bkd_p/nobkd_p

        if nobkd_ub + bkd_lb > 1:
            corr_ratio = (1-nobkd_ub)/(1-bkd_lb)
        else:
            corr_ratio = bkd_lb/nobkd_ub

        plain_eps = np.log(plain_ratio)
        corr_eps = np.log(corr_ratio)

        if best_plain_thresh[0] < plain_eps:
            best_plain_thresh = plain_eps, thresh
        if best_corr_thresh[0] < corr_eps:
            best_corr_thresh = corr_eps, thresh
    return best_corr_thresh[1]

def bkd_get_eps(n_repeat, cfg, nobkd_li, bkd_li, thresh, use_dkw=False):
    nobkd_arr = nobkd_li
    bkd_arr = bkd_li
    bkd_ct, nobkd_ct = (bkd_arr >= thresh).sum(), (nobkd_arr >= thresh).sum()
    bkd_p = bkd_ct/bkd_arr.shape[0]
    nobkd_p = nobkd_ct/nobkd_arr.shape[0]
       
    if use_dkw:
        nobkd_ub = nobkd_p + np.sqrt(np.log(2/.05)/nobkd_arr.shape[0])
        bkd_lb = bkd_p - np.sqrt(np.log(2/.05)/bkd_arr.shape[0])
    else:
        nobkd_lb, nobkd_ub = clopper_pearson(nobkd_ct, nobkd_arr.shape[0], .01)
        bkd_lb, bkd_ub = clopper_pearson(bkd_ct, bkd_arr.shape[0], .01)

    if bkd_ct in [bkd_arr.shape[0], 0] or nobkd_ct in [nobkd_arr.shape[0], 0]:
        plain_ratio = 1
    elif bkd_p + nobkd_p > 1:  # this makes ratio bigger
        plain_ratio = (1-nobkd_p)/(1-bkd_p)
    else:
        plain_ratio = bkd_p/nobkd_p

    if nobkd_ub + bkd_lb >= 1:
        corr_ratio = (1-nobkd_ub)/(1-bkd_lb)
    else:
        corr_ratio = bkd_lb/nobkd_ub

    plain_eps = np.log(np.max([plain_ratio, 1/plain_ratio]))/n_repeat
    corr_eps = np.log(np.max([corr_ratio, 1/corr_ratio]))/n_repeat

    if nobkd_ub + bkd_lb > 1:
        return (corr_eps, plain_eps, 1-nobkd_ub, 1-bkd_lb)
    else:
        return (corr_eps, plain_eps, bkd_lb, nobkd_ub)
