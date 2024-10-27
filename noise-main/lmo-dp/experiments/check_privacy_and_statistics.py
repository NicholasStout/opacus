import sys
import json
import numpy as np
from pathlib import Path

from experiments.accounting import rdp_accounting, accounting_manager
from experiments.lmo_noise_generator import generate_lmo_noise
from experiments.dataset.dataset_info import dataset_size, classification_steps, batchsize

max_order = 128
DEFAULT_ALPHAS = range(2, max_order + 1)  # list: [2, 3, ..., 128]
DEFAULT_DELTA = 1e-10
DEFAULT_DISTRIBUTIONS = ["Gamma", "Exponential","Uniform"]
DEFAULT_DELTA2 = 8e-6


def compute_overall_privacy(epsilon, delta=DEFAULT_DELTA, dataset="SST-2", sensitivity=1, compute_sigma_only=False):
    sigma = np.sqrt((2*np.log(1.25/delta)*float(sensitivity)**2)/float(epsilon)**2)
    if compute_sigma_only:
        return sigma
    
    # TODO explain why this sigma is smaller but reasonable.
    # manager = accounting_manager.RDPManager(alphas=DEFAULT_ALPHAS)
    # sigma_using_package = manager.compute_sigma(
    #         target_epsilon=epsilon, target_delta=delta, sample_rate=batchsize/dataset_size[dataset], epochs=classification_steps[dataset],
    #     )
    
    manager = accounting_manager.RDPManager(alphas=DEFAULT_ALPHAS)
    overall_epsilon = manager.compute_epsilon(
                            sigma=sigma,
                            sample_rate=batchsize[dataset]/dataset_size[dataset],
                            target_delta=delta,
                            steps=classification_steps[dataset],
    )
    return overall_epsilon, sigma


def compute_privacy_lmo_alpha(lmo, alpha):
    MGF1_1 = ((1-lmo['a1']*(alpha-1)*lmo['G_theta'])**(-lmo['G_k']))  # Gamma
    MGF1_3 = (lmo['E_lambda']/(lmo['E_lambda']-lmo['a3']*(alpha-1)))  # Exponential
    MGF1_4 = ((np.exp(lmo['a4']*(alpha-1)*lmo['U_b'])-np.exp(lmo['a4']*(alpha-1)*lmo['U_a']))/(lmo['a4']*(alpha-1)*(lmo['U_b']-lmo['U_a'])))  # Uniform
    MGF1 = MGF1_1 * MGF1_3 * MGF1_4
    
    MGF2_1 = ((1-lmo['a1']*(-alpha)*lmo['G_theta'])**(-lmo['G_k']))  # Gamma
    MGF2_3 = (lmo['E_lambda']/(lmo['E_lambda']-lmo['a3']*(-alpha)))  # Exponential
    MGF2_4 = ((np.exp(lmo['a4']*(-alpha)*lmo['U_b'])-np.exp(lmo['a4']*(-alpha)*lmo['U_a']))/(lmo['a4']*(-alpha)*(lmo['U_b']-lmo['U_a'])))  # Uniform
    MGF2 = MGF2_1 * MGF2_3 * MGF2_4
    
    rdp_lmo_ = (1/(alpha-1)) * np.log((alpha*MGF1+(alpha-1)*MGF2)/(2*alpha-1))
    return rdp_lmo_

def compute_privacy_lmo(lmo, steps=1):
    if lmo['a1']*(min(DEFAULT_ALPHAS)-1)<1/lmo['G_theta'] and lmo['a1']*(-max(DEFAULT_ALPHAS))<1/lmo['G_theta'] and lmo['a3']*(min(DEFAULT_ALPHAS)-1)<lmo['E_lambda'] and lmo['a3']*(-max(DEFAULT_ALPHAS))<lmo['E_lambda']:
        # print("Parameters meet the requirements.")
        pass
    else:
        # print("Parameters cannot meet the requirements.")
        return []

    rdp_lmo = np.zeros_like(DEFAULT_ALPHAS, dtype=float)
    for alpha in DEFAULT_ALPHAS:
        try:
            rdp_lmo_ = compute_privacy_lmo_alpha(lmo, alpha=alpha)
            
            if rdp_lmo_>0:
                rdp_lmo[int(alpha-2)] += rdp_lmo_
            else:
                rdp_lmo[int(alpha-2)] += np.inf
        except:
            rdp_lmo[int(alpha-2)] += np.inf
    
    try:
        overall_epsilon, opt_order = rdp_accounting.get_privacy_spent(DEFAULT_ALPHAS, rdp_lmo*steps, lmo['delta'])
    except:
        raise ArithmeticError
    
    return overall_epsilon, opt_order, rdp_lmo


def compute_usefulness_lmo(lmo, lmo_gamma=0.9):
    # usefulness = 1 - M1(-a1*gamma) * M3(-a3*gamma) * M4(-a4*gamma)
    MGF1 = ((1-(-lmo['a1']*lmo_gamma)*lmo['G_theta'])**(-lmo['G_k']))  # Gamma
    MGF3 = (lmo['E_lambda']/(lmo['E_lambda']-(-lmo['a3']*lmo_gamma)))  # Exponential
    MGF4 = ((np.exp((-lmo['a4']*lmo_gamma)*lmo['U_b'])-np.exp((-lmo['a4']*lmo_gamma)*lmo['U_a']))/((-lmo['a4']*lmo_gamma)*(lmo['U_b']-lmo['U_a'])))  # Uniform
    usefulness = 1 - MGF1 * MGF3 * MGF4
    
    return usefulness, lmo_gamma


if __name__ == '__main__':
    try:
        jsonpath=sys.argv[1]
        lmo = json.loads(Path(jsonpath).read_text())
        lmo['distributions'] = DEFAULT_DISTRIBUTIONS
        lmo['delta'] = DEFAULT_DELTA
    except:
        lmo={
            "a1": 0.1, "a3": 0.1, "a4": 0.1,
            "G_theta": 0.5, "G_k": 1, "E_lambda": 5, "U_b": 2, "U_a": 1,
            "distributions": DEFAULT_DISTRIBUTIONS,
            "delta": DEFAULT_DELTA,
        }
    
    overall_epsilon_lmo, opt_order, rdp_lmo = compute_privacy_lmo(lmo)
    print("  ")
    print("CHECK PRIVACY...")
    print(f"For LMO noise, the epsilon is {overall_epsilon_lmo} in (epsilon, delta)-DP when delta is {lmo['delta']}.")
    print(f"Now, the LMO noise parameters are {lmo}.")
    
    dataset="MNLI-matched"  # The options could be {"MNLI-matched", "SST-2", "QNLI", "QQP"}.
    overall_epsilon_gaussian, sigma = compute_overall_privacy(overall_epsilon_lmo, lmo['delta'], dataset=dataset)
    print("  ")
    print(f"Overall privacy computation for LMO noise and Gaussian noise in in {dataset} dataset:")
    print(f"The overall epsilon is {overall_epsilon_gaussian['eps_rdp']} in (epsilon, delta)-DP when delta is {lmo['delta']} and epoch is 6 (step is {classification_steps[dataset]}).")
    print(f"Now, the sigma for Gaussian noise is {sigma}.")
    
    # TODO check rdp of Gaussian and LMO noise: why this not equal but reasonable.
    # alpha = 20
    # rdp_gaussian = alpha/(2*sigma*sigma)  # q=1
    # rdp_lmo = compute_privacy_lmo_alpha(lmo, alpha=alpha)  # q=1
    
    print("  ")
    print("CHECK STATISTICS...")
    L = 1000
    lmo_noises = generate_lmo_noise(lmo, lmo['distributions'], noise_size=L)
    mean_lmo = np.mean(np.abs(lmo_noises))
    
    gaussian_noises = np.random.normal(0, sigma, L)
    mean_gaussian = np.mean(np.abs(gaussian_noises))
    print(f"The mean of abs(LMO noises) is {mean_lmo} and the mean of abs(Gaussian noise) is {mean_gaussian}.")
    
    print("  ")
    print("COMPUTING USEFULNESS...")
    usefulness, lmo_gamma = compute_usefulness_lmo(lmo)
    print(f"The Usefulness is {usefulness} when Gamma is equal to {lmo_gamma}.")

    print("  ")
    print("Change dataset, steps, epsilon or delta to find more privacy results...")

    print("  ")

    