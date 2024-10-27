import copy
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


dataset = "qnli"
# dataset = "sst2"
dataset = "fmnist"
# dataset = "p100"

# model = "2f"
model = "lr"
# model = "r"
# model = "b"
dataset_model = f"{dataset}_{model}"


overall_results = {}
overall_results[f"{dataset}_{model}"] = {}
noise_eps = {"2.27": "0.3", "3.41": "0.7", "9.74": "2", "22.73": "3"}
for noise in ['lmo', 'gaussian']:
    df = np.load(f"final/results.{dataset_model}.{noise}.npy", allow_pickle=True).item()
    if noise == "gaussian":
        new_df = {}
        for df_ in df:
            new_df_ = list(copy.deepcopy(df_))
            new_df_[1] = noise_eps[new_df_[1]]
            new_df[tuple(new_df_)] = df[df_]
        df = copy.deepcopy(new_df)
    
    overall_results[f"{dataset}_{model}"][noise] = df
print(overall_results)


for item in overall_results[f"{dataset}_{model}"]['gaussian']:
    noise_type, eps, pois_ct = item
    
    try:
        lmo = overall_results[f"{dataset}_{model}"]['lmo'][item]
        lmo_p0, lmo_p1 = lmo[1], lmo[2]
        epsilon_converted_lmo = (1/(2*int(pois_ct))) * np.log(np.max([lmo_p0/lmo_p1, lmo_p1/lmo_p0]))
        gau = overall_results[f"{dataset}_{model}"]['gaussian'][item]
        gau_p0, gau_p1 = gau[1], gau[2]
        epsilon_converted_gau = (1/(2*int(pois_ct))) * np.log(np.max([gau_p0/gau_p1, gau_p1/gau_p0]))
        new_item = [f"{dataset}_{model}", f"k={pois_ct}", f"eps={eps}", f"converted_epsilon: [lmo] {epsilon_converted_lmo}, [gau] {epsilon_converted_gau}"]
        print(new_item)
        
    except:
        pass


# # record best epsilon for lmo and gaussian from above results and plot the final results using the followings.
# eps = [0.3, 0.7, 2, 3]
# lmo = []
# gau = []

# sns.set_theme(style="whitegrid")
# sns.set_context("paper", font_scale=1.5)
# plt.rcParams['font.weight'] = 'bold'
# plt.rcParams['axes.labelweight'] = 'bold'
# plt.rcParams['axes.titlesize'] = 16
# plt.rcParams['axes.labelsize'] = 14
# plt.rcParams['legend.fontsize'] = 12
# plt.rcParams['xtick.labelsize'] = 12
# plt.rcParams['ytick.labelsize'] = 12

# values = np.array([0, 1, 2, 3])
# eps_all = np.array([eps, lmo, gau]).T
# data = pd.DataFrame(eps_all, values, columns=['Theory', f'{dataset_model}(lmo)', f'{dataset_model}(Gaussian)'])
# sns_plot=sns.lineplot(data=data, palette="tab10", linewidth=2.5, markers=True, markersize=8)

# sns_plot.set_yscale("log")
# sns_plot.set_yticks([10**i for i in range(-2, 2)])
# sns_plot.set_xlabel(r"Provable $\epsilon$")
# sns_plot.set_ylabel(r"Estimate $\epsilon$")
# sns_plot.legend(loc='upper left')

# plt.savefig(f"final/audit.{dataset_model}.png")
# plt.savefig(f"final/audit.{dataset_model}.pdf")

