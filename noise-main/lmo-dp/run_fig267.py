import os
import json
import numpy as np
np.random.seed(1000)
import pandas as pd
import seaborn as sns
from pathlib import Path
import matplotlib.pyplot as plt
from experiments.check_privacy_and_statistics import generate_lmo_noise, compute_overall_privacy

if not os.path.exists("results/fig267/plot_fig267"):
    os.makedirs("results/fig267/plot_fig267")

## fig 27
L = 100
for chosen_lmo in ['task1', 'optimized_usefulness']:
    for epsilon in [0.3, 0.7, 2, 3]:
        jsonpath=f"experiments/lmo_noise_parameters/{chosen_lmo}/lmo_eps{epsilon}.json"
        lmo = json.loads(Path(jsonpath).read_text())
        
        lmo_noises = generate_lmo_noise(lmo, lmo['distributions'], noise_size=L)
        
        sigma = compute_overall_privacy(epsilon, compute_sigma_only=True)
        gaussian_noises = np.random.normal(0, sigma, L)
        
        print(f"The mean of abs(LMO noises) is {np.mean(np.abs(lmo_noises))} and the mean of abs(Gaussian noise) is {np.mean(np.abs(gaussian_noises))} when epsilon is {epsilon}.")
        
        
        fig=plt.figure()
        plt.scatter(x=[i for i in range(1, L+1)],marker='o',y=gaussian_noises,color='chocolate',label='Gaussian')
        plt.scatter(x=[i for i in range(1, L+1)],marker='x',y=lmo_noises,color='steelblue',label='LMO-DP (Ours)')

        if epsilon==0.3:
            plt.ylim(-60,60)
        elif epsilon==0.7:
            plt.ylim(-30,30)
        elif epsilon==2:
            plt.ylim(-10,10)
        elif epsilon==3:
            plt.ylim(-6,6)
        
        plt.xlabel("Sampled times", fontsize='x-large')
        plt.ylabel("Noise value", fontsize='x-large')
        plt.legend(loc='lower right',
                fancybox=True,
                edgecolor='black',
                fontsize='x-large',
                facecolor='white',
                framealpha=0.6)
        plt.grid()
            
        if chosen_lmo == "task1":
            plt.savefig(f"results/fig267/plot_fig267/figure7_lmo_eps{epsilon}.png")
            plt.savefig(f"results/fig267/plot_fig267/figure7_lmo_eps{epsilon}.pdf")
        elif chosen_lmo == "optimized_usefulness":
            plt.savefig(f"results/fig267/plot_fig267/figure2_lmo_eps{epsilon}.png")
            plt.savefig(f"results/fig267/plot_fig267/figure2_lmo_eps{epsilon}.pdf")


## fig 6
# Init
entropy={"Gaussian":{}, "LMO":{}}
variance={"Gaussian":{}, "LMO":{}}
variance_subplot={"Gaussian":{}, "LMO":{}}
df_entropy=pd.DataFrame([], columns=['epsilon','NoiseType','Entropy'])
df_variance=pd.DataFrame([], columns=['epsilon','NoiseType','Variance'])
df_variance_subplot=pd.DataFrame([], columns=['epsilon','NoiseType','Variance'])

x_axis=[0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 2, 3, 4]
x_axis_subplot=[0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
x_axis_star=[0.3, 0.7, 2, 3]

L=100000
for j in range(10):
    bin_numbers=100
    for idx, epsilon in enumerate(x_axis):
        ## "LMO"
        jsonpath = f"experiments/lmo_noise_parameters/fig6/lmo_eps{epsilon}.json"
        lmo = json.loads(Path(jsonpath).read_text())

        lmo_noises = generate_lmo_noise(lmo, lmo['distributions'], noise_size=L)
        # plt.hist(lmo_noises, bins=bin_numbers, density=True, alpha=0.5, label='Histogram')
        hist_lmo, bins_lmo = np.histogram(lmo_noises, bins=bin_numbers, density=True)
        bin_lmo_centers = (bins_lmo[:-1] + bins_lmo[1:]) / 2
        pdf_lmo = hist_lmo / np.sum(hist_lmo)
        # plt.plot(bin_lmo_centers, pdf_lmo, label='PDF')
        # plt.xlabel('Value')
        # plt.ylabel('Density')
        # plt.title('Probability Density Function (PDF)')
        # plt.legend()
        
        ## Entropy LMO
        entropy_lmo = 0
        for p in pdf_lmo:
            if p>0:
                logp=np.log2(p)
                entropy_lmo=entropy_lmo-p*logp
        # print(entropy_lmo)
        
        if j==0:
            entropy['LMO'][epsilon]=entropy_lmo
            ## Variance LMO
            variance['LMO'][epsilon]=np.var(lmo_noises)
            ## subfigure of Variance LMO
            if epsilon in x_axis_subplot:
                variance_subplot['LMO'][epsilon]=np.var(lmo_noises)
        
        df_entropy.loc[len(df_entropy.index)] = [epsilon, "LMO-DP (Ours)", entropy_lmo]
        df_variance.loc[len(df_variance.index)] = [epsilon, "LMO-DP (Ours)", np.var(lmo_noises)]
        
        if epsilon in x_axis_subplot:
            df_variance_subplot.loc[len(df_variance_subplot.index)] = [epsilon, "LMO-DP (Ours)", np.var(lmo_noises)]
        
        
        ## "Gaussian"
        sigma = compute_overall_privacy(epsilon, compute_sigma_only=True)
        gaussian_noises = np.random.normal(0, sigma, L)
        
        hist_gaussian, bins_gaussian = np.histogram(gaussian_noises, bins=bin_numbers, density=True)
        bin_gaussian_centers = (bins_gaussian[:-1] + bins_gaussian[1:]) / 2
        pdf_gaussian = hist_gaussian / np.sum(hist_gaussian)
        
        ## Entropy Gaussian
        entropy_gaussian=0
        for p in pdf_gaussian:
            if p>0:
                logp=np.log2(p)
                entropy_gaussian=entropy_gaussian-p*logp
        # print(entropy_gaussian)
        
        if j==0:
            entropy['Gaussian'][epsilon]=entropy_gaussian
            ## Variance Gaussian
            variance['Gaussian'][epsilon]=np.var(gaussian_noises)
        
        df_entropy.loc[len(df_entropy.index)] = [epsilon, "Gaussian", entropy_gaussian]
        df_variance.loc[len(df_variance.index)] = [epsilon, "Gaussian", np.var(gaussian_noises)]


## Plot Entropy
fig=plt.figure()
sns.lineplot(x="epsilon", y="Entropy",
            hue="NoiseType",
            data=df_entropy)
for k in x_axis_star:
    plt.plot([k,k],list(df_entropy.loc[df_entropy['epsilon']==k, 'Entropy'])[:2],'x',color="grey")
plt.xlabel(r"$\epsilon$", fontsize='x-large')
plt.ylabel("Entropy", fontsize='x-large')
plt.legend(loc='lower right',
           fancybox=True,
           edgecolor='black',
           fontsize='x-large',
           facecolor='white',
           framealpha=0.6)
plt.grid()
plt.savefig("results/fig267/plot_fig267/figure6_Entropy.png")
plt.savefig("results/fig267/plot_fig267/figure6_Entropy.pdf")

## Plot Variance
fig=plt.figure()
sns.lineplot(x="epsilon", y="Variance",
            hue="NoiseType",
            data=df_variance)
for k in x_axis_star:
    plt.plot([k,k],list(df_variance.loc[df_variance['epsilon']==k, 'Variance'])[:2],'x',color="grey")
plt.xlabel(r"$\epsilon$", fontsize='x-large')
plt.ylabel("Variance", fontsize='x-large')
plt.legend(loc='upper right',
           fancybox=True,
           edgecolor='black',
           fontsize='x-large',
           facecolor='white',
           framealpha=0.6)
plt.grid()

## Plot Variance subplot
left, bottom, width, height = 0.55, 0.25, 0.3, 0.3
ax=fig.add_axes([left, bottom, width, height])
plt.plot(x_axis_subplot,[variance_subplot['LMO'][i] for i in x_axis_subplot],'-',label='LMO-DP (Ours)',color=(255/255, 127/255, 15/255))
plt.plot(x_axis_star[:2],[variance['LMO'][i] for i in x_axis_star[:2]],'x',color="grey")
plt.grid()
plt.savefig("results/fig267/plot_fig267/figure6_Variance.png")
plt.savefig("results/fig267/plot_fig267/figure6_Variance.pdf")
