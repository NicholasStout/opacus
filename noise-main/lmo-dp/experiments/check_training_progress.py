import os
import json
import matplotlib.pyplot as plt
from experiments.dataset.dataset_info import classification_steps

def check_json(jsonfile, plot_if=True):
    if not os.path.exists(jsonfile):
        print(f'this file does not exist -- {jsonfile}')
    
    with open(jsonfile, 'r') as json_file:
        json_=json.load(json_file)
    
    detail_Accuracy={}
    for idx, item in enumerate(json_[1:-1]):
        try:
            detail_Accuracy[json_[idx]['step']]=item['dev']['eval_acc']
        except:
            detail_Accuracy[json_[idx]['step']]=item['dev']['eval_mnli/acc']
    
    sorted_dict=dict(sorted(detail_Accuracy.items(),key=lambda item: item[1],reverse=True))
    if plot_if:
        plt.plot(list(range(len(list(detail_Accuracy.values())))),list(detail_Accuracy.values()))
    
    max_acc = list(sorted_dict.values())[0]
    needed_steps = list(sorted_dict.keys())[0]
    runned_steps = len(sorted_dict)
    return max_acc, needed_steps, runned_steps


if __name__ == "__main__":
    jsonfile = "../results/task1/jsonfiles/MNLI-matched@BERT-base@Gaussian_Noise@0.3@.json"
    # jsonfile = "../results/task1/jsonfiles/SST-2@BERT-base@Gaussian_Noise@0.3@.json"
    # jsonfile = "../results/task1/jsonfiles/QNLI@BERT-base@Gaussian_Noise@0.3@.json"
    # jsonfile = "../results/task1/jsonfiles/QQP@BERT-base@Gaussian_Noise@0.3@.json"
    
    dataset = "MNLI-matched"
    # dataset = "SST-2"
    # dataset = "QNLI"
    # dataset = "QQP"
    max_acc, needed_steps, runned_steps = check_json(jsonfile, plot_if=False)
    print(f"{max_acc} in step {needed_steps} and runned steps are {runned_steps}/{classification_steps[dataset]}")
    print("  ")
    