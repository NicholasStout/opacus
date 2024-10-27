import os
import argparse
import json
import shutil
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from experiments.check_privacy_and_statistics import compute_overall_privacy, DEFAULT_DELTA


dict_plot={
    "Gaussian": "Gaussian_Noise",
    "LMO": "LMO-DP_Noise_\(Ours\)",

    "sst-2": "SST-2",
    "mnli": "MNLI-matched",
    "qnli": "QNLI",
    "qqp": "QQP",

    'roberta-base': 'RoBERTa-base',
    'roberta-large': 'RoBERTa-large',
    'bert-base-cased': 'BERT-base',
    'bert-large-cased': 'BERT-large',
}


## task1 non-private accuracy
non_privacy_dev = {
    'RoBERTa-base': {
        'MNLI-matched': 87.6,
        'QNLI': 92.8,
        'QQP': 91.9,
        'SST-2': 94.8,
    },
    'RoBERTa-large': {
        'MNLI-matched': 90.2,
        'QNLI': 94.7,
        'QQP': 92.2,
        'SST-2': 96.4,
    },
    'BERT-base': { # 12-layer, 768-hidden, 12-heads, 110M parameters
        'MNLI-matched': 84.6, # MNLI-m
        'QNLI': 90.5,
        'QQP': None, # f1: 71.2; acc: https://huggingface.co/JeremiahZ/bert-base-uncased-qqp
        'SST-2': 93.5,
    },
    'BERT-large': { # 24-layer, 1024-hidden, 16-heads, 340M parameters
        'MNLI-matched': 86.7, # MNLI-m
        'QNLI': 92.7,
        'QQP': None, # f1: 72.1
        'SST-2': 94.9,
    }
}

def generate_task1_commands(num_GPU, task1_settings, out_folder="fig34589", lmo_folder="task1"):
    cmd_list=[]
    for noise_type in task1_settings['noise_type']:
        for task_name in task1_settings['task_name']:
            for model_name in task1_settings['model_name']:
                for epsilon in task1_settings['epsilon']:
                    sigma = compute_overall_privacy(float(epsilon), DEFAULT_DELTA, dataset="E2E", compute_sigma_only=True)
                    if noise_type=="LMO":
                        cmd=f"python3 -m classification.run_wrapper_lmo --task_name {task_name} --output_dir ../../results/{out_folder}/{dict_plot[noise_type]}/{dict_plot[task_name]}/{dict_plot[model_name]}/eps_{epsilon}/ --model_name_or_path {model_name} --eval_steps 1 --batch_size 2048 --learning_rate 0.0005 --few_shot_type finetune --num_train_epochs {task1_settings['epoch']} --clipping_mode ghost --per_example_max_grad_norm 1 --store_grads no --lmo_filepath ../lmo_noise_parameters/{lmo_folder}/lmo_eps{epsilon}.json 2>&1 > ../../running_logs/task1.{noise_type}.{task_name}.{model_name}.eps_{epsilon}.log"
                    elif noise_type=="Gaussian":
                        cmd=f"python3 -m classification.run_wrapper_mysetting --task_name {task_name} --output_dir ../../results/{out_folder}/{dict_plot[noise_type]}/{dict_plot[task_name]}/{dict_plot[model_name]}/eps_{epsilon}/ --model_name_or_path {model_name} --eval_steps 1 --batch_size 2048 --learning_rate 0.0005 --few_shot_type finetune --num_train_epochs {task1_settings['epoch']} --clipping_mode ghost --per_example_max_grad_norm 1 --store_grads no --noise_multiplier {sigma} 2>&1 > ../../running_logs/task1.{noise_type}.{task_name}.{model_name}.eps_{epsilon}.log"
                    cmd_list.append(cmd)
    
    GPU_commands = []
    for _ in range(num_GPU):
        GPU_commands.append([])
    for i in range(len(cmd_list)):
        GPU_idx = i % num_GPU
        GPU_commands[GPU_idx].append(cmd_list[i])

    for GPU_idx in range(num_GPU):
        serial_cmd = " && ".join(GPU_commands[GPU_idx])
        serial_cmd = f"CUDA_VISIBLE_DEVICES={GPU_idx} {serial_cmd} &"
        print(serial_cmd)


def copy_files_task1(sourcepath, destination, copy_files=False):
    for noise_type in ['Gaussian_Noise','LMO-DP_Noise_(Ours)']:
        for task_name in ['MNLI-matched','SST-2','QNLI','QQP']:
            for model_name in ['BERT-base','BERT-large','RoBERTa-base','RoBERTa-large']:
                for epsilon in [0.3,0.7,2,3]:
                    source_path=os.path.join(sourcepath,f"{noise_type}",f"{task_name}",f"{model_name}",f"eps_{epsilon}","log_history.json")
                    destination_path=os.path.join(destination,f"{task_name}@{model_name}@{noise_type}@{epsilon}@.json")
                    
                    if not os.path.exists(destination):
                        os.makedirs(destination)
                    
                    if os.path.exists(source_path) and copy_files:
                        shutil.copy(source_path, destination_path)
                    else:
                        print(source_path)


def plot_task1_figures_func(plotting_path):
    if not os.path.exists(plotting_path):
        os.makedirs(plotting_path)
    
    for model_family in ['base', 'large']:
        models={
            "base": ['BERT-base',"RoBERTa-base"],
            "large": ['BERT-large',"RoBERTa-large"],
        }

        steps_in_each_epoch = {
            "MNLI-matched": 191,
            "SST-2": 32,
            "QNLI": 306,
            "QQP": 177,
        }
        
        # plotting related settings
        linewidths=[3, 5]
        fontsize_labels=60
        markersize=20
        markeredgewidth=5
        lengend_fontsize=20

        step_gap = {
            "MNLI-matched": 30,
            "SST-2": 5,
            "QNLI": 50,
            "QQP": 30,
        }
        
        # init        
        detail_Accuracy={}
        for noise_type in ['Gaussian_Noise','LMO-DP_Noise_(Ours)']:
            detail_Accuracy[noise_type]={}
            for task_name in ['MNLI-matched','SST-2','QNLI','QQP']:
                detail_Accuracy[noise_type][task_name]={}
                for model_name in ['BERT-base','BERT-large','RoBERTa-base','RoBERTa-large']:
                    detail_Accuracy[noise_type][task_name][model_name]={}
                    for epsilon in [0.3,0.7,2,3]:
                        detail_Accuracy[noise_type][task_name][model_name][epsilon]={}

        # record contents from jsonfiles
        overall_epsilons={}
        for noise_type in ['Gaussian_Noise','LMO-DP_Noise_(Ours)']:
            for task_name in ['MNLI-matched','SST-2','QNLI','QQP']:
                overall_epsilons[task_name] = {}
                for model_name in models[model_family]:
                    for epsilon in [0.3,0.7,2,3]:
                        # compute privacy
                        overall_epsilon, _ = compute_overall_privacy(epsilon, DEFAULT_DELTA, dataset=task_name)
                        overall_epsilons[task_name][epsilon] = overall_epsilon['eps_rdp']
                
                        # record json
                        destination_path=os.path.join(destination,f"{task_name}@{model_name}@{noise_type}@{epsilon}@.json")

                        if not os.path.exists(destination_path):
                            print(f'this file does not exist -- {destination_path}')
                            continue
                        
                        with open(destination_path, 'r') as json_file:
                            json_=json.load(json_file)
                            
                            last_step=0
                            for idx, item in enumerate(json_):
                                if type(item['get_training_stats']['noise']) is type(None):
                                    continue
                                
                                if json_[idx]['step']==last_step+1 and json_[idx]['step'] < int(steps_in_each_epoch[task_name] * task1_settings['epoch']):
                                    try:
                                        detail_Accuracy[noise_type][task_name][model_name][epsilon][json_[idx]['step']]=item['dev']['eval_acc'] * 100
                                    except:
                                        detail_Accuracy[noise_type][task_name][model_name][epsilon][json_[idx]['step']]=item['dev']['eval_mnli/acc'] * 100
                                        
                                    last_step=last_step+1
                                else:
                                    continue
        
        
        for task_name in ['MNLI-matched','SST-2','QNLI','QQP']:
            # plot Figure 3 & Figure 4 {x-epsilon; y-accuracy}
            fig=plt.figure(figsize=(13,10))
            sns.set_theme(style="darkgrid")
            plt.style.use('ggplot')
            
            # plot non-private accuracy
            for model_name in models[model_family]:
                if non_privacy_dev[model_name][task_name] is None:
                    continue
                
                x_axis=list(overall_epsilons[task_name].values())
                if model_name == models[model_family][0]:
                    plt.plot(x_axis, [non_privacy_dev[model_name][task_name] for _ in range(len(x_axis))], '--', linewidth=linewidths[0], color="grey", label=f"{model_name} (non-private)")
                elif model_name == models[model_family][1]:
                    plt.plot(x_axis, [non_privacy_dev[model_name][task_name] for _ in range(len(x_axis))], '--', linewidth=linewidths[1], color="grey", label=f"{model_name} (non-private)")
            
            # plot runned results
            for noise_type in ['Gaussian_Noise','LMO-DP_Noise_(Ours)']:
                for model_name in models[model_family]:
                    df_plot_ = {}
                    for epsilon in [0.3,0.7,2,3]:
                        try:
                            my_dict=detail_Accuracy[noise_type][task_name][model_name][epsilon]
                            sorted_dict=dict(sorted(my_dict.items(),key=lambda item: item[1],reverse=True))
                            steps=list(sorted_dict.keys())[0]
                            accuracy=list(sorted_dict.values())[0]
                            
                            df_plot_[epsilon]=accuracy
                        except:
                            continue
                    
                    label_x = " ".join(noise_type.split("_"))
                    label_=f"{model_name} + {label_x}"
                    
                    x_axis=list(overall_epsilons[task_name].values())
                    if model_name==models[model_family][0] and noise_type=='Gaussian_Noise':
                        sns.lineplot(x=x_axis, y=df_plot_.values(), color='steelblue', markeredgecolor="steelblue", linewidth=linewidths[0], linestyle='--', marker='x', label=label_, markersize=markersize, markeredgewidth=markeredgewidth)
                    elif model_name==models[model_family][1] and noise_type=='Gaussian_Noise':
                        sns.lineplot(x=x_axis, y=df_plot_.values(), color='steelblue', markeredgecolor="steelblue", linewidth=linewidths[1], linestyle='--', marker='x', label=label_, markersize=markersize, markeredgewidth=markeredgewidth)
                    elif model_name==models[model_family][0] and noise_type=='LMO-DP_Noise_(Ours)':
                        sns.lineplot(x=x_axis, y=df_plot_.values(), color='chocolate', markeredgecolor="chocolate", linewidth=linewidths[0], linestyle='-', marker='x', label=label_, markersize=markersize, markeredgewidth=markeredgewidth)
                    elif model_name==models[model_family][1] and noise_type=='LMO-DP_Noise_(Ours)':
                        sns.lineplot(x=x_axis, y=df_plot_.values(), color='chocolate', markeredgecolor="chocolate", linewidth=linewidths[1], linestyle='-', marker='x', label=label_, markersize=markersize, markeredgewidth=markeredgewidth)
            
            if task_name=="MNLI-matched":
                plt.ylim(40,100)
            elif task_name=="SST-2":
                plt.ylim(40,100)
            elif task_name=="QNLI":
                plt.ylim(40,100)
            elif task_name=="QQP":
                plt.ylim(60,100)
            
            plt.xlabel(r'$\epsilon$', fontsize=fontsize_labels, fontweight='bold', color='black')
            plt.ylabel("Accuracy (%)", fontsize=fontsize_labels, fontweight='bold', color='black')
            plt.tick_params(axis='x', labelsize=fontsize_labels)
            plt.tick_params(axis='y', labelsize=fontsize_labels)
            plt.legend(loc='lower right',
                    fancybox=True,
                    edgecolor='black',
                    facecolor='white',
                    framealpha=0.4,
                    prop={'weight': 'bold', 'size': lengend_fontsize})
            ax = plt.gca()
            for label in ax.get_xticklabels():
                label.set_fontweight('bold')
                label.set_color('black')
            for label in ax.get_yticklabels():
                label.set_fontweight('bold')
                label.set_color('black')
            
            if model_family=='base':
                plt.savefig(os.path.join(plotting_path, f"figure3_{task_name}.pdf"))
                plt.savefig(os.path.join(plotting_path, f"figure3_{task_name}.png"))
            elif model_family=='large':
                plt.savefig(os.path.join(plotting_path, f"figure4_{task_name}.pdf"))
                plt.savefig(os.path.join(plotting_path, f"figure4_{task_name}.png"))


            # plot Figure 5 {x-epsilon; y-steps}
            if task_name in ['MNLI-matched', 'QQP']:
                # 1. store accuracy: when normal noise reach the highest acc, how many steps for lmo noise needs.
                min_acc=dict(zip(models[model_family],[1,1]))
                for noise_type in ['Gaussian_Noise','LMO-DP_Noise_(Ours)']:
                    for model_name in models[model_family]:
                        for epsilon in [0.3,0.7,2,3]:
                            my_dict=detail_Accuracy[noise_type][task_name][model_name][epsilon]
                            sorted_dict=dict(sorted(my_dict.items(),key=lambda item: item[1],reverse=True))
                            acc=list(sorted_dict.values())[0]
                            print(f"{noise_type} | {model_name} | {epsilon}: {acc}")
                            if acc<min_acc[model_name]:
                                min_acc[model_name]=acc
                
                acc_all={}
                for noise_type in ['Gaussian_Noise','LMO-DP_Noise_(Ours)']:
                    acc_all[noise_type]={}
                    for model_name in models[model_family]:
                        acc_all[noise_type][model_name]={}
                        for epsilon in [0.3,0.7,2,3]:
                            try:
                                my_dict=detail_Accuracy[noise_type][task_name][model_name][epsilon]
                                sorted_dict=dict(sorted(my_dict.items(),key=lambda item: item[1],reverse=True))
                                steps=list(sorted_dict.keys())[0]
                                accuracy=list(sorted_dict.values())[0]
                                
                                result_={key: value for key, value in sorted_dict.items() if value<min_acc[model_name]}
                                if result_=={}:
                                    acc_all[noise_type][model_name][epsilon]={steps:accuracy}
                                else:
                                    steps=list(result_.keys())[0]
                                    accuracy=list(result_.values())[0]

                                    acc_all[noise_type][model_name][epsilon]={steps:accuracy}
                            except:
                                continue
                
                # 2. plot the steps
                fig=plt.figure(figsize=(16,10))
                sns.set_theme(style="darkgrid")
                plt.style.use('ggplot')
                
                for noise_type in ['Gaussian_Noise','LMO-DP_Noise_(Ours)']:
                    for model_name in models[model_family]:
                        x0,y0=[],[]
                        for epsilon in [0.3,0.7,2,3]:
                            try:
                                y0.append(list(acc_all[noise_type][model_name][epsilon].keys())[0])
                                x0.append(epsilon)
                            except:
                                continue
                            
                        label_x = " ".join(noise_type.split("_"))
                        label_=f"{model_name} + {label_x}"

                        x0=list(overall_epsilons[task_name].values())
                        if model_name==models[model_family][0] and noise_type=='Gaussian_Noise':
                            sns.lineplot(x=x0, y=y0, color='steelblue', markeredgecolor="steelblue", linewidth=linewidths[0], linestyle='--', marker='x', label=label_, markersize=markersize, markeredgewidth=markeredgewidth)
                        elif model_name==models[model_family][1] and noise_type=='Gaussian_Noise':
                            sns.lineplot(x=x0, y=y0, color='steelblue', markeredgecolor="steelblue", linewidth=linewidths[1], linestyle='--', marker='x', label=label_, markersize=markersize, markeredgewidth=markeredgewidth)
                        elif model_name==models[model_family][0] and noise_type=='LMO-DP_Noise_(Ours)':
                            sns.lineplot(x=x0, y=y0, color='chocolate', markeredgecolor="chocolate", linewidth=linewidths[0], linestyle='-', marker='x', label=label_, markersize=markersize, markeredgewidth=markeredgewidth)
                        elif model_name==models[model_family][1] and noise_type=='LMO-DP_Noise_(Ours)':
                            sns.lineplot(x=x0, y=y0, color='chocolate', markeredgecolor="chocolate", linewidth=linewidths[1], linestyle='-', marker='x', label=label_, markersize=markersize, markeredgewidth=markeredgewidth)
                        
                plt.xlabel(r'$\epsilon$', fontsize=fontsize_labels, fontweight='bold', color='black')
                plt.ylabel("Steps numbers", fontsize=fontsize_labels, fontweight='bold', color='black')
                plt.tick_params(axis='x', labelsize=fontsize_labels)
                plt.tick_params(axis='y', labelsize=fontsize_labels)
                plt.legend(loc='upper right',
                        fancybox=True,
                        edgecolor='black',
                        facecolor='white',
                        framealpha=0.4,
                        prop={'weight': 'bold', 'size': lengend_fontsize})
                ax = plt.gca()
                for label in ax.get_xticklabels():
                    label.set_fontweight('bold')
                    label.set_color('black')
                for label in ax.get_yticklabels():
                    label.set_fontweight('bold')
                    label.set_color('black')
                
                if model_family=='base':
                    plt.savefig(os.path.join(plotting_path, f"figure5_{task_name}_base.pdf"))
                    plt.savefig(os.path.join(plotting_path, f"figure5_{task_name}_base.png"))
                elif model_family=='large':
                    plt.savefig(os.path.join(plotting_path, f"figure5_{task_name}_large.pdf"))
                    plt.savefig(os.path.join(plotting_path, f"figure5_{task_name}_large.png"))
            
            
            ## Plot Figure 8 & Figure 9 {x-steps; y-accuracy}
            fig=plt.figure(figsize=(13,11.5))
            sns.set_theme(style="darkgrid")
            plt.style.use('ggplot')
            for noise_type in ['Gaussian_Noise','LMO-DP_Noise_(Ours)']:
                for model_name in models[model_family]:
                    df_plot_=pd.DataFrame([],columns=["Training steps","Accuracy","type"])
                    for epsilon in [0.3,0.7]:
                        for step in detail_Accuracy[noise_type][task_name][model_name][epsilon]:
                            if step%step_gap[task_name]==int(min(step_gap.values())-1) and step<steps_in_each_epoch[task_name]*task1_settings['epoch']:
                                try:
                                    df_plot_.loc[len(df_plot_.index)]=[step,detail_Accuracy[noise_type][task_name][model_name][epsilon][step],f"{model_name} + {noise_type}"]
                                except:
                                    continue
                    
                    label_=f"{model_name} + {noise_type}"
                    if model_name==models[model_family][0] and noise_type=='Gaussian_Noise':
                        sns.lineplot(x="Training steps", y="Accuracy", color='steelblue', linestyle='--', label=label_, data=df_plot_, linewidth=linewidths[0])
                    elif model_name==models[model_family][1] and noise_type=='Gaussian_Noise':
                        sns.lineplot(x="Training steps", y="Accuracy", color='steelblue', linestyle='-', label=label_, data=df_plot_, linewidth=linewidths[0])
                    elif model_name==models[model_family][0] and noise_type=='LMO-DP_Noise_(Ours)':
                        sns.lineplot(x="Training steps", y="Accuracy", color='chocolate', linestyle='--', label=label_, data=df_plot_, linewidth=linewidths[0])
                    elif model_name==models[model_family][1] and noise_type=='LMO-DP_Noise_(Ours)':
                        sns.lineplot(x="Training steps", y="Accuracy", color='chocolate', linestyle='-', label=label_, data=df_plot_, linewidth=linewidths[0])
            
            if model_family=='base':
                if task_name=="MNLI-matched":
                    plt.ylim(20,80)
                elif task_name=="SST-2":
                    plt.ylim(35,90)
                elif task_name=="QNLI":
                    plt.ylim(35,90)
                elif task_name=="QQP":
                    plt.ylim(60,85)
            elif model_family=='large':
                if task_name=="MNLI-matched":
                    plt.ylim(20,85)
                elif task_name=="SST-2":
                    plt.ylim(35,95)
                elif task_name=="QNLI":
                    plt.ylim(40,90)
                elif task_name=="QQP":
                    plt.ylim(55,85)
            
            plt.xlabel("Training steps", fontsize=fontsize_labels, fontweight='bold', color='black')
            plt.ylabel("Accuracy (%)", fontsize=fontsize_labels, fontweight='bold', color='black')
            plt.tick_params(axis='x', labelsize=fontsize_labels)
            plt.tick_params(axis='y', labelsize=fontsize_labels)
            plt.legend(loc='lower right',
                    fancybox=True,
                    edgecolor='black',
                    facecolor='white',
                    framealpha=0.4,
                    prop={'weight': 'bold', 'size': lengend_fontsize})
            ax = plt.gca()
            for label in ax.get_xticklabels():
                label.set_fontweight('bold')
                label.set_color('black')
            for label in ax.get_yticklabels():
                label.set_fontweight('bold')
                label.set_color('black')
            
            if model_family=='base':
                plt.savefig(os.path.join(plotting_path, f"figure8_{task_name}.pdf"))
                plt.savefig(os.path.join(plotting_path, f"figure8_{task_name}.png"))
            elif model_family=='large':
                plt.savefig(os.path.join(plotting_path, f"figure9_{task_name}.pdf"))
                plt.savefig(os.path.join(plotting_path, f"figure9_{task_name}.png"))
            

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--generate_task1_commands_and_print', action='store_true')
    parser.add_argument('--done_task1_training', action='store_true')
    parser.add_argument('--plot_task1_figures', action='store_true')
    args = parser.parse_args()
    

    ## user define
    ## BEFORE fine-tuning: [task 1 settings]
    num_GPU=8  # GPU numbers in your environment
    task1_settings = {
        "noise_type": ['LMO', 'Gaussian'],
        "epsilon": [0.3, 0.7, 2, 3],  # epsilon for noise in one round in NLP fine-tuning.
        "task_name": ['mnli', 'sst-2', 'qnli', 'qqp'],
        "model_name": ['roberta-base', 'roberta-large', 'bert-base-cased', 'bert-large-cased'],
        "epoch": 6,
    }
    ## AFTER fine-tuning: [plot task 1 results]
    # 1. copy related files to one folder
    sourcepath = "results/fig34589/originalfiles"
    destination = "results/fig34589/jsonfiles"
    # 2. plot the task 1 figures
    plotting_path = "results/fig34589/plot_fig34589"


    ## BEFORE fine-tuning: [task 1 settings]
    if args.generate_task1_commands_and_print:
        generate_task1_commands(num_GPU, task1_settings)

        if not os.path.exists("running_logs"):
            os.makedirs("running_logs")
    ## AFTER fine-tuning: [plot task 1 results]
    # 1. copy related files to one folder
    if args.done_task1_training:
        copy_files_task1(sourcepath, destination, copy_files=True)
    # 2. plot the task 1 figures
    if args.plot_task1_figures:
        plot_task1_figures_func(plotting_path)
    
    