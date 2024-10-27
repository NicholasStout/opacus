import os
import shutil
import argparse
import pandas as pd
from experiments.check_privacy_and_statistics import compute_overall_privacy, DEFAULT_DELTA2


## BEFORE fine-tuning: [task 2 settings]
def generate_task2_command(despath, noise_type, task_name, model_name, epsilon, data_dir, GPU_idx=0):
    sigma = compute_overall_privacy(float(epsilon), DEFAULT_DELTA2, dataset="E2E", compute_sigma_only=True)
    cache_dir = os.path.join("..", despath, "cache", f"{noise_type}.{task_name}.{model_name}.eps_{epsilon}")
    despath = os.path.join("..", despath, f"{noise_type}.{task_name}.{model_name}.eps_{epsilon}")
    
    if noise_type=="LMO":
        lmo_filepath = os.path.join("experiments", "lmo_noise_parameters", "table2", f"lmo_eps{noise_type}.json")
        
        cmd = f'''
CUDA_VISIBLE_DEVICES={GPU_idx} python3 -m table2text.run_language_modeling_lmo \
--output_dir {despath} --overwrite_output_dir \
--task_mode e2e \
--model_name_or_path {model_name} \
--tokenizer_name {model_name} \
--do_train --do_eval \
--line_by_line \
--save_steps 100 --save_total_limit 1 --save_at_last no \
--logging_dir {despath} --logging_steps -1 \
--seed 0 \
--eval_steps 100 --eval_epochs 2 --max_eval_batches 100 --evaluation_strategy epoch --evaluate_before_training "no" --evaluate_during_training "yes" --per_device_eval_batch_size 10 \
--max_generations 9223372036854775807 --max_generations_train 10 --max_generations_valid 9223372036854775807 \
--max_train_examples 9223372036854775807 --max_valid_examples 9223372036854775807 --max_eval_examples 9223372036854775807 \
--data_folder {data_dir} --max_seq_len 100 --format_mode cat \
--per_example_max_grad_norm 1 --target_delta 8e-6 \
--noise_multiplier {sigma} --lmo_filepath {lmo_filepath} \
--learning_rate 2e-3 --lr_decay "no" --num_train_epochs 10 --per_device_train_batch_size 16 --gradient_accumulation_steps 64 \
--non_private no --clipping_mode ghost \
--cache_dir {cache_dir} 2>&1 > ../../running_logs/table2.{noise_type}.{task_name}.{model_name}.eps_{epsilon}.log &
'''
    elif noise_type=="Gaussian":
        cmd = f'''
CUDA_VISIBLE_DEVICES={GPU_idx} python3 -m table2text.run_language_modeling \
--output_dir {despath} --overwrite_output_dir \
--task_mode e2e \
--model_name_or_path {model_name} \
--tokenizer_name {model_name} \
--do_train --do_eval \
--line_by_line \
--save_steps 100 --save_total_limit 1 --save_at_last no \
--logging_dir {despath} --logging_steps -1 \
--seed 0 \
--eval_steps 100 --eval_epochs 2 --max_eval_batches 100 --evaluation_strategy epoch --evaluate_before_training "no" --evaluate_during_training "yes" --per_device_eval_batch_size 10 \
--max_generations 9223372036854775807 --max_generations_train 10 --max_generations_valid 9223372036854775807 \
--max_train_examples 9223372036854775807 --max_valid_examples 9223372036854775807 --max_eval_examples 9223372036854775807 \
--data_folder {data_dir} --max_seq_len 100 --format_mode cat \
--per_example_max_grad_norm 1 --target_delta 8e-6 \
--noise_multiplier {sigma} \
--learning_rate 2e-3 --lr_decay "no" --num_train_epochs 10 --per_device_train_batch_size 16 --gradient_accumulation_steps 64 \
--non_private no --clipping_mode ghost \
--cache_dir {cache_dir} 2>&1 > ../../running_logs/table2.{noise_type}.{task_name}.{model_name}.eps_{epsilon}.log &
'''

    return cmd


def generate_task2_commands(num_GPU, task2_settings, destination0, data_dir):
    GPU_idx=0
    for noise_type in task2_settings['noise_type']:
        for task_name in task2_settings['task_name']:
            for model_name in task2_settings['model_name']:
                for epsilon in task2_settings['epsilon']:
                    if noise_type=="LMO":
                        cmd=generate_task2_command(destination0, noise_type, task_name, model_name, epsilon, data_dir, GPU_idx)
                    elif noise_type=="Gaussian":
                        cmd=generate_task2_command(destination0, noise_type, task_name, model_name, epsilon, data_dir, GPU_idx)
                    
                    # write commands into scipts
                    print(cmd)
                    
                    GPU_idx = GPU_idx + 1
                    if GPU_idx == num_GPU:
                        GPU_idx = 0


## AFTER fine-tuning: [task 2 settings]
## copy related files to specific folder
dict_plot={
    "Gaussian": "Gaussian_Noise",
    "LMO": "LMO-DP_Noise_(Ours)",
}


def copy_files_task2(sourcepath, destination, Non_private=False, copy_files=False):
    # dataset: E2E
    for noisetype in ['Gaussian','LMO']:
        for epsilon in [0.3,0.7,2,3]:
            source_path=os.path.join(sourcepath,f"{noisetype}.eps{epsilon}_v1","eval","global_step_00000410.txt")
            destination_path=os.path.join(destination,f"E2E@GPT-2@{dict_plot[noisetype]}@{epsilon}@.txt")
            
            if not os.path.exists(destination):
                os.makedirs(destination)
            
            if os.path.exists(source_path) and copy_files:
                shutil.copy(source_path, destination_path)

    if Non_private:
        # Non_private
        source_path=os.path.join(sourcepath,f"Non_private","eval","global_step_00000410.txt")
        destination_path=os.path.join(destination,f"E2E@GPT-2@Non_private@Non_private@.txt")
        
        if os.path.exists(source_path) and copy_files:
            shutil.copy(source_path, destination_path)


def statistics_task2(sourcepath, destination, generate_csv=False):
    results=pd.DataFrame([], columns=['noisetype','eps','num','BLEU','NIST','METEOR','ROUGE_L','CIDEr'])
    for idx, log in enumerate(os.listdir(sourcepath)):
        filepath=os.path.join(sourcepath, log)

        noisetype, eps, num = log.split("_")
        try:
            overall_epsilon, _ = compute_overall_privacy(float(eps), DEFAULT_DELTA2, dataset="E2E")
        except:
            overall_epsilon = {}
            overall_epsilon['eps_rdp'] = 0
        num=num.split(".")[0]
        
        with open(filepath, 'r', encoding='utf-8') as f:
            lines=f.readlines()
        for line in lines:
            try:
                if line.startswith("BLEU"):
                    BLEU = float(line.strip().split("BLEU: ")[1]) * 100
                elif line.startswith("NIST"):
                    NIST = float(line.strip().split("NIST: ")[1])
                elif line.startswith("METEOR"):
                    METEOR = float(line.strip().split("METEOR: ")[1])
                elif line.startswith("ROUGE_L"):
                    ROUGE_L = float(line.strip().split("ROUGE_L: ")[1]) * 100
                elif line.startswith("CIDEr"):
                    CIDEr = float(line.strip().split("CIDEr: ")[1])
            except:
                continue

        results.loc[len(results.index)] = [noisetype, overall_epsilon['eps_rdp'], num, BLEU, NIST, METEOR, ROUGE_L, CIDEr]

    if generate_csv:
        results.to_csv(destination, index=False)
    else:
        print(results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--generate_task2_commands_and_print', action='store_true')
    parser.add_argument('--done_task2_training', action='store_true')
    parser.add_argument('--generate_task2_csv', action='store_true')
    args = parser.parse_args()
    
    
    ## user define
    ## BEFORE fine-tuning: [task 1 settings]
    num_GPU=8  # GPU numbers in your environment
    task2_settings = {
        "noise_type": ['LMO', 'Gaussian'],
        "epsilon": [0.3, 0.7, 2, 3],  # epsilon for noise in one round in NLP fine-tuning.
        "task_name": ['E2E'],
        "model_name": ['GPT-2'],
        "epoch": 10,
    }
    destination0 = "results/table2/table2"
    data_dir = "/mnt/data/nlp/prefix-tuning"
    ## AFTER fine-tuning: [plot task 1 results]
    # 1. copy related files to one folder
    sourcepath1 = "results/table2/table2"
    destination1 = "results/table2/generated_txt"
    # 2. compute the metrics of generated txt
    sourcepath2 = "results/table2/generated_txt"
    destination2 = "results/table2/metrics"
    # 3. plot the task 1 figures
    sourcepath3 = "results/table2/metrics"
    destination3 = "results/table2/table2.csv"

    ## BEFORE fine-tuning: [task 1 settings]
    if args.generate_task2_commands_and_print:
        generate_task2_commands(num_GPU, task2_settings, destination0, data_dir)

        if not os.path.exists("running_logs"):
            os.makedirs("running_logs")
    ## AFTER fine-tuning: [plot task 1 results]
    # 1. copy related files to one folder
    if args.done_task2_training:
        copy_files_task2(sourcepath1, destination1, Non_private=True, copy_files=args.done_task2_training)

    # 2. compute the metrics of generated txt

    # 3. plot the task 1 tables
    if args.generate_task2_csv:
        statistics_task2(sourcepath3, destination3, generate_csv=args.generate_task2_csv)
