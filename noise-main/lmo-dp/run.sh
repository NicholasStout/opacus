## demo to show the privacy computation process
python3 -m experiments.check_privacy_and_statistics
wait

## running experiments for Figure 2 6 7 (when compared with optimized-usefulness noises)
# fig2 parameters: ./experiments/lmo_noise_parameters/optimized_usefulness
# fig6 parameters: ./experiments/lmo_noise_parameters/fig6
# fig7 parameters: ./experiments/lmo_noise_parameters/task1
# output folder: ./results/fig267/plot_fig267
python3 run_fig267.py
wait

## running experiments for task 1 and plotting figures of Figure 3 4 5 8 9
# [BEFORE running task 1] generate running commands for task 1
# task 1 parameters: ./experiments/lmo_noise_parameters/task1
python3 run_fig34589.py --generate_task1_commands_and_print > experiments/private-transformers/commands_task1.sh
cd experiments/private-transformers/classification/data/
bash download_dataset.sh
wait
cd ../..
bash commands_task1.sh
wait
# [AFTER running task 1] plot the figures after collecting all the results
cd ../..
python3 run_fig34589.py --done_task1_training --plot_task1_figures
wait

# running experiments for task 2 in table 2
# [BEFORE running task 2] generate running commands for task 2
# task 2 parameters: ./experiments/lmo_noise_parameters/task2
python3 run_table2.py --generate_task2_commands_and_print > experiments/private-transformers/commands_task2.sh
cd experiments/experiments/private-transformers
bash commands_task2.sh
wait
# [AFTER running task 2] plot the figures after collecting all the results
cd ../..
python3 run_table2.py --done_task2_training
# compute metrics by e2e-metrics
python3 run_table2.py --generate_task2_csv
wait

# running experiments for table 3 -- task 1 (SST-2)
# task 3 parameters: ./experiments/lmo_noise_parameters/table3
# Running Bu(2022 & 2023)
cd experiments/fast-differential-privacy/text_classification/data/
bash download_dataset.sh
wait
cd ../..
bash run.sh
cd ..
# Running Yu
cd Differentially-Private-Fine-tuning-of-Language-Models/Language-Understanding-RoBERTa/bert_lora
bash run.sh

