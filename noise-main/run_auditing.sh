## <auditing> || DPSGD | LMODP
# 1 generate settings
cd auditing/args
python generate_settings.py nlp
python generate_settings.py cv
cd ..
# 2 run code for CV datasets.
conda activate tf_env
python main1.py gaussian fmnist lr ok
python main1.py lmo fmnist lr ok
python main2.py fmnist lr ok
python main3.py gaussian fmnist lr 30
python main3.py lmo fmnist lr 30
# 3 run code for NLP datasets.
# conda activate lmo-llama2
# python main1.py gaussian qnli r ok
# python main1.py lmo qnli r ok
# python main2.py qnli r ok
# python main3.py gaussian qnli r 10
# python main3.py lmo qnli r 10
