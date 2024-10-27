import os, sys, json
import numpy as np
import subprocess

from absl import app
from absl import flags



flags.DEFINE_string('dataset', 'fmnist', 'fmnist, p100, sst2, qnli.')
flags.DEFINE_string('model', '2f', '[fmnist, p100:] 2f, lr; [sst2, qnli:] r, b.')
flags.DEFINE_integer('n_pois', 8, '[Numbers of poisoned samples:] 1, 2, 4, 8.')
flags.DEFINE_float('l2_norm_clip', 1.0, '[Clipping norm] 1')
flags.DEFINE_string('exp_name', None, '[Name of experiment] dataset, model, n/bkd, clip_norm, noise_type, noise_param, trial')
flags.DEFINE_string('noise_type', 'gaussian', '[type of noise] gaussian, lmo')
flags.DEFINE_float('noise_params', 1.1, '[For gaussian: ratio of the standard deviation to the clipping norm; For lmo: lmo params index]')
flags.DEFINE_boolean('backdoor', False, '[whether to backdoor] False, True.')
flags.DEFINE_float('learning_rate', 0.15, 'Learning rate for training')
flags.DEFINE_integer('microbatches', 250, 'Number of microbatches (must evenly divide batch_size)')
flags.DEFINE_integer('epochs', 24, 'Number of epochs')
flags.DEFINE_integer('batch_size', 250, 'Batch size')
FLAGS = flags.FLAGS



import torch
from ml_swissknife import utils
from transformers import AutoConfig, AutoTokenizer, AutoModelForSequenceClassification
from transformers import HfArgumentParser

from lmo_config.privacy_setting import DEFAULT_DELTA
from lmo_config.src.init import *
from lmo_config.src.trainer import Trainer
from lmo_config.src.private_transformers import PrivacyEngine
from lmo_config.src.compiled_args import PrivacyArguments, AuxiliaryArguments

from init import CACHE_DIR
os.environ["TOKENIZERS_PARALLELISM"] = "false"



class auditing_NLP:
    def load_data(self, data_dir):
        FLAGS.batch_size = 256
        FLAGS.epochs = 3
        FLAGS.learning_rate = 5e-4
        modelname = "roberta-base" if FLAGS.model.startswith("r") else "bert-base-uncased"
        
        data = {}
        for i in range(4):
            path = os.path.join(data_dir, f"{FLAGS.dataset}", modelname, f"{FLAGS.dataset}-{FLAGS.n_pois}-{i}.npy")
            data[i] = np.load(path, allow_pickle=True)
        nobkd_trn, bkd_trn, p, tst = data[0].tolist(), data[1].tolist(), data[2].tolist(), data[3].tolist()
        
        if FLAGS.dataset.startswith("sst2"):
            nobkd_trn_x, nobkd_trn_y = [i[0] for i in nobkd_trn], [int(i[1]) for i in nobkd_trn]
            bkd_trn_x, bkd_trn_y = [i[0] for i in bkd_trn], [int(i[1]) for i in bkd_trn]
            bkd_x, bkd_y = [i[0] for i in p], [int(i[1]) for i in p]
            tst_x, tst_y = [i[0] for i in tst], [int(i[1]) for i in tst] 
        elif FLAGS.dataset.startswith("qnli"):
            nobkd_trn_x, nobkd_trn_y = [[i[1], i[2]] for i in nobkd_trn], [i[3] for i in nobkd_trn]
            bkd_trn_x, bkd_trn_y = [[i[1], i[2]] for i in bkd_trn], [i[3] for i in bkd_trn]
            bkd_x, bkd_y = [[i[1], i[2]] for i in p], [i[3] for i in p]
            tst_x, tst_y = [[i[1], i[2]] for i in tst], [i[3] for i in tst]
            
        return bkd_trn_x, bkd_trn_y, nobkd_trn_x, nobkd_trn_y, bkd_x, bkd_y, tst_x, tst_y

    def build_model(self, tokenizer):
        if FLAGS.model == "b":
            model_name_or_path = "bert-base-uncased"
        elif FLAGS.model == "r":
            model_name_or_path = "roberta-base"
        else:
            model_name_or_path = FLAGS.model
        
        config = AutoConfig.from_pretrained(
            model_name_or_path,
            num_labels=2,
            finetuning_task="sst-2" if FLAGS.dataset.startswith('sst2') else "qnli",
            cache_dir=CACHE_DIR,
        )
        
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name_or_path,
            from_tf=bool(".ckpt" in model_name_or_path),
            config=config,
            cache_dir=CACHE_DIR,
        )
        
        if config.model_type == 'bert':
            model.resize_token_embeddings(len(tokenizer))
            resize_token_type_embeddings(model, new_num_types=10, random_segment=False)
        
        return model
    
    def save_model(self, filepath, imported_args):
        command = ["mv", os.path.join(filepath, "model.safetensors"), filepath+".safetensors"]
        mvmodels = subprocess.run(command, capture_output=True, text=True)
        command = ["mv", os.path.join(filepath, "log_history.json"), filepath+".log_history"]
        mvlogs = subprocess.run(command, capture_output=True, text=True)
        
        if mvmodels.returncode == 0 and mvlogs.returncode == 0:
            command = ["rm", "-rf", filepath]
            subprocess.run(command, capture_output=True, text=True)
            print(f"Model is successfully saved at {filepath}.safetensors.")
        else:
            print(f"Please check the code and results for {imported_args}")
    
    def train_model_run(self, model, train_dataset, dev_dataset, filepath, new_seed):
        torch.manual_seed(new_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(new_seed)
            torch.cuda.manual_seed_all(new_seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        
        ignore_args = ['--backdoor', '--dataset', '--nobackdoor', '--n_pois', '--l2_norm_clip', '--exp_name', '--noise_type', '--noise_params']
        filtered_args = [arg for arg in sys.argv if not any(arg.startswith(ignore) for ignore in ignore_args)][1:]
        filtered_args.append(f'--output_dir={filepath}')
        parser = HfArgumentParser((ModelArguments, DynamicTrainingArguments, PrivacyArguments, AuxiliaryArguments))
        model_args, training_args, privacy_args, auxiliary_args = parser.parse_args_into_dataclasses(args=filtered_args)
        training_args.local_rank=-1
        
        if FLAGS.model == "b":
            model_args.model_name_or_path = "bert-base-uncased"
        elif FLAGS.model == "r":
            model_args.model_name_or_path = "roberta-base"
        
        training_args.per_device_train_batch_size = FLAGS.batch_size
        training_args.num_train_epochs = FLAGS.epochs
        
        trainer = Trainer(
                model=model,
                args=training_args,
                model_args=model_args,
                privacy_args=privacy_args,
                auxiliary_args=auxiliary_args,
                train_dataset=train_dataset,
                eval_dataset=dev_dataset,
                compute_metrics=build_compute_metrics_fn("sst-2" if FLAGS.dataset.startswith('sst2') else "qnli")
            )
        
        named_params = [(name, param) for name, param in model.named_parameters() if param.requires_grad]
        print('Params to update: ')
        print(json.dumps([name for name, param in named_params], indent=4))
        num_differentiable_params = utils.count_parameters(model, only_differentiable=True)
        print(f'Number of differentiable params: {num_differentiable_params / 1e6:.3f} million')
        
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in named_params if not any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
            {'params': [p for n, p in named_params if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = trainer.optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters,
            lr=FLAGS.learning_rate,
            betas=(0.9, 0.98),
            eps=1e-08,
        )
        trainer.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(trainer.optimizer, lambda _: 1.)
        print(FLAGS.noise_type)
        privacy_engine = PrivacyEngine(
            module=model,
            batch_size=FLAGS.batch_size,
            sample_size=len(train_dataset),
            max_grad_norm=FLAGS.l2_norm_clip,
            noise_multiplier=FLAGS.noise_params,
            epochs=FLAGS.epochs,
            target_epsilon=None,
            target_delta=DEFAULT_DELTA,
            accounting_mode="rdp",
            clipping_mode="ghost",
            skip_checks=True,
            lmo_filepath=FLAGS.noise_type,
        )
        
        privacy_engine.attach(optimizer)
        
        trainer.train(model_path=None)
        trainer.save_model(filepath)

    def preprocess(self, x, y, dataset):
        if FLAGS.model == "b":
            model_name_or_path = "bert-base-uncased"
        elif FLAGS.model == "r":
            model_name_or_path = "roberta-base"
        else:
            model_name_or_path = FLAGS.model
        
        tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path, use_fast=True,
            additional_special_tokens=[],
            cache_dir=CACHE_DIR,
        )
        
        if dataset.startswith("sst2"):
            encoded_inputs = [
                tokenizer(xi,
                padding='max_length',
                truncation=True,
                return_tensors="pt",
                max_length=60,
                ) for xi in x]
        elif dataset.startswith("qnli"):
            encoded_inputs = [
                tokenizer(xi[0], xi[1],
                padding='max_length',
                truncation=True,
                return_tensors="pt",
                max_length=60,
                ) for xi in x]
        for idx, item in enumerate(encoded_inputs):
            encoded_inputs[idx]['input_ids'] = item['input_ids'][0]

        if dataset.startswith("qnli"):
            map_labels={"entailment": 0, "not_entailment": 1}
            y = [map_labels[yi] for yi in y]
        
        return encoded_inputs, y, tokenizer

    def train_model(self, model, train_x, train_y, dev_x, dev_y, filepath, new_seed):
        train_dataset = MyDataset(train_x, train_y)
        
        torch.manual_seed(new_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(new_seed)
            torch.cuda.manual_seed_all(new_seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        
        self.train_model_run(model, train_dataset, train_dataset, filepath, new_seed)


def main(_):
    from init import init
    exp_type, data_dir, _, model, _, _, _, _, save_dir, _, _ = init(FLAGS.noise_type, FLAGS.dataset, FLAGS.model)
    
    
    assert "cv" in exp_type or "nlp" in exp_type
    suffix = '.h5' if "cv" in exp_type else '.safetensors'
    savepath = os.path.join(save_dir, FLAGS.exp_name+suffix)
    if os.path.exists(savepath):
        print(f"{savepath} exists.")
        exit(0)
    savepath = os.path.join(save_dir, FLAGS.exp_name)
    
    np.random.seed(0)
    auditing = auditing_NLP()

    bkd_trn_x, bkd_trn_y, nobkd_trn_x, nobkd_trn_y, bkd_x, bkd_y, tst_x, tst_y = auditing.load_data(data_dir)
    bkd_trn_x, bkd_trn_y, _ = auditing.preprocess(bkd_trn_x, bkd_trn_y, FLAGS.dataset)
    nobkd_trn_x, nobkd_trn_y, _ = auditing.preprocess(nobkd_trn_x, nobkd_trn_y, FLAGS.dataset)
    bkd_x, bkd_y, _ = auditing.preprocess(bkd_x, bkd_y, FLAGS.dataset)
    tst_x, tst_y, tokenizer = auditing.preprocess(tst_x, tst_y, FLAGS.dataset)

    model = auditing.build_model(tokenizer)

    if FLAGS.backdoor:
        trn_x, trn_y = bkd_trn_x, bkd_trn_y
    else:
        trn_x, trn_y = nobkd_trn_x, nobkd_trn_y
    
    
    np.random.seed(None)
    new_seed = np.random.randint(1000000)
    auditing.train_model(model, trn_x, trn_y, trn_x, trn_y, savepath, new_seed)
    auditing.save_model(savepath, f'args.{FLAGS.noise_type}.{FLAGS.dataset}_{FLAGS.model}')



if __name__ == '__main__':
    app.run(main)