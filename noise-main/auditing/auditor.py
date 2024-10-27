import os
import numpy as np


class auditor:
    def backdoor_cv(self, modelname, bkd_x, bkd_y, save_dir, subtract=False):
        import tensorflow as tf
        model = tf.keras.models.load_model(os.path.join(save_dir, modelname))
        predsw = model.predict(bkd_x)
        predswo = model.predict(np.zeros_like(bkd_x))
        
        if subtract:
            diff = predsw - predswo
        else:
            diff = predsw
        pred = np.multiply(bkd_y, diff).sum()
        
        return pred

    
    def backdoor_nlp(self, config, x, y, model, modelpath, dataset, subtract=False):
        bkd_x, bkd_y, tokenizer = self.preprocess(x, y, dataset)
        model = self.build_model(config, tokenizer, model, modelpath)
        
        bkd_x, bkd_y = self.make_it_batch(bkd_x, bkd_y)
        
        model.eval()
        if "sst2b" in modelpath or "sst2_b" in modelpath or "qnli_b" in modelpath or "qnlib" in modelpath:
            bkd_x.pop("token_type_ids")
        try:
            predsw = model(**bkd_x)
        except:
            try:
                bkd_x.pop("token_type_ids")
                predsw = model(**bkd_x)
            except:
                raise NotImplementedError
        diff = predsw['logits'].detach().numpy()
        pred = np.multiply(bkd_y, diff).sum()
        print(pred)
        return pred
    
    def build_model_nlp_init(dataset, modelname):
        from init import CACHE_DIR
        from transformers import AutoConfig, AutoModelForSequenceClassification
        
        if modelname == "b":
            model_name_or_path = "bert-base-uncased"
        elif modelname == "r":
            model_name_or_path = "roberta-base"
        else:
            model_name_or_path = modelname
        
        config = AutoConfig.from_pretrained(
            model_name_or_path,
            num_labels=2,
            finetuning_task="sst-2" if dataset.startswith('sst2') else "qnli",
            cache_dir=CACHE_DIR,
        )

        model = AutoModelForSequenceClassification.from_pretrained(
            model_name_or_path,
            from_tf=bool(".ckpt" in model_name_or_path),
            config=config,
            cache_dir=CACHE_DIR,
        )
        
        return config, model

    def make_it_batch(self, x, y):
        import torch
        assert len(x) < 20
        new_x = {key: [] for key in x[0]}
        for xi in x:
            for item in xi:
                if item == "attention_mask":
                    xi[item] = new_x[item].append(xi[item][0])
                else:
                    xi[item] = new_x[item].append(xi[item])
        
        for x in new_x:
            new_x[x] = torch.stack(new_x[x])
        
        return new_x, [[i] for i in y]


