import os, sys
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import numpy as np
np.random.seed(None)
from collections import defaultdict

from absl import app
from absl import flags

BATCH_SIZE = 50



flags.DEFINE_string('dataset', 'fmnist', 'fmnist, p100, sst2, qnli.')
flags.DEFINE_string('model', '2f', '[fmnist, p100:] 2f, lr; [sst2, qnli:] r, b.')
flags.DEFINE_integer('n_pois', 8, '[number of clusters:] 1, 2, 4, 8.')
flags.DEFINE_float('l2_norm_clip', 1.0, '[Clipping norm] 1')
flags.DEFINE_string('exp_name', None, '[name of experiment] dataset, model, n/bkd, clip_norm, noise_type, noise_param, trial')
flags.DEFINE_string('noise_type', 'gaussian', '[type of noise] gaussian, lmo')
flags.DEFINE_float('noise_params', 1.1, '[For gaussian: ratio of the standard deviation to the clipping norm; For lmo: lmo params index]')
flags.DEFINE_boolean('backdoor', False, '[whether to backdoor] False, True.')
flags.DEFINE_float('learning_rate', 0.15, 'Learning rate for training')
flags.DEFINE_integer('microbatches', 1, 'Number of microbatches (must evenly divide batch_size)')
flags.DEFINE_integer('epochs', 24, 'Number of epochs')
flags.DEFINE_integer('batch_size', 250, 'Batch size')
FLAGS = flags.FLAGS


from init import init
from utils import get_cfg
from auditor import auditor


def main(_):
    # python infer_nlp.py 0 50 bkd gaussian 22.73 2 sst2 r
    start = sys.argv[1]
    end = sys.argv[2]
    bkd_if = sys.argv[3]
    noise_type = sys.argv[4]
    noise_param = str(sys.argv[5])
    pois_ct = sys.argv[6]
    dataset = sys.argv[7]
    model = sys.argv[8]
    
    
    if bkd_if=="nbkd" and int(pois_ct) > 1:
        exit(0)
    
    exp_type, data_dir, dataset, model, _, _, _, noise_type, save_dir, _, _ = init(noise_type, dataset, model)
    res_dir = os.path.join(save_dir, "results")
    os.makedirs(res_dir, exist_ok=True)
    
    cfg_map = defaultdict(list)
    tensors = [fname for fname in os.listdir(save_dir) if fname.endswith('.safetensors')]
    for tensor in tensors:
        cfg_map[get_cfg(tensor)].append(tensor)
    for val in cfg_map:
        cfg_map[val] = sorted(cfg_map[val], key=lambda h: int(h.split('-')[-1].split(".")[0]))
        print(val, cfg_map[val]) # ('bkd', 'lmo', '3', '8') ['sst2_r-bkd-lmo-3-8-0.safetensors', 'sst2_r-bkd-lmo-3-8-1.safetensors']
    
    saved_name = '-'.join([bkd_if, noise_type, noise_param, pois_ct, start, end])
    
    cfg_key = (bkd_if, noise_type, noise_param, pois_ct)
    print(saved_name, len(cfg_map[cfg_key]))
    
    
    
    assert exp_type == "nlp"
    modelname = "roberta-base" if model.startswith("r") else "bert-base-uncased"
    path = os.path.join(os.path.join(data_dir, dataset, modelname), f"{dataset}-{pois_ct}-2.npy")
    p = np.load(path, allow_pickle=True).tolist()
    if "sst2" in dataset:
        bkd_x, bkd_y = [i[0] for i in p], [np.eye(2)[int(i[1])] for i in p]
    elif "qnli" in dataset:
        bkd_x, bkd_y = [[i[1], i[2]] for i in p], [i[3] for i in p]
    all_bkds = (bkd_x[0:1], bkd_y[0:1])
    print(all_bkds)
    
    
    
    alls = []
    auditing = auditor()
    for h5 in cfg_map[cfg_key][int(start):int(end)]:
        x, y = all_bkds
        
        modelpath = os.path.join(save_dir, h5)
        config, model = auditing.build_model_nlp_init(dataset, modelname)
        nob_vals = auditing.backdoor_nlp(config, x, y, model, modelpath)
        alls.append(nob_vals)
    
    if alls == []:
        print(f'check this setting {cfg_key}')
    else:
        np.save(os.path.join(res_dir, '-'.join(["batch", saved_name])), np.array(alls))
        print(f"the results are saved in {res_dir}!")
    

   
    
if __name__ == '__main__':
    app.run(main)
    

# from sys import argv
# key = argv[1]
# start = int(argv[2])
# end = int(argv[3])
# pois_ct = argv[4]
# clip_norm = argv[5]
# noise = argv[6]
# init  = argv[7]
# data_dir = auditing_args.args["data_dir"]
# save_dir = auditing_args.args["save_dir"]
# res_dir = os.path.join(save_dir, "results")
# os.makedirs(res_dir, exist_ok=True)
# get_mi = False

# # sig  = {0.3: 22.73, 0.7: 9.74, 2: 3.41, 3: 2.27}
# # sig2 = {22.73: 0.3, 9.74: 0.7, 3.41: 2, 2.27: 3}
# # if float(noise) in sig2:
# #     noise = str(sig2[float(noise)])
    
# dataset_auditing = auditing_args.args["dataset"]

# if "fmnist" in dataset_auditing:
#     all_bkds = {
#             "p": np.load(data_dir + "/fmnist/clipbkd-new-1.npy", allow_pickle=True)[2],
#             "tst": np.load(data_dir + "/fmnist/clipbkd-new-1.npy", allow_pickle=True)[3],
#             "trn": np.load(data_dir + "/fmnist/clipbkd-new-1.npy", allow_pickle=True)[0]
#             }
#     all_bkds["p"] = all_bkds["p"][0].reshape((-1, 28, 28, 1)), np.eye(2)[all_bkds["p"][1]][None, :]
#     all_bkds["tst"] = all_bkds["tst"][0].reshape((-1, 28, 28, 1)), np.eye(2)[all_bkds["tst"][1]]
#     all_bkds["trn"] = all_bkds["trn"][0].reshape((-1, 28, 28, 1)), np.eye(2)[all_bkds["trn"][1]]
# elif "p100" in dataset_auditing:
#     # x[0][0].shape (10000, 100)
#     # x[0][1].shape (10000,)
#     # x[1][0].shape (10000, 100)
#     # x[1][1].shape (10000,)
#     # x[2][0].shape (1, 100)
#     # x[2][1]       11
#     # x[3][0].shape (10000, 100)
#     # x[3][1].shape (10000,)
#     # （用下毒数据替换了pois_ct行后的新数据，但pois数据的标签反了），（用下毒数据替换了pois_ct行后的新数据），（下毒数据），（测试数据）
#     all_bkds = {
#             "p": np.load(data_dir + "/p100/p100_1.npy", allow_pickle=True)[2],
#             "tst": np.load(data_dir + "/p100/p100_1.npy", allow_pickle=True)[3],
#             "trn": np.load(data_dir + "/p100/p100_1.npy", allow_pickle=True)[0]
#             }
#     all_bkds["p"] = all_bkds["p"][0].reshape((-1, 100)), all_bkds["p"][1]
#     all_bkds["tst"] = all_bkds["tst"][0].reshape((-1, 100)), all_bkds["tst"][1]
#     all_bkds["trn"] = all_bkds["trn"][0].reshape((-1, 100)), all_bkds["tst"][1]

# if "qnli" in dataset_auditing:
#     modelname = "roberta-base" if "r" in dataset_auditing else "bert-base-uncased"
#     if pois_ct != ".":
#         data = {}
#         for i in range(4):
#             path = os.path.join(os.path.join(data_dir, "qnli_auditing", modelname), f"qnli-{pois_ct}-{i}.npy")
#             data[i] = np.load(path, allow_pickle=True)
#         nobkd_trn, bkd_trn, p, tst = data[0].tolist(), data[1].tolist(), data[2].tolist(), data[3].tolist()
        
#         # nobkd_trn_x, nobkd_trn_y = [i[0] for i in nobkd_trn], [int(i[1]) for i in nobkd_trn]
#         # bkd_trn_x, bkd_trn_y = [i[0] for i in bkd_trn], [int(i[1]) for i in bkd_trn]
#         # bkd_x, bkd_y = [i[0] for i in p], [int(i[1]) for i in p]
#         # tst_x, tst_y = [i[0] for i in tst], [int(i[1]) for i in tst]
        
            
#         nobkd_trn_x, nobkd_trn_y = [[i[1], i[2]] for i in nobkd_trn], [i[3] for i in nobkd_trn]
#         bkd_trn_x, bkd_trn_y = [[i[1], i[2]] for i in bkd_trn], [i[3] for i in bkd_trn]
#         bkd_x, bkd_y = [[i[1], i[2]] for i in p], [i[3] for i in p]
#         tst_x, tst_y = [[i[1], i[2]] for i in tst], [i[3] for i in tst]
        
        
#         # all_bkds = {
#         #     "p": (bkd_x, bkd_y),
#         #     "tst": (tst_x, tst_y),
#         #     "trn": (nobkd_trn_x, nobkd_trn_y)
#         # }
        
#         all_bkds = {
#             "p": (bkd_x[0:1], bkd_y[0:1]),
#             "tst": (tst_x, tst_y),
#             "trn": (nobkd_trn_x, nobkd_trn_y)
#         }
#     else:
#         all_bkds = {}
#         for j in [1,2,4,8]:
#             data = {}
#             for i in range(4):
#                 path = os.path.join(os.path.join(data_dir, "sst2_auditing", modelname), f"sst2-{j}-{i}.npy")
#                 data[i] = np.load(path, allow_pickle=True)
#             nobkd_trn, bkd_trn, p, tst = data[0].tolist(), data[1].tolist(), data[2].tolist(), data[3].tolist()
            
#             # nobkd_trn_x, nobkd_trn_y = [i[0] for i in nobkd_trn], [np.eye(2)[int(i[1])] for i in nobkd_trn]
#             # bkd_trn_x, bkd_trn_y = [i[0] for i in bkd_trn], [np.eye(2)[int(i[1])] for i in bkd_trn]
#             # bkd_x, bkd_y = [i[0] for i in p], [np.eye(2)[int(i[1])] for i in p]
#             # tst_x, tst_y = [i[0] for i in tst], [np.eye(2)[int(i[1])] for i in tst]
            
#             nobkd_trn_x, nobkd_trn_y = [[i[1], i[2]] for i in nobkd_trn], [i[3] for i in nobkd_trn]
#             bkd_trn_x, bkd_trn_y = [[i[1], i[2]] for i in bkd_trn], [i[3] for i in bkd_trn]
#             bkd_x, bkd_y = [[i[1], i[2]] for i in p], [i[3] for i in p]
#             tst_x, tst_y = [[i[1], i[2]] for i in tst], [i[3] for i in tst]
        
#             all_bkds_ = {
#                 "p": (bkd_x[0:1], bkd_y[0:1]),
#                 "tst": (tst_x, tst_y),
#                 "trn": (nobkd_trn_x, nobkd_trn_y)
#             }
#             all_bkds[j] = all_bkds_

# h5s = [fname for fname in os.listdir(save_dir) if fname.endswith('.safetensors')]

# def argv_to_cfg():
#     if key == 'no':
#         return ('no', '.', clip_norm, noise, init)
#     else: 
#         return ('new', pois_ct, clip_norm, noise, init)


# def get_cfg(h5):
#     splt = h5.split('-')
#     if 'no' in h5:
#         return ('no', splt[1], splt[2], splt[3], splt[4])
#         # return ('no', '.', splt[2], splt[3], splt[4])
#     else:
#         return ('new', splt[1], splt[2], splt[3], splt[4])

# cfg_map = defaultdict(list)

# for h5 in h5s:
#     cfg_map[get_cfg(h5)].append(h5)
# cfg_key = argv_to_cfg()

# def mi(h5name):
#     from scipy.special import softmax
#     model = tf.keras.models.load_model(os.path.join(save_dir, h5name))
#     trn_x, trn_y = all_bkds['trn']
#     tst_x, tst_y = all_bkds['tst']
#     print(trn_y.shape, tst_y.shape)
#     np.random.seed(0)
#     tst_y_len = tst_y.shape[0]
#     trn_y_inds = np.random.choice(trn_y.shape[0], tst_y_len, replace=False)
#     trn_x, trn_y = trn_x[trn_y_inds], trn_y[trn_y_inds]
#     trn_preds = softmax(model.predict(trn_x), axis=1)
#     tst_preds = softmax(model.predict(tst_x), axis=1)
    
#     trn_loss = np.multiply(trn_preds, trn_y).sum(axis=1)
#     tst_loss = np.multiply(tst_preds, tst_y).sum(axis=1)
    
#     trn_loss_mean = trn_loss.mean()
#     trn_thresh = (trn_preds >= trn_loss_mean).sum()
#     tst_thresh = tst_y_len - (tst_preds >= trn_loss_mean).sum()
#     acc = (trn_thresh + tst_thresh) / tst_y_len
#     print(acc)
#     return np.log(acc)

# def backdoor(h5name, bkd_x, bkd_y, subtract=False):
#     model = tf.keras.models.load_model(os.path.join(save_dir, h5name))
#     predsw = model.predict(bkd_x)
#     predswo = model.predict(np.zeros_like(bkd_x))
#     if subtract:
#         diff = predsw - predswo
#     else:
#         diff = predsw
#     pred = np.multiply(bkd_y, diff).sum()
#     print(pred)
#     return pred

# def build_model_nlp_init(dataset, modelname):
#   if modelname == "b":
#     model_name_or_path = "bert-base-uncased"
#   elif modelname == "r":
#     model_name_or_path = "roberta-base"
#   else:
#     model_name_or_path = modelname
  
#   config = AutoConfig.from_pretrained(
#       model_name_or_path,
#       num_labels=2,
#       finetuning_task="sst-2" if dataset.startswith('sst2') else "qnli",
#       cache_dir=CACHE_DIR,
#   )

#   model = AutoModelForSequenceClassification.from_pretrained(
#       model_name_or_path,
#       from_tf=bool(".ckpt" in model_name_or_path),
#       config=config,
#       cache_dir=CACHE_DIR,
#   )
  
#   return config, model


# def build_model_nlp(config, tokenizer, model, modelpath):  
#   if config.model_type == 'bert':
#     model.resize_token_embeddings(len(tokenizer))
#     resize_token_type_embeddings(model, new_num_types=10, random_segment=False)
  
#   weights = load_file(modelpath)
#   model.load_state_dict(weights)
  
#   return model




# def preprocess_nlp(x, y, dataset):
#   if modelname == "b":
#     model_name_or_path = "bert-base-uncased"
#   elif modelname == "r":
#     model_name_or_path = "roberta-base"
#   else:
#     model_name_or_path = modelname
  
#   tokenizer = AutoTokenizer.from_pretrained(
#     model_name_or_path, use_fast=True,
#     additional_special_tokens=[],
#     cache_dir=CACHE_DIR,
#   )
  
#   if dataset.startswith("sst2"):
#     encoded_inputs = [
#       tokenizer(xi,
#       padding='max_length',
#       truncation=True,
#       return_tensors="pt",
#       max_length=60,
#     ) for xi in x]
#   elif dataset.startswith("qnli"):
#     encoded_inputs = [
#       tokenizer(xi[0], xi[1],
#       padding='max_length',
#       truncation=True,
#       return_tensors="pt",
#       max_length=60,
#     ) for xi in x
#     ]
#   for idx, item in enumerate(encoded_inputs):
#     encoded_inputs[idx]['input_ids'] = item['input_ids'][0]

#   if dataset.startswith("qnli"):
#     map_labels={"entailment": 0, "not_entailment": 1}
#     y = [map_labels[yi] for yi in y]
#     y = np.eye(2)[y]
  
#   return encoded_inputs, y, tokenizer


# def make_it_batch(x, y):
#     assert len(x) < 20
#     new_x = {key: [] for key in x[0]}
#     for xi in x:
#         for item in xi:
#             if item == "attention_mask":
#                 xi[item] = new_x[item].append(xi[item][0])
#             else:
#                 xi[item] = new_x[item].append(xi[item])
    
#     for x in new_x:
#         new_x[x] = torch.stack(new_x[x])
    
#     return new_x, [[i] for i in y]



# def backdoor_nlp(config, bkd_x, bkd_y, model, modelpath, subtract=False):
#     bkd_x, bkd_y, tokenizer = preprocess_nlp(bkd_x, bkd_y, dataset_auditing)
#     model = build_model_nlp(config, tokenizer, model, modelpath)
    
#     bkd_x, bkd_y = make_it_batch(bkd_x, bkd_y)
    
#     model.eval()
#     if "sst2b" in modelpath or "sst2_b" in modelpath or "qnli_b" in modelpath or "qnlib" in modelpath:
#         bkd_x.pop("token_type_ids")
#     try:
#         predsw = model(**bkd_x)
#     except:
#         try:
#             bkd_x.pop("token_type_ids")
#             predsw = model(**bkd_x)
#         except:
#             raise NotImplementedError
#     diff = predsw['logits'].detach().numpy()
#     pred = np.multiply(bkd_y, diff).sum()
#     print(pred)
#     return pred

# for val in cfg_map:
#     cfg_map[val] = sorted(cfg_map[val], key=lambda h: int(h.split('-')[-1].split(".")[0]))

# name = '-'.join([key, str(start), str(end), pois_ct, clip_norm, noise, init])
# print(name, len(cfg_map[cfg_key]))
# alls = []
# mis = []

# old_bkd = auditing_args.args["old_bkd"]
# subtract = old_bkd

# if old_bkd:
#     subtract = False

# config, model = build_model_nlp_init(dataset_auditing, modelname)

# for jdx in [1,2,4,8]:
#     if cfg_map[cfg_key] == []:
#         print(cfg_key)
#         raise NotImplementedError
#     for h5 in cfg_map[cfg_key][start:end]:
#         modelpath = os.path.join(save_dir, h5)
#         x, y = all_bkds[jdx]['p'] if pois_ct == "." else all_bkds['p']
#         if get_mi:
#             mis.append(mi(h5))
#         if "fmnist" in dataset_auditing or "p100" in dataset_auditing:
#             nob_vals = backdoor(h5, x,  y, subtract=subtract)
#         else:
#             nob_vals = backdoor_nlp(config, x, y, model, modelpath, subtract=subtract)
#         alls.append(nob_vals)

#     if get_mi:
#         print("mi:", np.mean(mis))

#     if pois_ct != ".":
#         np.save(os.path.join(res_dir, '-'.join(["batch", name])), np.array(alls))
#         print("{} saved! ".format(os.path.join(res_dir, '-'.join(["batch", name]))))
#         break
#     else:
#         np.save(os.path.join(res_dir, '-'.join(["batch", name, f"@{jdx}"])), np.array(alls))
#         print("{} saved! ".format(os.path.join(res_dir, '-'.join(["batch", name, f"@{jdx}"]))))