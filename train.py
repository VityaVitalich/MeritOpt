import os
import yaml

with open(r"./configs/train.yml") as file:
    params_list = yaml.load(file, Loader=yaml.FullLoader)

SAVING_DIR = os.environ.get("SAVING_DIR")
HF_TOKEN = os.environ.get("HF_TOKEN")
os.environ["TRANSFORMERS_CACHE"] = SAVING_DIR + "hf_cache/"
os.environ["HF_HOME"] = SAVING_DIR + "hf_cache/"

os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(
    map(str, params_list["CUDA_VISIBLE_DEVICES"])
)



import sys
import torch
import pandas as pd
from torch import nn
import numpy as np
from torch.optim.lr_scheduler import ExponentialLR
import wandb
from dataclasses import dataclass, field

from accelerate import dispatch_model, infer_auto_device_map
from accelerate.utils import get_balanced_memory



from pipeline_src.train import train
from pipeline_src.logger.logger import WanDBWriter
from pipeline_src.trainer.train_epoch import train_epoch, predict
from pipeline_src.dataset.dataset import init_data
from pipeline_src.optimizers import MeritFedA

print(torch.cuda.is_available())
if torch.cuda.is_available():
    device = "cuda"
    print("GPU")
else:
    device = "cpu"
    print("CPU")


SEED = params_list["SEED"][0]
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)
print(torch.cuda.device_count())

from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    AutoConfig,
    AutoModelForCausalLM,
    LlamaTokenizer,
    LlamaForCausalLM,
    M2M100ForConditionalGeneration,
    M2M100Tokenizer

)

from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_kbit_training,
)

@dataclass
class TaskConfig:
    project_name: str = "MTFL"
    seed: int = SEED

if __name__ == "__main__":
    config = TaskConfig()

    config.n_epochs = params_list["EPOCHS"][0]
    config.batch_size = params_list["BATCH_SIZE"][0]
    config.lr = float(params_list["LR"][0])
    config.min_lr = float(params_list["MIN_LR"][0])
    config.validation = int(params_list["VAL_EVERY_EPOCH"][0])
    config.save_every = int(params_list["SAVE_EVERY"][0])
    config.max_seq_len = int(params_list["MAX_SEQ_LEN"][0])
    config.max_steps = int(params_list["MAX_STEPS"][0])
    config.data_path = params_list["DATA_PATH"][0]
    config.device = device
    config.model_type = params_list["MODEL_TYPE"][0] 
    config.wandb_log_dir = SAVING_DIR + "wandb/"
    config.model_checkpoint = params_list["MODEL_CHECKPOINT"][0]
    config.exp_name = (params_list["RUN_NAME"][0])
    config.save_strategy = params_list["SAVE_STRATEGY"][0]
    config.saving_path = SAVING_DIR + "model_checkpoints/" + config.exp_name + '_seed' + str(SEED)
    config.log_pred_every = params_list["LOG_PRED_EVERY"][0]
    config.target_lang = params_list["TARGET_LANG"][0]
    config.compute_metrics_every = params_list["COMPUTE_METRICS_EVERY"][0]
    config.saving_predictions_path = SAVING_DIR + "model_outputs/"
    config.fl = params_list["FL"][0]
    config.adaptive_batch_size = params_list["ADAPTIVE_BATCH_SIZE"][0]
    config.total_batch_size = params_list["TOTAL_BATCH_SIZE"][0]
    config.min_batch_size = params_list["MIN_BATCH_SIZE"][0]
    config.max_batch_size = params_list["MAX_BATCH_SIZE"][0]
    config.fl_lr = params_list["FL_LR"][0]
    config.fl_niters = params_list["FL_NITERS"][0]

    config.gen_args =  {
    "no_repeat_ngram_size": params_list["NO_REPEAT_NGRAM"][0],
    "do_sample": params_list["SAMPLING"][0],
    "max_new_tokens": 256,
    "temperature": params_list["TEMPERATURE"][0],
    "top_k": params_list["TOP_K"][0],
    "num_return_sequences": 1,
    "num_beams": params_list["NUM_BEAMS"][0],   
    }

    model_params = {}
    if config.model_type == "AutoLM":
        model_type = AutoModelForCausalLM
        tokenizer_type = AutoTokenizer
        model_params['device_map'] = 'auto'
    elif config.model_type == "M2M100":
        model_type = M2M100ForConditionalGeneration
        tokenizer_type = M2M100Tokenizer
        
    elif config.model_type == "Seq2Seq":
        model_type = AutoModelForSeq2SeqLM
        tokenizer_type = AutoTokenizer

    print(config.model_checkpoint)

    model = model_type.from_pretrained(
        config.model_checkpoint,
        use_auth_token=HF_TOKEN,
        torch_dtype=torch.bfloat16,
        **model_params
    )


    if params_list["DATA_PARALLEL"][0] and (config.model_type != "AutoLM"):
        model = torch.nn.DataParallel(model).to("cuda")
    else:
        model.to('cuda')


    tokenizer = tokenizer_type.from_pretrained(
        config.model_checkpoint,
        use_auth_token=HF_TOKEN,
    )
    if not tokenizer.pad_token_id:
        tokenizer.pad_token = tokenizer.eos_token
    if 'flores' in config.model_checkpoint:
        tokenizer.lang_token_to_id = {t: i for t, i in zip(tokenizer.all_special_tokens, tokenizer.all_special_ids) if i > 5}
        tokenizer.lang_code_to_token = {s.strip("_"): s for s in tokenizer.lang_token_to_id}
        tokenizer.lang_code_to_id = {s.strip("_"): i for s, i in tokenizer.lang_token_to_id.items()}
        tokenizer.id_to_lang_token = {i: s for s, i in tokenizer.lang_token_to_id.items()}


    if any(name in config.data_path for name in ['sma', 'sms', 'smn', 'sme']): 
        embedding_size = model.get_input_embeddings().weight.shape[0]
        new_langs = ['sma', 'sms', 'smn', 'sme']
        for lang, new_lang_id in zip(new_langs, range(embedding_size, embedding_size + len(new_langs))):
            tokenizer.lang_token_to_id[f'__{lang}__'] = new_lang_id
            tokenizer.lang_code_to_id[lang] = new_lang_id
            tokenizer.lang_code_to_token[lang] = f'__{lang}__'
            tokenizer.id_to_lang_token[new_lang_id] = f'__{lang}__'
            tokenizer.added_tokens_encoder[f'__{lang}__'] = new_lang_id
            tokenizer.additional_special_tokens.append(f'__{lang}__')
        
        config.target_lang = 'fi'
        model.resize_token_embeddings( embedding_size + len(new_langs))

    if config.model_type == 'M2M100':
        config.gen_args['forced_bos_token_id'] = tokenizer.get_lang_id(config.target_lang)


    train_dataset, val_dataset, test_dataset, train_loader, val_loader, test_loader, weight_name_map = init_data(tokenizer, config)
    # Setup max steps and scheduler steps
    if config.fl:
        assert config.max_steps > 0, "FL only works with max steps"
        steps_per_loader = min([len(i) for i in train_loader])
        scheduler_steps = config.max_steps
        config.n_epochs = (config.max_steps // steps_per_loader) + 1

    else:
        if config.max_steps == -1:
            config.max_steps = float('inf')
            scheduler_steps = len(train_loader) * config.n_epochs
        else:
            config.n_epochs = (config.max_steps // len(train_loader)) + 1
            scheduler_steps = config.max_steps


    if config.fl:
        config.npeers = len(train_loader)
        config.mdlr_ = config.fl_lr
        config.mdniters_ = config.fl_niters

        optimizer = MeritFedA(
            model.parameters(), config, device, weight_name_map
        )
    else:
        optimizer = torch.optim.Adam(
            model.parameters(), lr=config.lr, betas=(0.9, 0.98), eps=1e-9
        )

    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=scheduler_steps, eta_min=config.min_lr
    )

    logger = WanDBWriter(config)

    train(
        model,
        tokenizer,
        train_loader,
        val_loader,
        test_loader,
        optimizer,
        scheduler,
        criterion,
        logger,
        config,
    )
