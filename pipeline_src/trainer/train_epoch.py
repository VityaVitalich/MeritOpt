from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import os
from copy import deepcopy
from torch import nn
import torch.nn.functional as F
import torch
import gc
import wandb
import json
import itertools
from collections import Counter
import pickle
import pandas as pd
import multiprocessing
import time
from typing import Tuple, List
from accelerate.utils import gather_object


def dict_to_device(d, device='cuda'):
    out = {}
    for k, val in d.items():
        if isinstance(val, dict):
            out[k] = dict_to_device(val)
        else:
            out[k] = val.to(device)

    return out


def train_epoch(*args, **kwargs):

    if kwargs['config'].fl:
        return train_epoch_fl(*args, **kwargs)
    else:
        return train_epoch_base(*args, **kwargs)

def train_epoch_base(
    model,
    tokenizer,
    optimizer,
    scheduler,
    train_loader,
    val_batch,
    logger,
    accelerator,
    config,
    epoch,
):

    model.train()
    pbar = tqdm(enumerate(train_loader), total=len(train_loader), disable=not accelerator.is_local_main_process)
    for batch_idx, batch in pbar:
        st = logger.get_step()
        logger.set_step(step=st+1, mode="train")

        model_input = dict_to_device(batch['model_input'], device=config.device)
        model_input.pop('src')
        model_input.pop('src_att_mask')
        output = model.forward(**model_input)
        optimizer.zero_grad()
        loss = output["loss"]

        #loss.backward()
        accelerator.backward(loss)
        optimizer.step()
        scheduler.step()

        if accelerator.sync_gradients:
            logger.add_scalar("loss", loss)

        if (st % config.save_every == 0) and (config.save_strategy == 'steps'):
            save_model(model, tokenizer, config, iteration=st, accelerator=accelerator)

        if st >= config.max_steps:
            break


def prepare_fl_loaders(train_loaders, optimizer, epoch, config):

    if config.drop_threshold > 0:
        weights = optimizer.optimizer.weights
        train_loader_iterator = {w_id: iter(loader) for w_id, loader in train_loaders.items() if weights[w_id] > config.drop_threshold}
        optimizer.optimizer.drop_weights()
        nsteps = min([len(loader) for w_id, loader in train_loaders.items() if weights[w_id] > config.drop_threshold])
    else:
        if config.enable_fl_every > 0:
            if (epoch % config.enable_fl_every == 0):
                print('enabled fl')
                optimizer.optimizer.enabled_fl = True
                train_loader_iterator = {w_id: iter(loader) for w_id, loader in train_loaders.items()}
                optimizer.optimizer.active_weight_ids = torch.ones_like(optimizer.optimizer.active_weight_ids)
                if epoch > 0:
                    optimizer.optimizer.weights = deepcopy(optimizer.optimizer.prev_weights)
                nsteps = min([len(loader) for w_id, loader in train_loaders.items()])
            else:
                optimizer.optimizer.enabled_fl = False
                weights = optimizer.optimizer.weights
                argmax_weight = torch.argmax(weights)
                train_loader_iterator = {w_id: iter(loader) for w_id, loader in train_loaders.items() if w_id == argmax_weight}
                optimizer.optimizer.active_weight_ids = torch.zeros_like(optimizer.optimizer.active_weight_ids)
                optimizer.optimizer.active_weight_ids[argmax_weight] = True

                #### WE ALSO NEED TO MAKE WEIGHT OF ACTIVE WEIGHT TO 1 !!!
                optimizer.optimizer.prev_weights = deepcopy(weights)
                optimizer.optimizer.weights = torch.zeros_like(weights)
                optimizer.optimizer.weights[argmax_weight] = 1

                nsteps = min([len(loader) for w_id, loader in train_loaders.items() if w_id == argmax_weight])
        else:
            train_loader_iterator = {w_id: iter(loader) for w_id, loader in train_loaders.items()}
            nsteps = min([len(loader) for w_id, loader in train_loaders.items()])

    return train_loader_iterator, nsteps

def train_epoch_fl(
    model,
    tokenizer,
    optimizer,
    scheduler,
    train_loaders,
    val_loader,
    logger,
    accelerator,
    config,
    epoch,
):

    model.train()
    train_loader_iterator, nsteps = prepare_fl_loaders(train_loaders, optimizer, epoch, config)

    pbar = tqdm(range(nsteps), total=nsteps, disable=not accelerator.is_local_main_process)
    for i in pbar:
        overloader_loss = torch.tensor(0., device=accelerator.device)
        
        st = logger.get_step() + 1
        logger.set_step(step=st, mode="train")
        for i, (w_id,  iteraror) in enumerate(train_loader_iterator.items()):
            batch = next(iteraror)

            model_input = dict_to_device(batch['model_input'], device=accelerator.device)
            model_input.pop('src')
            model_input.pop('src_att_mask')
            output = model.forward(**model_input)
            loss = output["loss"]

            accelerator.backward(loss)
            overloader_loss += loss.item()

            # sync grad before registering into optimizer
            if accelerator.sync_gradients:
                optimizer.optimizer.register_worker_grad(w_id)
            if i < len(train_loader_iterator) - 1: 
                optimizer.zero_grad()  

        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()
        if accelerator.sync_gradients:
            overloader_loss = accelerator.reduce(overloader_loss, reduction='mean')
            overloader_loss = overloader_loss / len(train_loader_iterator)
            logger.add_scalar("loss", overloader_loss)


        weights = optimizer.optimizer.metrics()
        logger.add_dict(weights)
        if (st % config.save_every == 0) and (config.save_strategy == 'steps'):
            save_model(model, tokenizer, config, iteration=st, accelerator=accelerator)

        if st >= config.max_steps:
            break


@torch.inference_mode()
def validate(model, val_loader, logger, accelerator, config):
    model.eval()

    mean_loss = 0
    for batch_idx, batch in tqdm(enumerate(val_loader), desc="eval going", disable=not accelerator.is_local_main_process):

        with torch.no_grad():
            model_input = dict_to_device(batch['model_input'], device=config.device)
            model_input.pop('src')
            model_input.pop('src_att_mask')
            output = model.forward(**model_input)
            loss = output["loss"]

            if accelerator.sync_gradients:
                mean_loss += loss.item()
    

    mean_loss = mean_loss / (batch_idx + 1)
    logger.add_scalar("Val_loss", mean_loss)

    return mean_loss

@torch.inference_mode()
def predict(model, tokenizer, val_loader, config, accelerator, epoch=""):
    model.eval()

    all_preds = []
    all_labels = []

    #output_batches = []
    for lang_loader in val_loader:
        evalbar = tqdm(enumerate(lang_loader), total=len(lang_loader), desc="prediction going", disable=not accelerator.is_local_main_process)
        for batch_idx, batch in evalbar:
            
            if config.model_type == 'M2M100':
                target_lang = batch['info'][0]['elem_target_lang']
                config.gen_args['forced_bos_token_id'] = tokenizer.get_lang_id(target_lang)

            pred, gold = get_one_sample(model, tokenizer, batch, config, accelerator=accelerator)
            predictions = gather_object(pred)
            references = gather_object(gold)

            all_preds.extend(predictions)
            all_labels.extend(references)

    return all_preds, all_labels


@torch.inference_mode()
def get_one_sample(model, tokenizer, batch, config, accelerator) -> Tuple[List[str], List[str]]:
    model.eval()
    print(model) 
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        generated_tokens = accelerator.unwrap_model(model).generate(
            inputs=batch['model_input']['src'].to(config.device),
            attention_mask=batch['model_input']['src_att_mask'].to(config.device),
            pad_token_id=tokenizer.pad_token_id,
            **config.gen_args,
        )
    else:
        generated_tokens = model.generate(
            inputs=batch['src']['input_ids'].to(config.device),
            attention_mask=batch['model_input']['src_att_mask'].to(config.device),
            pad_token_id=tokenizer.pad_token_id,
            **config.gen_args,
        )

    if config.model_type == 'AutoLM':
        src_len = batch['model_input']['src'].size(1)
        generated_tokens = generated_tokens[:,src_len:]

    decoded = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
    refs = [elem['output'] for elem in batch['info']]

    return decoded, refs


def save_model(model, tokenizer, config, epoch=None, iteration=None, accelerator=None):
    if not os.path.exists(config.saving_path):
        os.makedirs(config.saving_path)

    if epoch is not None:
        ckpt_name = f"{config.saving_path}/checkpoint_{epoch}/"
    elif iteration is not None:
        ckpt_name = f"{config.saving_path}/checkpoint_{iteration}/"
    else:
        raise ValueError('no iter or epoch on save')

    if accelerator:
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(
            ckpt_name,
            is_main_process=accelerator.is_main_process,
            save_function=accelerator.save,
        )
    else:
        if isinstance(model, torch.nn.DataParallel):
            model.module.save_pretrained(ckpt_name)
        else:
            model.save_pretrained(ckpt_name)

    tokenizer.save_pretrained(ckpt_name)
    if accelerator.is_main_process:
        print("saved")
