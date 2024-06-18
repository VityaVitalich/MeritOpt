from tqdm import tqdm
import numpy as np
import os
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


class EmptyCacheTimeoutError(Exception):
    pass


def empty_cache_with_timeout(timeout):
    def empty_cache():
        torch.cuda.empty_cache()

    process = multiprocessing.Process(target=empty_cache)

    try:
        process.start()
        process.join(timeout)
    except multiprocessing.TimeoutError:
        process.terminate()
        raise EmptyCacheTimeoutError(
            "torch.cuda.empty_cache() took too long to execute."
        )
    else:
        process.terminate()

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
    crit,
    logger,
    config,
    epoch,
):

    model.train()
    pbar = tqdm(enumerate(train_loader), total=len(train_loader))
    for batch_idx, batch in pbar:

        st = logger.get_step() + 1
        logger.set_step(step=st, mode="train")

        model_input = dict_to_device(batch['model_input'])
        output = model.forward(**model_input)

        optimizer.zero_grad()
        loss = output["loss"]

        if loss.dim() != 0:
            loss = loss.mean()
        loss.backward()
        optimizer.step()
        scheduler.step()

        logger.add_scalar("loss", loss.item())
        pbar.set_postfix({"Loss": loss.item()})

        if (st % config.save_every == 0) and (config.save_strategy == 'steps'):
            save_model(model, tokenizer, config, iteration=st)

        if st >= config.max_steps:
            break


    return None
    # return loss ...


def train_epoch_fl(
    model,
    tokenizer,
    optimizer,
    scheduler,
    train_loaders,
    val_loader,
    crit,
    logger,
    config,
    epoch,
):

    model.train()
    train_loader_iterator = [iter(loader) for loader in train_loaders]
    ### how to deal with imbalanceness ###
    nsteps = min([len(i) for i in train_loaders])

    pbar = tqdm(range(nsteps), total=nsteps)
    for i in pbar:
        overloader_loss = 0

        st = logger.get_step() + 1
        logger.set_step(step=st, mode="train")

        for w_id,  iteraror in enumerate(train_loader_iterator):
            accum_iter = config.acc_steps[w_id]
            accum_loss = 0
            for step in range(accum_iter):
                batch = next(iteraror)
                model_input = dict_to_device(batch['model_input'])
                output = model.forward(**model_input)

                loss = output["loss"]
                if loss.dim() != 0:
                    loss = loss.mean()
                # normalize loss to account for batch accumulation
                loss = loss / accum_iter
                loss.backward()
                accum_loss += loss.item()

            overloader_loss += accum_loss

            optimizer.step(w_id, model, crit, val_loader)
            optimizer.zero_grad()  
          
        scheduler.step()
        overloader_loss = overloader_loss / len(train_loader_iterator)
        logger.add_scalar("loss", overloader_loss)
        pbar.set_postfix({"Loss": overloader_loss})

        weights = optimizer.metrics()
        logger.add_dict(weights)
        if (st % config.save_every == 0) and (config.save_strategy == 'steps'):
            save_model(model, tokenizer, config, iteration=st)

        if st >= config.max_steps:
            break
    

    return None


@torch.inference_mode()
def validate(model, val_loader, logger, config):
    model.eval()

    mean_loss = 0
    for batch_idx, batch in tqdm(enumerate(val_loader)):

        with torch.no_grad():
            model_input = dict_to_device(batch['model_input'])
            output = model.forward(**model_input)
            loss = output["loss"]
            if loss.dim() != 0:
                loss = loss.mean()
            mean_loss += loss.item()
    

    mean_loss = mean_loss / (batch_idx + 1)
    logger.add_scalar("Val_loss", mean_loss)


    return mean_loss

@torch.inference_mode()
def predict(model, tokenizer, val_loader, config, epoch=""):
    model.eval()

    all_preds = []
    all_labels = []

    saving_path = (
        config.saving_predictions_path + "_" + config.exp_name + "_" + str(epoch)
    )

    evalbar = tqdm(enumerate(val_loader), total=len(val_loader), desc="eval going")
    for batch_idx, batch in evalbar:

        pred, gold = get_one_sample(model, tokenizer, batch, config)

        all_preds.extend(pred)
        all_labels.extend(gold)

    return all_preds, all_labels


@torch.inference_mode()
def get_one_sample(model, tokenizer, batch, config) -> Tuple[List[str], List[str]]:
    model.eval()
    
    if isinstance(model, torch.nn.DataParallel):
        generated_tokens = model.module.generate(
            inputs=batch['model_input']['input_ids'].to('cuda'),
            attention_mask=batch['model_input']['attention_mask'].to('cuda'),
            pad_token_id=tokenizer.eos_token_id,
            **config.gen_args,
        )
    else:
        generated_tokens = model.generate(
            inputs=batch['model_input']['input_ids'].to('cuda'),
            attention_mask=batch['model_input']['attention_mask'].to('cuda'),
            pad_token_id=tokenizer.eos_token_id,
            **config.gen_args,
        )
    decoded = tokenizer.batch_decode(generated_tokens.cpu(), skip_special_tokens=True)
    refs = [elem['output'] for elem in batch['info']]

    return decoded, refs


def save_model(model, tokenizer, config, epoch=None, iteration=None):
    if not os.path.exists(config.saving_path):
        os.makedirs(config.saving_path)

    if epoch is not None:
        ckpt_name = f"{config.saving_path}/checkpoint_{epoch}/"
    elif iteration is not None:
        ckpt_name = f"{config.saving_path}/checkpoint_{iteration}/"
    else:
        raise ValueError('no iter or epoch on save')

    if isinstance(model, torch.nn.DataParallel):
        model.module.save_pretrained(ckpt_name)
    else:
        model.save_pretrained(ckpt_name)

    tokenizer.save_pretrained(ckpt_name)
    print(f"saved")
