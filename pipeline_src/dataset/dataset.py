import torch
import torch.nn as nn
from tqdm import tqdm_notebook as tqdm
from multiprocessing import cpu_count
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset, load_from_disk
import os
import math
from functools import partial
from transformers import M2M100Tokenizer

def extract_tokens_m2m100(elem, tokenizer, m100_mapping, max_seq_len):
    tokenizer.src_lang = m100_mapping[elem['language']]
    src = tokenizer(elem['input'], return_tensors='pt', truncation=True, max_length=max_seq_len)

    tokenizer.src_lang = 'en'
    trg = tokenizer(elem['output'], return_tensors='pt', truncation=True, max_length=max_seq_len)

    labels = torch.clone(trg['input_ids'])
    labels[:,0] = -100

    model_input = {
        'input_ids': src['input_ids'].squeeze(0),
        'labels': labels.squeeze(0)}

    elem['len'] = len(src['input_ids'].squeeze(0))

    elem['model_input'] = model_input

    elem['info'] = {
        'input': elem.pop('input'),
        'output': elem.pop('output'),
        'lang': elem.pop('language')
    }
    return elem
    

def extract_tokens_seq2seq(elem, tokenizer, m100_mapping, max_seq_len):
    src = tokenizer(elem['input'], return_tensors='pt', truncation=True, max_length=max_seq_len)
    trg = tokenizer(elem['output'], return_tensors='pt', truncation=True, max_length=max_seq_len)

    labels = torch.clone(trg['input_ids'])
    labels[:,0] = -100

    model_input = {
        'input_ids': src['input_ids'].squeeze(0),
        'labels': labels.squeeze(0)}
    elem['len'] = len(src['input_ids'].squeeze(0))
    elem['model_input'] = model_input

    elem['info'] = {
        'input': elem.pop('input'),
        'output': elem.pop('output'),
        'lang': elem.pop('language')
    }
    return elem


def extract_tokens_LM(elem, tokenizer, max_seq_len, system_prompt, **kwargs):
    system_prompt = tokenizer(system_prompt, return_tensors='pt', truncation=True, max_length=max_seq_len)
    src = tokenizer(f'Language: {elem["language"]}. {elem["input"]}[/INST]', return_tensors='pt', truncation=True, max_length=max_seq_len)
    trg = tokenizer(elem['output'], return_tensors='pt', truncation=True, max_length=max_seq_len)
    
    input_seq = torch.cat([system_prompt['input_ids'], src['input_ids'], trg['input_ids']], dim=1)
    labels = torch.clone(input_seq)

    len_instruct = system_prompt['input_ids'].size(1) + src['input_ids'].size(1)
    labels[:,:len_instruct] = -100

    model_input = {
        'input_ids': input_seq.squeeze(0),
        'labels': labels.squeeze(0)}
    elem['len'] = len(input_seq.squeeze(0))
    elem['model_input'] = model_input

    elem['info'] = {
        'input': elem.pop('input'),
        'output': elem.pop('output'),
        'lang': elem.pop('language')
    }
    return elem

class MTDataset(Dataset):
    def __init__(
        self, 
        dataset, 
        tokenizer,
        config,
        m100_mapping={
            'jav': 'jv',
            'ind': 'id',
            'tam': 'ta',
            'mal': 'ms',
            'tgl': 'tl',
            'hun': 'hu',
            'sma': 'sma',
            'sms': 'sms',
            'smn': 'smn',
            'sme': 'sme'
        },
        max_seq_len=1024
    ):  
        if isinstance(tokenizer, M2M100Tokenizer):
            extract_func = extract_tokens_m2m100
        elif config.model_type == 'AutoLM':
            system_prompt = """<s>[INST] <<SYS>> You are a translation model.
            Your task is to translate the given text from the source language to the English language.
            Provide only the translated text with no additional comments, explanations, or modifications. <</SYS>>"""

            extract_func = partial(extract_tokens_LM, system_prompt=system_prompt)
        else:
            extract_func = extract_tokens_seq2seq

        self.data = dataset.map(partial(extract_func, tokenizer=tokenizer, m100_mapping=m100_mapping, max_seq_len=max_seq_len))
        #self.data = self.data.sort('len', reverse=True)
        self.tokenizer = tokenizer
        self.m100_mapping = m100_mapping

    def __getitem__(self, index):
        elem = self.data[index]

        return elem

    def __len__(self):
        return len(self.data)
        

class Collator:
    def __init__(self, pad_token_id, mask_token_id, padding_side):
        self.pad_token_id = pad_token_id
        self.mask_token_id = mask_token_id
        self.padding_side = padding_side
    def __call__(self, batch):
        info = []
        
        input_ids = []
        dec_ids = []
        labels = []

        for elem in batch:
            info.append(elem['info'])
            model_input = elem['model_input']

            input_ids.append(torch.tensor(model_input['input_ids']))
            labels.append(torch.tensor(model_input['labels']))

        padding_func = self.right_padding if self.padding_side == 'right' else self.left_padding
        input_ids, labels = padding_func(input_ids, labels)

        att_mask_inputs = torch.zeros_like(input_ids)
        att_mask_inputs[input_ids != self.pad_token_id] = 1

        model_input = {
        'input_ids': input_ids,
        'attention_mask': att_mask_inputs,
        'labels': labels}

        return {'model_input': model_input, 'info': info}

    def right_padding(self, input_ids, labels):
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=self.mask_token_id
        )
        return input_ids, labels

    def left_padding(self, input_ids, labels):
        # Find the maximum sequence length
        max_length_input_ids = max(len(ids) for ids in input_ids)
        max_length_labels = max(len(lbl) for lbl in labels)

        # Pad sequences on the left
        input_ids = [self.pad_left(seq, max_length_input_ids, self.pad_token_id) for seq in input_ids]
        labels = [self.pad_left(seq, max_length_labels, self.mask_token_id) for seq in labels]

        input_ids = torch.stack(input_ids)
        labels = torch.stack(labels)

        return input_ids, labels

    def pad_left(self, seq, max_length, pad_value):
        """Pad sequence on the left to the max length with the given pad value."""
        padding_length = max_length - len(seq)
        return torch.cat([torch.full((padding_length,), pad_value), seq])


def init_data(tokenizer, config, mask_label_token=-100, token=None):
    if os.path.exists(config.data_path):
        ds = load_from_disk(config.data_path)
    else:
        ds = load_dataset(config.data_path, token=token)


    # data

    val_dataset = MTDataset(
        dataset=ds['val'],
        tokenizer=tokenizer,
        config=config,
        max_seq_len=config.max_seq_len
    )
    test_dataset = MTDataset(
        dataset=ds['test'],
        tokenizer=tokenizer,
        config=config
    )

    if config.model_type == 'AutoLM':
        padding_side = 'left'
    else:
        padding_side = 'right'
    collator = Collator(
        pad_token_id=tokenizer.pad_token_id, mask_token_id=mask_label_token, padding_side=padding_side
    )

    train_dataset, train_loader, val_loader, test_loader, weight_name_map = init_loaders(ds, val_dataset, test_dataset, collator, config, tokenizer)



    return train_dataset, val_dataset, test_dataset, train_loader, val_loader, test_loader, weight_name_map

def init_loaders(ds, val_dataset, test_dataset, collator, config, tokenizer):

    num_workers = min(16, cpu_count())

    if config.fl:
        loader_func = init_loaders_fl
    else:
        loader_func = init_loaders_baseline 
        
    return loader_func(ds, val_dataset, test_dataset, collator, config, num_workers, tokenizer)

def init_loaders_fl(ds, val_dataset, test_dataset, collator, config, num_workers, tokenizer):


    all_langs = set(ds['train']['language'])
    train_hf_datasets = [ds['train'].filter(lambda elem: elem['language'] == lang)
            for lang in all_langs]

    train_datasets = [
        MTDataset(
            dataset=lang_ds,
            tokenizer=tokenizer,
            config=config,
            max_seq_len=config.max_seq_len
        )
        for lang_ds in train_hf_datasets]
    
    if config.adaptive_batch_size:
        batch_sizes = []
        acc_steps = []
        for lang_ds in train_hf_datasets:
            cur_portion = len(lang_ds) / len(ds['train'])
            cur_batch_size = int(config.total_batch_size * cur_portion)
            cur_batch_size = max(config.min_batch_size, cur_batch_size)
            cur_batch_size = min(config.max_batch_size, cur_batch_size)
            # setup accumulation if batch too big
            acc_steps.append(math.ceil(cur_batch_size / config.batch_size))
            # config.batch size if cur_batch is bigger
            real_batch_size = min(config.batch_size, cur_batch_size)
            batch_sizes.append(real_batch_size)
    else:
        batch_sizes = [config.batch_size] * len(train_hf_datasets)
        acc_steps = [1] * len(train_hf_datasets)
    config.acc_steps = acc_steps
    print(f"Batch sizes {batch_sizes}, Acc Steps: {acc_steps}")
    train_loader = [
        DataLoader(
            lang_dataset,
            batch_size=batch_size,
            collate_fn=collator,
            shuffle=True,
            num_workers=num_workers,
            drop_last=True,
            pin_memory=True,
        ) for lang_dataset, batch_size in zip(train_datasets, batch_sizes)
    ]

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        collate_fn=collator,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        collate_fn=collator,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
        pin_memory=True,
    )
    weight_name_map = {i: cur_lang for i, cur_lang in enumerate(all_langs)}
    return train_datasets, train_loader, val_loader, test_loader, weight_name_map

def init_loaders_baseline(ds, val_dataset, test_dataset, collator, config, num_workers, tokenizer):

    train_dataset = MTDataset(
        dataset=ds['train'],
        tokenizer=tokenizer,
        config=config,
        max_seq_len=config.max_seq_len
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        collate_fn=collator,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        collate_fn=collator,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        collate_fn=collator,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
        pin_memory=True,
    )

    return train_dataset, train_loader, val_loader, test_loader, None

