#!/usr/bin/env python

import os
os.environ['TOKENIZERS_PARALLELISM'] = '0'

import pickle
import transformers
import tokenizers
import datasets
import torch
import numpy as np
import subprocess
from tqdm.auto import tqdm
import tempfile
import random
import urllib.request
from typing import NamedTuple
import functools
import argparse
import contextlib
import enum
import threading
from collections import defaultdict
from functools import lru_cache, _make_key

def threadsafe_lru(func):
    func = lru_cache()(func)
    lock_dict = defaultdict(threading.Lock)

    def _thread_lru(*args, **kwargs):
        key = _make_key(args, kwargs, typed=False)
        with lock_dict[key]:
            return func(*args, **kwargs)

    return _thread_lru

DATA_DIR = os.path.join(os.path.expanduser('~'), '.cache', 'transformer-surrogates')
BLOCKS_URL = 'https://storage.googleapis.com/renda-transformer-surrogates/mca-blocks'
DATA_FILE = os.path.join(DATA_DIR, 'blocks')

os.makedirs(DATA_DIR, exist_ok=True)
if not os.path.exists(DATA_FILE):
    urllib.request.urlretrieve(BLOCKS_URL, DATA_FILE)


@threadsafe_lru
def blocks():
    with open(DATA_FILE, 'rb') as f:
        return pickle.load(f)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class BertForMAPE(transformers.BertForSequenceClassification):
    def __init__(self, config):
        super().__init__(config)
        assert self.num_labels == 1
        self.config.problem_type = "regression"

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            labs = labels.view(-1, 1)
            loss = ((logits.view(-1, self.num_labels) - labs).abs() / labs).mean()

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return transformers.modeling_outputs.SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


def tokenizer_name():
    return os.path.join(DATA_DIR, 'tokenizer.json')

_special_tokens = ['[PAD]']

def make_tokenizer():
    with open('/tmp/intel_code', 'w') as f:
        f.write('\n'.join(blocks()['code'].values))

    tokenizer = tokenizers.Tokenizer(tokenizers.models.WordPiece())
    tokenizer.pre_tokenizer = tokenizers.pre_tokenizers.BertPreTokenizer()
    tokenizer.normalizer = tokenizers.normalizers.Sequence([tokenizers.normalizers.NFD(), tokenizers.normalizers.Lowercase(), tokenizers.normalizers.StripAccents()])
    tokenizer.add_special_tokens(_special_tokens)
    trainer = tokenizers.trainers.WordPieceTrainer(special_tokens=_special_tokens)
    tokenizer.train(['/tmp/intel_code'], trainer)

    tokenizer.save(tokenizer_name())

@threadsafe_lru
def load_tokenizer():
    fname = tokenizer_name()
    if not os.path.exists(fname):
        make_tokenizer()
    return transformers.PreTrainedTokenizerFast(tokenizer_file=fname, pad_token='[PAD]', additional_special_tokens=_special_tokens)

@threadsafe_lru
def load_dataset(max_length, real):
    savename = os.path.join(DATA_DIR, 'dataset-{}-{}'.format(max_length, real))
    if os.path.exists(savename):
        with open(savename, 'rb') as f:
            return pickle.load(f)

    tcol = 'mca'
    if real:
        tcol = 'hsw-true'

    ds = datasets.Dataset.from_pandas(blocks()[['code', tcol]])
    ds = ds.rename_column(tcol, 'label')

    def do_normalize(row):
        row['label'] /= 10000
        return row
    ds = ds.map(do_normalize)

    fast_tokenizer = load_tokenizer()

    def do_tokenize(row):
        res = fast_tokenizer(
            row['code'],
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors='np',
        )
        res['attention_mask'][res['input_ids'] == 0] = 0
        return res

    ds = ds.map(do_tokenize, batched=True, batch_size=64)
    ds = ds.remove_columns(['code', 'idx'])

    n_test = int(len(ds) * 0.2)
    n_val = int(len(ds) * 0.1)

    generator = np.random.default_rng(0)
    permutation = generator.permutation(len(ds))
    test_indices = permutation[:n_test]
    val_indices = permutation[n_test:n_test+n_val]
    train_indices = permutation[n_test+n_val:]

    ds = datasets.DatasetDict({
        "train": ds.select(train_indices),
        "test": ds.select(test_indices),
        "val": ds.select(val_indices),
    })
    ds.set_format("torch")

    with open(savename, 'wb') as f:
        pickle.dump(ds, f)

    return ds

def get_data_fractions():
    return np.exp(np.linspace(0, 1, 10) * (np.log(215729) - np.log(100)) + np.log(100)) / 215729

def create_bert(hidden_size=64, hidden_layers=2, attention_heads=2, dropout=0, max_position=64):
    fast_tokenizer = load_tokenizer()
    return BertForMAPE(transformers.BertConfig(
        vocab_size=fast_tokenizer.vocab_size,
        hidden_size=hidden_size,
        num_hidden_layers=hidden_layers,
        num_attention_heads=attention_heads,
        intermediate_size=hidden_size*4,
        hidden_dropout_prob=dropout,
        attention_probs_dropout_prob=dropout,
        max_position_embeddings=max_position,
        num_labels=1,
    )).to(device, non_blocking=True)

def load_bert(fname):
    return BertForMAPE.from_pretrained(os.path.abspath(fname)).to(device, non_blocking=True)

def train_bert(bert, epochs, fname, max_length, data_seed=None, real_data=False, data_fraction=None):
    ds = load_dataset(max_length, real_data)

    trds = ds['train']
    if data_fraction is not None:
        trds = trds.shuffle(data_seed).select(range(int(data_fraction * len(trds))))

    train_dataloader = torch.utils.data.DataLoader(trds, shuffle=True, batch_size=64)
    num_training_steps = epochs * len(train_dataloader)
    progress_bar = tqdm(range(num_training_steps))

    loss_ema = 0
    ema_beta = 0.99
    idx = 0

    optimizer = transformers.AdamW(bert.parameters(), lr=1e-4)
    lr_scheduler = transformers.get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=1000,
        num_training_steps=num_training_steps
    )

    bert.train()
    for epoch in range(epochs):
        for batch in train_dataloader:
            input_ids = batch['input_ids'].to(device, non_blocking=True)
            token_type_ids = batch['token_type_ids'].to(device, non_blocking=True)
            attention_mask = batch['attention_mask'].to(device, non_blocking=True)
            labels = batch['label'].to(device, non_blocking=True)

            outputs = bert(
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                input_ids=input_ids,
                labels=labels,
            )
            loss = outputs.loss.mean()
            loss.backward()

            loss_ema = loss_ema * ema_beta + loss.item() * (1 - ema_beta)

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)
            idx += 1
            progress_bar.set_description('loss={:.3f}'.format(loss_ema / (1 - ema_beta ** idx)))

        savename = fname + '-epoch-{}'.format(epoch)
        bert.save_pretrained(savename)
        with open(os.path.join(savename, 'train_loss'), 'w') as f:
            f.write('{}\n'.format(loss_ema / (1 - ema_beta ** idx)))


        def do_test(savename, typ, real):
            val_loss = test_bert(load_bert(savename), typ, max_length, real)
            with open(os.path.join(savename, '{}_{}_loss'.format('haswell' if real else 'mca', typ)), 'w') as f:
                f.write('{}\n'.format(val_loss))

        threading.Thread(
            target=do_test,
            args=(savename, 'val', False)
        ).start()
        threading.Thread(
            target=do_test,
            args=(savename, 'val', True)
        ).start()
        threading.Thread(
            target=do_test,
            args=(savename, 'test', False)
        ).start()
        threading.Thread(
            target=do_test,
            args=(savename, 'test', True)
        ).start()

_dl_dict = {}
def cached_dataloader(data_type, batch_size, max_length, real_data=False):
    key = (data_type, batch_size, real_data)
    if key in _dl_dict:
        yield from _dl_dict[key]
        return
    else:
        _dl_dict[key] = []

    ds = load_dataset(max_length, real_data)
    dataloader = torch.utils.data.DataLoader(ds[data_type], shuffle=True, batch_size=batch_size)

    for batch in tqdm(dataloader):
        input_ids = batch['input_ids'].to(device, non_blocking=True)
        token_type_ids = batch['token_type_ids'].to(device, non_blocking=True)
        attention_mask = batch['attention_mask'].to(device, non_blocking=True)
        labels = batch['label'].to(device, non_blocking=True)

        _dl_dict[key].append({
            'input_ids': input_ids,
            'token_type_ids': token_type_ids,
            'attention_mask': attention_mask,
            'label': labels,
        })

    yield from _dl_dict[key]


def test_bert(bert, data_type, max_length, real_data=False):
    dataloader = cached_dataloader(data_type, 1024, max_length, real_data=real_data)
    num_steps = len(load_dataset(max_length, real_data))
    progress_bar = tqdm(range(num_steps))

    loss_ema = 0
    ema_beta = 0.99
    idx = 0

    losses = []

    bert.eval()
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device, non_blocking=True)
            token_type_ids = batch['token_type_ids'].to(device, non_blocking=True)
            attention_mask = batch['attention_mask'].to(device, non_blocking=True)
            labels = batch['label'].to(device, non_blocking=True)

            outputs = bert(
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                input_ids=input_ids,
                labels=labels,
            )
            loss = outputs.loss.mean()
            losses.append(loss.item())

            loss_ema = loss_ema * ema_beta + loss.item() * (1 - ema_beta)

            progress_bar.update(1)
            idx += 1
            progress_bar.set_description('loss={:.3f}'.format(loss_ema / (1 - ema_beta ** idx)))

    return np.mean(losses)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--haswell-data', action='store_true', default=False)
    parser.add_argument('--max-length', type=int, default=64)

    sp = parser.add_subparsers(dest='action')
    train = sp.add_parser('train')
    train.add_argument('--trial', type=int, default=1)
    train.add_argument('--epochs', type=int, default=1)
    train.add_argument('--load-from')
    train.add_argument('--save-name', required=True)
    train.add_argument('--hidden-size', type=int, default=64)
    train.add_argument('--hidden-layers', type=int, default=2)
    train.add_argument('--attention-heads', type=int, default=2)
    train.add_argument('--train-fraction-idx', type=int)

    test = sp.add_parser('test')
    test.add_argument('--models', nargs='+')
    test.add_argument('--type', choices=['test', 'val'], required=True)

    test = sp.add_parser('cache')

    args = parser.parse_args()


    if args.action == 'train':
        data_fraction = get_data_fractions()[args.train_fraction_idx] if args.train_fraction_idx is not None else None

        _ = list(cached_dataloader('val', 1024, args.max_length, real_data=False)), \
            list(cached_dataloader('test', 1024, args.max_length, real_data=False)), \
            list(cached_dataloader('test', 1024, args.max_length, real_data=True)),

        if args.load_from:
            bert = load_bert(args.load_from)
        else:
            bert = create_bert(max_position=args.max_length, hidden_size=args.hidden_size, hidden_layers=args.hidden_layers, attention_heads=args.attention_heads)
        train_bert(bert, args.epochs, os.path.join(args.save_name, 'trial-{}'.format(args.trial)), args.max_length, data_seed=args.trial, real_data=args.haswell_data, data_fraction=data_fraction)
    elif args.action == 'test':
        _ = list(cached_dataloader(args.type, 1024, args.max_length, real_data=args.haswell_data))

        for model in tqdm(args.models):
            bert = load_bert(model)
            err = test_bert(bert, args.type, args.max_length, real_data=args.haswell_data)
            print('{} {}'.format(model, err))
    elif args.action == 'cache':
        load_tokenizer()
        load_dataset(args.max_length, args.haswell_data)
    else:
        raise ValueError(args.action)


if __name__ == '__main__':
    main()
