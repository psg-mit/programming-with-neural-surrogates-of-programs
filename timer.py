#!/usr/bin/env python3

import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import torch
torch.set_num_threads(1)

import argparse
import pickle
import transformers
import tokenizers
import datasets
import tqdm.auto as tqdm
import torch
import numpy as np
import time
import subprocess
import urllib.request
import onnxruntime
import shutil
import tempfile
import random

DATA_DIR = os.path.join(os.path.expanduser('~'), '.cache', 'transformer-surrogates')
BLOCKS_URL = 'https://storage.googleapis.com/renda-transformer-surrogates/mca-blocks'
TOKENIZER_FILE = os.path.join(DATA_DIR, 'tokenizer.json')
DATA_FILE = os.path.join(DATA_DIR, 'blocks')
EXPORT_FILE = os.path.join(DATA_DIR, 'export.py')
OPT_FILE = os.path.join(DATA_DIR, 'opt.py')

os.makedirs(DATA_DIR, exist_ok=True)
if not os.path.exists(DATA_FILE):
    urllib.request.urlretrieve(BLOCKS_URL, DATA_FILE)
if not os.path.exists(EXPORT_FILE):
    urllib.request.urlretrieve('https://github.com/huggingface/transformers/raw/acc3bd9d2a73fcc7d3509767d65b2f40962d9330/src/transformers/convert_graph_to_onnx.py', EXPORT_FILE)
if not os.path.exists(OPT_FILE):
    urllib.request.urlretrieve('https://github.com/microsoft/onnxruntime/raw/4fd9fef9ee04c0844d679e81264779402cfa445c/onnxruntime/python/tools/transformers/optimizer.py', OPT_FILE)

with open(DATA_FILE, 'rb') as f:
    blocks = pickle.load(f)
blocks['code'] = blocks['code'].apply(lambda x: x + '\n[PAD]\n')

codes = list(blocks['code'].values)[:]
mca_trues = list(blocks['mca'].values)[:]
hsw_trues = list(blocks['hsw-true'].values)[:]
seed = 0
random.Random(seed).shuffle(codes)
random.Random(seed).shuffle(mca_trues)
random.Random(seed).shuffle(hsw_trues)
test_codes = codes[:10000]
test_codes_nopad = ['\n'.join(x.split('\n')[:-2]) + '\n' for x in test_codes]
test_mca_trues = mca_trues#[:10000]
test_hsw_trues = hsw_trues#[:10000]

opts = onnxruntime.SessionOptions()
opts.intra_op_num_threads = 1
opts.inter_op_num_threads = 1
opts.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL
opts.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
opts.enable_profiling = False


def time_mca():
    times = []
    for i in tqdm.trange(13):
        mca_proc = subprocess.Popen(
            'numactl -C 0'
            ' DiffTune/llvm-mca-parametric/build/bin/llvm-mca -parameters noop'
            ' -mtriple=x86_64-unknown-unknown -march=x86-64 -mcpu=haswell --all-views=0'
            ' --summary-view -iterations=100',
            stdin=subprocess.PIPE, universal_newlines=True,shell=True,
            bufsize=1, stdout=subprocess.PIPE, )

        time.sleep(1)

        t1 = time.time()
        for code in test_codes_nopad:
            mca_proc.stdin.write('# LLVM-MCA-BEGIN\n{}# LLVM-MCA-END\n'.format(
                code,
            ))

        mca_preds = []
        mca_proc.stdin.close()
        for line in mca_proc.stdout:
            if 'Total Cycles' in line:
                mca_preds.append(float(line.split()[-1]))
        mca_proc.wait()
        t2 = time.time()

        if i > 3:
            times.append(t2 - t1)

    return np.mean(times)


def time_surrogate(model):
    times = []

    tmpdir = tempfile.mkdtemp()
    surrogate_onnx = os.path.join(tmpdir, 'surrogate.onnx')

    tfile = os.path.join(model, 'tokenizer.json')
    tokenizer_existed_before = os.path.exists(tfile)

    try:
        done_move = False
        if tokenizer_existed_before:
            temp_tfile = tempfile.mkstemp()[1]
            shutil.move(tfile, temp_tfile)
            done_move = True

        shutil.copy(TOKENIZER_FILE, tfile)

        subprocess.check_call(['python', EXPORT_FILE, '--pipeline', 'feature-extraction', '--model', model, '--framework', 'pt', surrogate_onnx])
    finally:
        if tokenizer_existed_before:
            if done_move:
                shutil.move(temp_tfile, tfile)

    ort_session = onnxruntime.InferenceSession(
        surrogate_onnx,
        sess_options=opts)

    bert = transformers.BertForSequenceClassification.from_pretrained(model)

    wt = bert.classifier.weight.detach().numpy().ravel()
    bs = bert.classifier.bias.detach().numpy().ravel()

    fast_tokenizer = transformers.PreTrainedTokenizerFast(tokenizer_file=TOKENIZER_FILE, pad_token='[PAD]')

    with torch.no_grad():
        for i in tqdm.trange(13):

            bert_preds = []

            t1 = time.time()
            for row in test_codes_nopad:
                tokens = fast_tokenizer(row, truncation=True, max_length=64)
                tokens = {name: np.atleast_2d(value) for (name, value) in tokens.items()}
                bert_preds.append(((ort_session.run(None, tokens)[1] @ wt + bs) * 10000)[0])

            t2 = time.time()

            if i > 3:
                times.append(t2 - t1)

    return np.mean(times)


def main():
    parser = argparse.ArgumentParser()
    sp = parser.add_subparsers(dest='action')
    mca = sp.add_parser('mca')
    surrogate = sp.add_parser('surrogate')
    surrogate.add_argument('surrogate', required=True)

    args = parser.parse_args()

    if args.action == 'mca':
        print(time_mca())
    elif args.action == 'surrogate':
        print(time_surrogate(args.surrogate))

if __name__ == '__main__':
    main()
