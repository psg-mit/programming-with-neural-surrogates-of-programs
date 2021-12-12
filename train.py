#!/usr/bin/env python

import subprocess
import os
import multiprocessing as mp
import numpy as np
import argparse
import results
import glob

def run_trial(flags):
    args = [
        'python', 'main.py', *map(str, flags)
    ]
    print(args)
    subprocess.run(args)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--jobs', type=int)
    parser.add_argument('action', choices=['hyperparameter-search', 'adaptation'])
    args = parser.parse_args()

    jobs = []
    if args.action == 'hyperparameter-search':
        for width in [16, 32, 64, 128]:
            for trial in [1, 2, 3]:
                jobs.append(['train', '--epochs', 100, '--trial', trial, '--save-name', 'hyperparameter-search-{}'.format(width), '--hidden-size', width, '--hidden-layers', 2, '--attention-heads', 2])
    elif args.action == 'adaptation':
        df = results.read_models(glob.glob('hyperparameter-search-*'))
        best_vals = df.groupby(['model', 'trial'])['mca_val_loss'].idxmin()
        df = df.set_index(['model', 'trial'])
        valid_rows = df.iloc[best_vals.values]['mca_val_loss'] <= 0.1
        best_speedup = df.iloc[best_vals.values][valid_rows].reset_index().groupby('model').mean()['speed'].idxmin()
        mbv = best_vals.reset_index().set_index('model').loc[best_speedup]['mca_val_loss']
        rows = [df.iloc[idx] for idx in mbv.values]

        for trial in [1, 2, 3]:
            for n_data in range(10):
                for train_on_real_data in [False, True]:
                    jobs.append(['--haswell-data', 'train', '--epochs', 100, '--trial', trial, '--save-name', 'adaptation-{}-{}'.format('baseline' if train_on_real_data else 'adapt', n_data), '--hidden-size', best_speedup.split('-')[-1], '--hidden-layers', 2, '--attention-heads', 2, '--train-fraction-idx', n_data] + ([] if train_on_real_data else ['--load-from', '{}/trial-{}-epoch-{}'.format(best_speedup, trial, int(rows[trial - 1]['epoch']))]))

    else:
        raise ValueError(args.action)

    n_jobs = args.jobs or 1
    with mp.Pool(n_jobs) as p:
        p.map(run_trial, jobs)

if __name__ == '__main__':
    main()
