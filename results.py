#!/usr/bin/env python

import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import glob
import timer
import pickle
import numpy as np

DATA_DIR = os.path.join(os.path.expanduser('~'), '.cache', 'transformer-surrogates')

if not os.path.exists('figures'):
    os.mkdir('figures')

def normalize_df_time(df):
    df = df.set_index(['model', 'trial'])
    mins = df.reset_index().groupby(['model', 'trial']).min()['mtime']
    for l in mins.index:
        df.loc[l, 'mtime'] = df.loc[l]['mtime'] - mins.loc[l]
    return df.reset_index()

def time_model(model):
    dn = os.path.basename(os.path.dirname(model))
    speed_fname = os.path.join(DATA_DIR, '{}-speed'.format(dn))
    if os.path.exists(speed_fname):
        with open(speed_fname) as f:
            return float(f.read())
    timing = timer.time_surrogate(model)
    with open(speed_fname, 'w+') as f:
        f.write(str(timing))
    return timing

def time_mca():
    speed_fname = os.path.join(DATA_DIR, 'mca-speed')
    if os.path.exists(speed_fname):
        with open(speed_fname) as f:
            return float(f.read())
    timing = timer.time_mca()
    with open(speed_fname, 'w+') as f:
        f.write(str(timing))
    return timing

def read_models(models, read_speed=True):
    data = []
    for model in models:
        for d in os.listdir(model):
            _, trial, _, epoch = d.split('-')
            trial, epoch = int(trial), int(epoch)
            mtime = os.path.getmtime(os.path.join(model, d, 'train_loss'))
            with open(os.path.join(model, d, 'train_loss')) as f:
                train_loss = float(f.read())
            with open(os.path.join(model, d, 'mca_val_loss')) as f:
                mca_val_loss = float(f.read())
            with open(os.path.join(model, d, 'haswell_val_loss')) as f:
                haswell_val_loss = float(f.read())
            with open(os.path.join(model, d, 'mca_test_loss')) as f:
                mca_test_loss = float(f.read())
            with open(os.path.join(model, d, 'haswell_test_loss')) as f:
                haswell_test_loss = float(f.read())

            if read_speed:
                timing = time_model(os.path.join(model, d))
                data.append((model, trial, epoch, mtime, train_loss, mca_val_loss, haswell_val_loss, mca_test_loss, haswell_test_loss, timing))
            else:
                data.append((model, trial, epoch, mtime, train_loss, mca_val_loss, haswell_val_loss, mca_test_loss, haswell_test_loss))

    return pd.DataFrame(data, columns=['model', 'trial', 'epoch', 'mtime', 'train_loss', 'mca_val_loss', 'haswell_val_loss', 'mca_test_loss', 'haswell_test_loss'] + (['speed'] if read_speed else []))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['section-3', 'table-4', 'figure-2', 'surrogate-compilation-telemetry', 'surrogate-adaptation-telemetry'])
    args = parser.parse_args()

    if args.action == 'table-4':
        df = read_models(glob.glob('hyperparameter-search-*'))
        best_vals = df.groupby(['model', 'trial'])['mca_val_loss'].idxmin()
        df = df.set_index(['model', 'trial'])

        print(r'''
        \begin{tabular}{c|ccc}
        \toprule
        \textbf{Embedding Width} & \textbf{MAPE} & \textbf{Speedup over W=128} \\
        \midrule
        ''')

        valid_rows = df.iloc[best_vals.values]['mca_val_loss'] <= 0.1
        best_speedup = df.iloc[best_vals.values][valid_rows].reset_index().groupby('model').mean()['speed'].idxmin()

        for m in [128, 64, 32, 16]:
            mname = 'hyperparameter-search-{}'.format(m)
            mbv = best_vals.reset_index().set_index('model').loc[mname]['mca_val_loss']
            rows = [df.iloc[idx] for idx in mbv.values]

            speedups = [df.sort_index().loc[('hyperparameter-search-128', 'speed')].values[0] / row['speed'] for row in rows]
            def boldit(x):
                if mname == best_speedup:
                    return r'\textbf{{{}}}'.format(x)
                else:
                    return x
            print(r'{} & ${}$ & ${}$ \\'.format(
                boldit(m),
                boldit(r'{:.1f}\%'.format(np.mean([row['mca_val_loss'] * 100 for row in rows]))),
                boldit(r'{:.3g}\times'.format(np.mean(speedups)))
            ))
        print(r'''
        \bottomrule
        \end{tabular}%
        ''')
    elif args.action == 'section-3':
        df = read_models(glob.glob('hyperparameter-search-*'))
        best_vals = df.groupby(['model', 'trial'])['mca_val_loss'].idxmin()
        df = df.set_index(['model', 'trial'])

        valid_rows = df.iloc[best_vals.values]['mca_val_loss'] <= 0.1
        best_speedup = df.iloc[best_vals.values][valid_rows].reset_index().groupby('model').mean()['speed'].idxmin()

        mbv = best_vals.reset_index().set_index('model').loc[best_speedup]['mca_val_loss']
        rows = [df.iloc[idx] for idx in mbv.values]

        mca_timing = time_mca()
        print('llvm-mca:  {} blocks/second'.format(len(timer.test_codes) / mca_timing))
        print('surrogate: {} blocks/second'.format(len(timer.test_codes) / np.mean([row['speed'] for row in rows])))
        print('Speedup: {}'.format(np.mean([mca_timing / row['speed'] for row in rows])))

        print('MAPE against llvm-mca:      {}'.format(np.mean([row['mca_test_loss'] for row in rows])))
        print('MAPE against ground truth:  {}'.format(np.mean([row['haswell_test_loss'] for row in rows])))
        # print('Ground truth error: {}'.format())

    elif args.action == 'surrogate-compilation-telemetry':
        df = normalize_df_time(read_models(glob.glob('hyperparameter-search-*')))
        best_vals = df.groupby(['model', 'trial'])['mca_val_loss'].idxmin()
        df = df.set_index(['model', 'trial'])

        valid_rows = df.iloc[best_vals.values]['mca_val_loss'] <= 0.1
        best_speedup = df.iloc[best_vals.values][valid_rows].reset_index().groupby('model').mean()['speed'].idxmin()

        rows = df.reset_index().set_index('model').loc[best_speedup].groupby('epoch').agg({'train_loss': ['min', 'median', 'max'], 'mca_val_loss': ['min', 'median', 'max'], 'mca_test_loss': ['min', 'median', 'max'], 'mtime': 'mean'}).sort_index().reset_index()

        for typ in ['train', 'val', 'test']:
            for x_axis in ['epochs', 'hours']:
                plt.figure()
                xs = rows['mtime' if x_axis == 'hours' else 'epoch'].values
                ys = rows['{}_loss'.format({'train': 'train', 'val': 'mca_val', 'test': 'mca_test'}[typ])]
                ys_lower = ys['median'] - ys['min']
                ys_upper = ys['max'] - ys['median']
                ys_median = ys['median']

                plt.errorbar(xs, ys_median, [ys_lower, ys_upper], fmt='-', label='{} MAPE'.format(typ))
                best_epochs = rows['mca_val_loss'].idxmin()

                zz = df.reset_index()
                zz = zz[zz['model'] == best_speedup]
                idxmin = zz.groupby('trial').idxmin()['mca_val_loss']
                xs = np.array([zz.loc[i]['mtime' if x_axis == 'hours' else 'epoch'] for i in idxmin.values])
                ys = [zz.loc[i]['{}_loss'.format({'train': 'train', 'val': 'mca_val', 'test': 'mca_test'}[typ])] for i in idxmin.values]
                plt.plot(xs, ys, '*', label='Min-validation Epoch', zorder=-1)

                plt.legend()
                plt.title('{} MAPE v.s. {}'.format(typ, x_axis))
                plt.xlabel(x_axis)
                plt.ylabel(typ)
                plt.savefig('figures/compilation-telemetry-{}-{}.pdf'.format(typ, x_axis))

    elif args.action == 'surrogate-adaptation-telemetry':
        df = normalize_df_time(read_models(glob.glob('adaptation-*'), False))
        df['typ'] = df['model'].apply(lambda x: x.split('-')[1])
        df['idx'] = df['model'].apply(lambda x: x.split('-')[2]).apply(int)
        best_vals = df.groupby(['model', 'trial'])['mca_val_loss'].idxmin()
        df = df.set_index(['model', 'trial'])


        for idx in range(10):
            rows = df[df['idx'] == idx].reset_index().groupby(['model', 'typ', 'epoch']).agg({'train_loss': ['min', 'median', 'max'], 'mca_val_loss': ['min', 'median', 'max'], 'mca_test_loss': ['min', 'median', 'max'], 'mtime': 'mean'}).reset_index().sort_values('epoch')
            for typ in ['train', 'val', 'test']:
                for x_axis in ['epochs', 'hours']:
                    plt.figure()
                    for traintyp in ['baseline', 'adapt']:
                        mrows = rows[rows['typ'] == traintyp]
                        xs = mrows['mtime' if x_axis == 'hours' else 'epoch'].values
                        ys = mrows['{}_loss'.format({'train': 'train', 'val': 'mca_val', 'test': 'mca_test'}[typ])]
                        ys_lower = ys['median'] - ys['min']
                        ys_upper = ys['max'] - ys['median']
                        ys_median = ys['median']

                        plt.errorbar(xs, ys_median, [ys_lower, ys_upper], fmt='-', label='{} {} MAPE'.format(traintyp, typ))
                        best_epochs = mrows['mca_val_loss'].idxmin()

                        zz = df.reset_index()
                        zz = zz[zz['model'] == 'adaptation-{}-{}'.format(traintyp, idx)]
                        idxmin = zz.groupby('trial').idxmin()['mca_val_loss']
                        xs = np.array([zz.loc[i]['mtime' if x_axis == 'hours' else 'epoch'] for i in idxmin.values])
                        ys = [zz.loc[i]['{}_loss'.format({'train': 'train', 'val': 'mca_val', 'test': 'mca_test'}[typ])] for i in idxmin.values]
                        plt.plot(xs, ys, '*', label='Min-val. Epoch', zorder=-1)

                    plt.legend()
                    data_frac = (np.exp(np.linspace(0, 1, 10) * (np.log(215729) - np.log(100)) + np.log(100)) / 215729)[idx]
                    plt.title('{} MAPE v.s. {} ({:.2%} data)'.format(typ, x_axis, data_frac))
                    plt.xlabel(x_axis)
                    plt.ylabel(typ)
                    plt.savefig('figures/adaptation-telemetry-{}-{}-{}.pdf'.format(idx, typ, x_axis))

    elif args.action == 'figure-2':
        df = read_models(glob.glob('adaptation-*'), False)
        best_vals = df.groupby(['model', 'trial'])['haswell_val_loss'].idxmin()
        df['typ'] = df['model'].apply(lambda x: x.split('-')[1])
        df['idx'] = df['model'].apply(lambda x: x.split('-')[2]).apply(int)
        df = df.set_index(['model', 'trial'])

        plt.figure()


        with open(os.path.join(DATA_DIR, 'blocks'), 'rb') as f:
            blocks = pickle.load(f)

        mca_err = ((blocks['mca'] - blocks['hsw-true']) / blocks['hsw-true']).abs().mean()
        xs = np.exp(np.linspace(0, 1, 10) * (np.log(215729) - np.log(100)) + np.log(100))
        plt.plot(xs, [mca_err for _ in range(len(xs))], '--', color='k', label='llvm-mca')

        for typ in ['baseline', 'adapt']:
            ys = np.array([
                [df.iloc[best_vals.loc[('adaptation-{}-{}'.format(typ, idx), trial)]]['haswell_test_loss'] for trial in range(1, 4)]
                for idx in range(10)
            ])
            plt.errorbar(xs, np.median(ys, axis=1), [np.median(ys, axis=1) - ys.min(axis=1), ys.max(axis=1) - np.median(ys, axis=1)], fmt='o--', label={'baseline': 'Scratch Neural Network', 'adapt': 'Surrogate Adaptation'}[typ])
        plt.legend()
        plt.xscale('log')
        plt.xlabel('Number of Ground-Truth Training Data Points')
        plt.ylabel('MAPE Against Ground-Truth')
        plt.tight_layout()
        plt.savefig('figures/figure-2.pdf')

    else:

        raise ValueError(args.action)


if __name__ == '__main__':
    main()
