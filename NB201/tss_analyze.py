import os, sys, time, glob, random, argparse
import wandb
import numpy as np
import pandas as pd

from nats_bench import create
from ZeroShotProxy import *
from tss_utils import plot_stats, get_stats, get_metrics, analyze_results, generate_accs, get_scores
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

run_dict = {
    'cifar10': {
        # same images
        # 'nazderaze/VKDNW/n6n44keg': 1,
        # 'nazderaze/VKDNW/6c1pv095': 2,
        # 'nazderaze/VKDNW/45us11be': 3,
        # 'nazderaze/VKDNW/rwp2qvkw': 4,
        # 'nazderaze/VKDNW/6bsssh4z': 5,

        # different images
        # 'nazderaze/VKDNW/7a8jm975' : -1 # testing (successfull)
        'nazderaze/VKDNW/cixw6r9b': 1,
        'nazderaze/VKDNW/1nikalf3': 2,
        'nazderaze/VKDNW/r5l3qg12': 3,
        'nazderaze/VKDNW/fq581h0d': 4,
        'nazderaze/VKDNW/0yfzaxra': 5,
    },
    'cifar100': {
        # same images
        # 'nazderaze/VKDNW/qmq5vp3k': 1,
        # 'nazderaze/VKDNW/31lrq6p7': 2,
        # 'nazderaze/VKDNW/3qazc6po': 3,
        # 'nazderaze/VKDNW/424twoyv': 4,
        # 'nazderaze/VKDNW/783h4opf': 5,

        # different images
        'nazderaze/VKDNW/n2m8i53l': 1,
        'nazderaze/VKDNW/48xbnfdg': 2,
        'nazderaze/VKDNW/f0czkx5u': 3,
        'nazderaze/VKDNW/ldv3b1bh': 4,
        'nazderaze/VKDNW/qwae4nqx': 5,
    },
    'ImageNet16-120': {
        # same images
        # 'nazderaze/VKDNW/ftg0tdsa': 1,
        # 'nazderaze/VKDNW/vqf1ey6x': 2,
        # 'nazderaze/VKDNW/v0a0m67q': 3,
        # 'nazderaze/VKDNW/uiv37u18': 4,
        # 'nazderaze/VKDNW/c1338vfg': 5,

        # different images
        'nazderaze/VKDNW/55f1omxn': 1,
        'nazderaze/VKDNW/amdcxrz7': 2,
        'nazderaze/VKDNW/sl0rjhwh': 3,
        'nazderaze/VKDNW/z2ph6iav': 4,
        'nazderaze/VKDNW/ol9rwkeo': 5,
    }
}

def main(dataset, compute_graf):

    api_nats = create('/mnt/personal/tyblondr/NATS-tss-v1_0-3ffb9-simple/', 'tss', fast_mode=True, verbose=False)

    if os.path.exists(f"./tss_features_{dataset}.pickle"):
        archs = pd.read_pickle(f"./tss_features_{dataset}.pickle")
    else:
        archs = generate_accs(api_nats, dataset=dataset)
        print(f'No. of generated archs: {archs.shape[0]}')
        archs.to_pickle(f"./tss_features_{dataset}.pickle")


    log = None
    results = None
    for run_id, seed in run_dict[dataset].items():

        run = pd.DataFrame(api_wandb.run(run_id).scan_history())
        run.rename({'arch': 'net_str'}, axis=1, inplace=True)

        run = pd.merge(archs, run, on='net_str', how='inner')
        if compute_graf:
            run = run.loc[run['net'].notnull(), :]  # keep only nets with features

        print(f'No. of archs for seed {seed} after filtering: {run.shape[0]}.')

        for col in run.columns:
            if col not in ['net_str', 'net']:
                run[col] = run[col].astype(float)

        if 'jacov' in run.columns:
            run['jacov'] = run['jacov'].fillna(run['jacov'].min()).astype(float)

        df_scores = get_scores(run.copy(), compute_graf=compute_graf, zero_cost_score_list=zero_cost_score_list)
        df_scores['dataset'] = dataset
        df_scores['seed'] = seed
        if results is None:
            results = df_scores
        else:
            results = pd.concat([results, df_scores], ignore_index=True)
    print(f'Total number of records: {results.shape[0]}')

    log = None
    for seed in results['seed'].unique():

        results_temp = results.loc[results['seed'] == seed, :].copy()
        for zero_cost_rank in [p for p in results_temp.columns if '_rank' in p]:
            results_temp[[zero_cost_rank]] = results_temp[[zero_cost_rank]].apply(
                lambda x: x.replace(-np.inf, x[x != -np.inf].min()))
            results_temp[[zero_cost_rank]] = results_temp[[zero_cost_rank]].apply(
                lambda x: x.replace(np.inf, x[x != np.inf].max()))
            log_temp = pd.DataFrame(get_metrics(results_temp, pred_name=zero_cost_rank, show_plot=False))

            if log is None:
                log = log_temp.copy()
            else:
                log = pd.concat([log, log_temp.copy()], ignore_index=True)

            if seed == min(results['seed'].unique()):
                plot_stats(get_stats(results_temp, 'vkdnw_dim', target, zero_cost_rank), 'vkdnw_dim', target,
                           zero_cost_rank, f'{dataset}_{str(compute_graf)}_{zero_cost_rank}')

    log = log.groupby('pred_name', as_index=False).agg(['mean', 'std']).reset_index()
    log['dataset'] = dataset
    log['no_seeds'] = len(results['seed'].unique())
    log['archs_filtered'] = str(compute_graf)

    log_train = None

    pred_lists = {
        'model_vkdnw': [p for p in results.columns if '_lambda_' in p] + ['vkdnw_chisquare', 'vkdnw_dim', 'flops'],
        'model_vkdnw+zs': [p for p in results.columns if '_lambda_' in p] + ['vkdnw_chisquare', 'vkdnw_dim',
                                                                             'flops'] + ['expressivity',
                                                                                         'progressivity',
                                                                                         'trainability', 'jacov',
                                                                                         'gradsign', 'zico', 'zen',
                                                                                         'grad_norm', 'naswot',
                                                                                         'synflow', 'snip', 'grasp',
                                                                                         'ntk', 'linear_region'],
        'model_vkdnw+zs+graf': [p for p in results.columns if '_lambda_' in p] + ['vkdnw_chisquare', 'vkdnw_dim',
                                                                                  'flops'] + ['expressivity',
                                                                                              'progressivity',
                                                                                              'trainability',
                                                                                              'jacov', 'gradsign',
                                                                                              'zico', 'zen',
                                                                                              'grad_norm', 'naswot',
                                                                                              'synflow', 'snip',
                                                                                              'grasp', 'ntk',
                                                                                              'linear_region'] + [p
                                                                                                                  for
                                                                                                                  p
                                                                                                                  in
                                                                                                                  results.columns
                                                                                                                  if
                                                                                                                  'op_' in p] + [
                                   p for p in results.columns if 'node_' in p]
    }

    for train_size in [32, 128, 1024]:
        for seed in results['seed'].unique():

            results_temp = results.loc[results['seed'] == seed, :].copy()
            for model_name, pred_list in pred_lists.items():

                results_temp[pred_list] = results_temp[pred_list].apply(lambda x: x.replace(-np.inf, x[x != -np.inf].min()))
                results_temp[pred_list] = results_temp[pred_list].apply(lambda x: x.replace(np.inf, x[x != np.inf].max()))

                train_df, test_df = train_test_split(results_temp, test_size=1 - (1024 / results_temp.shape[0]),
                                                     random_state=seed)
                model = Pipeline([
                    ('scaler', StandardScaler()),  # Step 1: Standardize features
                    ('regressor', RandomForestRegressor(n_estimators=300))  # Step 2: Train RandomForestRegressor
                ])
                model.fit(train_df[pred_list], train_df[target])
                test_df['pred_' + model_name] = model.predict(test_df[pred_list])
                log_train_temp = pd.DataFrame(get_metrics(test_df, 'pred_' + model_name, show_plot=False))
                log_train_temp['train_size'] = train_size

                if log_train is None:
                    log_train = log_train_temp.copy()
                else:
                    log_train = pd.concat([log_train, log_train_temp.copy()], ignore_index=True)

                if seed == min(results['seed'].unique()):
                    plot_stats(get_stats(test_df, 'vkdnw_dim', target, 'pred_' + model_name), 'vkdnw_dim', target,
                               model_name, f'{dataset}_{str(compute_graf)}_{model_name}_{train_size}')

    log_train = log_train.groupby(['pred_name', 'train_size'], as_index=False).agg(['mean', 'std']).reset_index()
    log_train['dataset'] = dataset
    log_train['no_seeds'] = len(results['seed'].unique())
    log_train['archs_filtered'] = str(compute_graf)
    log_train


if __name__ == '__main__':

    target = 'val_accs'
    zero_cost_score_list = ['vkdnw', 'vkdnw_dim', 'vkdnw_chisquare', 'az_nas', 'jacov','gradsign', 'zico', 'zen', 'grad_norm', 'naswot', 'synflow', 'snip', 'grasp', 'te_nas']

    api_wandb = wandb.Api()

    for ds in ['cifar10', 'cifar100', 'ImageNet16-120']:
        for filter in [True, False]:
            main(ds, filter)