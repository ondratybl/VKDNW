import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import ndcg_score, auc
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import xgboost as xgb
import scipy.stats as stats

def plot_roc(arr, model_name, show_plot=True):
    arr = arr[arr[:, 0].argsort()]
    tpr = np.zeros(arr.shape[0] + 1)
    fpr = np.zeros(arr.shape[0] + 1)
    ppv = np.zeros(arr.shape[0] + 1)

    for i, row in enumerate(reversed(arr)):
        pred, tar = tuple(row)
        tpr[i + 1] = tar + tpr[i]
        fpr[i + 1] = (1 - tar) + fpr[i]
        ppv[i + 1] = (tar + ppv[i] * i) / (i + 1)

    tpr = tpr / (arr[:, 1].sum())
    fpr = fpr / (arr.shape[0] - arr[:, 1].sum())

    auc_roc = auc(fpr, tpr)
    auc_pr = auc(tpr, ppv)
    auc_pr10 = auc(tpr[tpr <= 0.1], ppv[tpr <= 0.1])

    if show_plot:
        fig = plt.figure(figsize=(10, 10))
        gs = GridSpec(1, 2, width_ratios=[1, 1])

        ax1 = fig.add_subplot(gs[0])
        ax1.plot(fpr, tpr, color='b')
        ax1.plot([0, 1], [0, 1], linestyle='--', color='r')
        ax1.set_title(f'{model_name}: AUC-ROC {auc_roc:.3f}')
        ax1.set_xlabel('FPR')
        ax1.set_ylabel('TPR')

        ax2 = fig.add_subplot(gs[1])
        ax2.plot(tpr[1:], ppv[1:], color='b')
        ax2.set_title(f'{model_name}:AUC-PR {auc_pr:.3f}')
        ax2.set_xlabel('TPR (recall)')
        ax2.set_ylabel('PPV (precision)')

        plt.show()

    return auc_roc, auc_pr, auc_pr10


def plot_log(log, metric_name):
    plt.figure(figsize=(8, 6))
    for name, group in log.groupby('pred_name'):
        if 'ours' in name:
            plt.plot(group['start_batch'], group[metric_name], label=name, linewidth=4)
        else:
            plt.plot(group['start_batch'], group[metric_name], label=name)
    plt.xlabel('start_batch')
    plt.ylabel(metric_name)
    plt.legend()

    plt.savefig(f'data/{metric_name}.jpg', format='jpg')
    plt.show()


def get_prediction(data, pred_list, target, show_shap=False):
    if 'params' not in pred_list:
        pred_list.append('params')

    data[target + '_logit'] = data[target].apply(lambda p: np.log(p / (100 - p)))

    train_df, temp_df = train_test_split(data, test_size=0.89, random_state=42)
    valid_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

    model = xgb.XGBRegressor(
        objective='reg:squarederror',  # Define the objective as regression with squared error
        n_estimators=300,  # Number of boosting rounds
        learning_rate=0.01,  # Small learning rate to reduce overfitting
        max_depth=3,  # Maximum depth of a tree to avoid overfitting
        subsample=0.8,  # Subsample ratio of the training instances
        colsample_bytree=1,  # Subsample ratio of columns when constructing each tree
        gamma=0.001,  # Minimum loss reduction required to make a further partition
        reg_alpha=1.0,  # L1 regularization term on weights
        reg_lambda=1.0,  # L2 regularization term on weights
    )

    from sklearn.ensemble import RandomForestRegressor
    model = RandomForestRegressor()

    model.fit(
        train_df[pred_list], train_df[target + '_logit'],
        # eval_set=[(train_df[pred_list], train_df[target+'_logit']), (valid_df[pred_list], valid_df[target+'_logit'])],
        # verbose=False
    )

    test_df['prediction'] = model.predict(test_df[pred_list])
    test_df['prediction'] = test_df['prediction'].apply(lambda x: 100 / (1 + np.exp(-x)))

    if show_shap:
        import shap
        explainer = shap.Explainer(model, train_df[pred_list])
        shap.summary_plot(explainer(train_df[pred_list]), train_df[pred_list], max_display=32)

    return test_df[['net', 'fisher_dim', 'prediction', target]].copy()


def get_custom_gain(arr, exp=0.9, k=None):
    if not k:
        k = arr.shape[0]

    tar = arr[arr[:, 0].argsort()][::-1, 1][:k]
    if exp < 0:
        coef = np.log(1 + np.arange(k))
    else:
        coef = np.array([1 * (exp ** i) for i in range(k)])

    coef = coef / np.sum(coef)

    return np.sum(np.multiply(tar, coef))


def get_metrics(test_df, pred_name, target='val_accs', top=50, show_plot=True):

    auc_roc, auc_pr, auc_pr10 = plot_roc(test_df[[pred_name, target]].to_numpy() / 100, pred_name, show_plot)
    gain_exp = get_custom_gain(test_df[[pred_name, target]].to_numpy() / 100, 1 / 2, 10)
    gain_log = get_custom_gain(test_df[[pred_name, target]].to_numpy() / 100, -1, 10)
    kendall = test_df[[pred_name, target]].corr(method='kendall').iloc[0, 1]
    spearman = test_df[[pred_name, target]].corr(method='spearman').iloc[0, 1]
    pearson = test_df[[pred_name, target]].corr(method='pearson').iloc[0, 1]
    gain_norm = ndcg_score(y_true=np.array([test_df[target].astype(float)]),
                           y_score=np.array([test_df[pred_name].astype(float)]), k=20)
    acc_top1 = test_df.nlargest(1, pred_name)[target].mean()
    acc_top_mean = test_df.nlargest(top, pred_name)[target].mean()

    return {
        'pred_name': [pred_name],
        'kendall': [kendall],
        'spearman': [spearman],
        'pearson': [pearson],
        'auc_roc': auc_roc,
        'auc_pr': auc_pr,
        'auc_pr10': auc_pr10,
        'gain_norm': [gain_norm],
        'gain_exp': [gain_exp],
        'gain_log': [gain_log],
        'acc_top1': [acc_top1],
        f'acc_top{top}': [acc_top_mean],
    }


def compute_group_stats(group, col1, col2, method):
    row_count = group.shape[0]
    correlation = group[col1].corr(group[col2], method=method)
    return pd.Series({'corr': correlation, 'count': row_count})


def get_stats(results_tmp, group, target, pred, method='kendall'):
    group_stats = results_tmp.groupby(group).apply(lambda g: compute_group_stats(g, target, pred, method)).reset_index()
    group_stats[group] = [str(p)[0:5] for p in group_stats[group]]

    return group_stats


def plot_stats(grouped, group, target, pred, fig_name, plot=False):
    total_obs = grouped['count'].sum()

    fig, ax1 = plt.subplots()

    ax1.set_xlabel(group)
    ax1.set_ylabel('count', color='tab:blue')
    ax1.bar(grouped[group], grouped['count'], color='tab:blue', alpha=0.6, label='count')
    ax1.tick_params(axis='y', labelcolor='tab:blue')

    plt.xticks(rotation=45)

    ax2 = ax1.twinx()
    ax2.set_ylabel('corr', color='tab:red')
    ax2.plot(grouped[group], grouped['corr'], color='tab:red', marker='o', linestyle='-', linewidth=2, label='corr')
    ax2.tick_params(axis='y', labelcolor='tab:red')
    ax2.set_ylim(-1, 1)
    ax2.axhline(y=0, color='y', linestyle='--', label='corr = 0')

    fig.suptitle(target + ' & ' + pred + ' corr. ' + f'Obs. {total_obs}')
    fig.legend(loc='upper left', bbox_to_anchor=(0.1, 0.9))

    plt.savefig(f'{fig_name}.eps', format='eps')
    if plot:
        plt.show()
    else:
        plt.close()


def analyze_results(api, df_results, zero_shot_score, target):
    archs = list(df_results['net_str'])
    api_valid_accs = list(df_results[target])
    api_flops = list(df_results['flops'])

    if zero_shot_score == 'vkdnw':
        import pandas as pd
        results = pd.DataFrame(df_results)
        results['vkdnw_ratio'] = -(results['vkdnw_lambda_8'] / results['vkdnw_lambda_3']).apply(np.log)
        results['vkdnw'] = results[['vkdnw_dim', 'vkdnw_ratio']].apply(tuple, axis=1).rank(method='dense',
                                                                                           ascending=True).astype(
            int)
        results = {'vkdnw': results['vkdnw'], 'expressivity': list(df_results['expressivity']),
                   'trainability': list(df_results['trainability'])}
    elif zero_shot_score == 'te_nas':
        results = {'ntk': list(df_results['ntk']), 'linear_region': list(df_results['linear_region'])}
    elif zero_shot_score == 'az_nas':
        results = {'expressivity': list(df_results['expressivity']), 'progressivity': list(df_results['progressivity']),
                   'trainability': list(df_results['trainability'])}
    else:
        results = {zero_shot_score: list(df_results[zero_shot_score])}

    fig_scale = 1.1

    if zero_shot_score.lower() == 'az_nas' or zero_shot_score.lower() == 'vkdnw':
        rank_agg = None
        l = len(api_flops)
        rank_agg = np.log(stats.rankdata(api_flops) / l)
        for k in results.keys():
            print(k)
            if rank_agg is None:
                rank_agg = np.log(stats.rankdata(results[k]) / l)
            else:
                rank_agg = rank_agg + np.log(stats.rankdata(results[k]) / l)

    elif zero_shot_score == 'te_nas':
        rank_agg = None
        for k in results.keys():
            print(k)
            if rank_agg is None:
                rank_agg = stats.rankdata(results[k])
            else:
                rank_agg = rank_agg + stats.rankdata(results[k])
    else:
        for k, v in results.items():
            print(k)
            rank_agg = v
    best_idx = np.argmax(rank_agg)

    best_arch, acc = archs[best_idx], api_valid_accs[best_idx]
    if api is not None:
        print("{:}".format(api.query_by_arch(best_arch, "200")))

    x = stats.rankdata(rank_agg)
    y = stats.rankdata(api_valid_accs)
    kendalltau = stats.kendalltau(x, y)
    spearmanr = stats.spearmanr(x, y)
    pearsonr = stats.pearsonr(x, y)
    print("{}: {}\t{}\t{}\t".format(k, kendalltau[0], pearsonr[0], spearmanr[0]))
    plt.figure(figsize=(4 * fig_scale, 3 * fig_scale))
    plt.scatter(x, y, linewidths=0.1)
    plt.scatter(x[best_idx], y[best_idx], c="r", linewidths=0.1)
    plt.title(f"{zero_shot_score} with acc {acc:.2f}")
    plt.show()


def compute_vkdnw(df_results, ind_high=8, ind_low=3):
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter(action='ignore', category=pd.errors.SettingWithCopyWarning)

        df_results.loc[:, 'temp'] = -(df_results.loc[:, f'vkdnw_lambda_{ind_high}'] / df_results.loc[:, f'vkdnw_lambda_{ind_low}']).apply(np.log)
        ret = df_results[['vkdnw_dim', 'temp']].apply(tuple, axis=1).rank(method='dense',
                                                                                           ascending=True).astype(
            int)
        return list(ret)


def get_results_from_api(api, arch, dataset='cifar10'):
    dataset_candidates = ['cifar10-valid', 'cifar10', 'cifar100', 'ImageNet16-120']
    assert dataset in dataset_candidates
    index = api.query_index_by_arch(arch)
    api._prepare_info(index)
    archresult = api.arch2infos_dict[index]['200']

    if dataset == 'cifar10-valid':
        acc = archresult.get_metrics(dataset, 'x-valid', iepoch=None, is_random=False)['accuracy']
    elif dataset == 'cifar10':
        acc = archresult.get_metrics(dataset, 'ori-test', iepoch=None, is_random=False)['accuracy']
    else:
        acc = archresult.get_metrics(dataset, 'x-test', iepoch=None, is_random=False)['accuracy']
    flops = archresult.get_compute_costs(dataset)['flops']
    params = archresult.get_compute_costs(dataset)['params']

    return acc, flops, params


def generate_accs(api, dataset=None, features_path='../../GRAF/zc_combine/data/nb201_features.csv'):

    import os
    import pickle
    if os.path.exists("./tss_all_arch.pickle"):
        with open("./tss_all_arch.pickle", "rb") as fp:
            archs = pickle.load(fp)
    else:
        import random
        from xautodl.models.cell_searchs.genotypes import Structure
        from xautodl.models import get_search_spaces
        def random_genotype(max_nodes, op_names):
            genotypes = []
            for i in range(1, max_nodes):
                xlist = []
                for j in range(i):
                    node_str = "{:}<-{:}".format(i, j)
                    op_name = random.choice(op_names)
                    xlist.append((op_name, j))
                genotypes.append(tuple(xlist))
            arch = Structure(genotypes)
            return arch
        search_space = get_search_spaces('tss', "nats-bench")
        archs = random_genotype(4, search_space).gen_all(search_space, 4, False)

    accs = pd.DataFrame({'net_str': [a.tostr() for a in archs]})

    api_valid_accs, api_flops, api_params = [], [], []
    for a in accs['net_str']:
        try:
            valid_acc, flops, params = get_results_from_api(api, a, dataset=dataset)
            api_valid_accs.append(valid_acc)
            api_flops.append(flops)
            api_params.append(params)
        except Exception as e:
            print(e)

    accs['val_accs'] = api_valid_accs
    accs['flops'] = api_flops
    accs['params'] = api_params

    features = pd.read_csv(features_path)

    accs = pd.merge(accs, features, how='left', on='net_str')

    return accs


def get_scores(df_run, compute_graf=True, zero_cost_score_list=None):

    # ZS scores
    for zero_shot_score in zero_cost_score_list:
        print(f'Running {zero_shot_score}')
        if zero_shot_score.lower() == 'az_nas':
            rank_agg = None
            l = df_run.shape[0]
            rank_agg = np.log(stats.rankdata(df_run.loc[:, 'flops']) / l)
            for k in ['expressivity', 'progressivity', 'trainability']:
                if rank_agg is None:
                    rank_agg = np.log(stats.rankdata(df_run.loc[:, k]) / l)
                else:
                    rank_agg = rank_agg + np.log(stats.rankdata(df_run.loc[:, k]) / l)
            df_run[zero_shot_score + '_rank'] = rank_agg

        elif zero_shot_score.lower() == 'te_nas':
            rank_agg = None
            for k in ['ntk', 'linear_region']:
                if rank_agg is None:
                    rank_agg = stats.rankdata(df_run.loc[:, k])
                else:
                    rank_agg = rank_agg + stats.rankdata(df_run.loc[:, k])

            df_run[zero_shot_score + '_rank'] = rank_agg

        elif zero_shot_score.lower() == 'vkdnw':
            df_run.loc[:, 'temp'] = -(df_run.loc[:, f'vkdnw_lambda_{8}'] / df_run.loc[:, f'vkdnw_lambda_{3}']).apply(
                np.log)
            df_run[zero_shot_score + '_rank'] = df_run[['vkdnw_dim', 'temp']].apply(tuple, axis=1).rank(method='dense',
                                                                                                        ascending=True).astype(
                int).apply(np.log)

            df_run[zero_shot_score + '_exp_rank'] = df_run[zero_shot_score + '_rank'] + df_run[
                'progressivity'].rank().apply(np.log) + df_run['trainability'].rank().apply(np.log) + df_run[
                                                        'flops'].rank().apply(np.log)

            df_run[zero_shot_score + '_prog_rank'] = df_run[zero_shot_score + '_rank'] + df_run[
                'expressivity'].rank().apply(np.log) + df_run['trainability'].rank().apply(np.log) + df_run[
                                                         'flops'].rank().apply(np.log)

            df_run[zero_shot_score + '_train_rank'] = df_run[zero_shot_score + '_rank'] + df_run[
                'progressivity'].rank().apply(np.log) + df_run['expressivity'].rank().apply(np.log) + df_run[
                                                          'flops'].rank().apply(np.log)

            df_run[zero_shot_score + '_comb_rank'] = df_run[zero_shot_score + '_rank'] + df_run[
                'expressivity'].rank().apply(np.log) + df_run['trainability'].rank().apply(np.log) + df_run[
                                                         'flops'].rank().apply(np.log) + df_run['jacov'].rank().apply(np.log)

            df_run['vkdnw_entropy_rank'] = df_run[['vkdnw_dim', 'vkdnw_entropy']].apply(tuple, axis=1).rank(method='dense',
                                                                                         ascending=True).astype(int)

        else:
            df_run[zero_shot_score + '_rank'] = df_run.loc[:, zero_shot_score]

        # Trained model
        ...

    return df_run
