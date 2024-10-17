import os, sys, time, glob, random, argparse
import numpy as np
from copy import deepcopy
import torch
import torch.nn as nn
import time
import tqdm
import scipy.stats as stats
import matplotlib.pyplot as plt
import pickle

# XAutoDL
from xautodl.config_utils import load_config, dict2config, configure2str
from xautodl.datasets import get_datasets, get_nas_search_loaders
from xautodl.procedures import (
    prepare_seed,
    prepare_logger,
    save_checkpoint,
    copy_checkpoint,
    get_optim_scheduler,
)
from xautodl.utils import get_model_infos, obtain_accuracy
from xautodl.log_utils import AverageMeter, time_string, convert_secs2time
from xautodl.models import get_search_spaces

# API
from nats_bench import create

# custom modules
from custom.tss_model import TinyNetwork
from xautodl.models.cell_searchs.genotypes import Structure
from ZeroShotProxy import *

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

parser = argparse.ArgumentParser("Training-free NAS on NAS-Bench-201 (NATS-Bench-TSS)")
parser.add_argument("--data_path", type=str, default='./cifar.python', help="The path to dataset")
parser.add_argument("--dataset", type=str, default='cifar10',choices=["cifar10", "cifar100", "ImageNet16-120"], help="Choose between Cifar10/100 and ImageNet-16.")

# channels and number-of-cells
parser.add_argument("--search_space", type=str, default='tss', help="The search space name.")
parser.add_argument("--config_path", type=str, default='./configs/nas-benchmark/algos/weight-sharing.config', help="The path to the configuration.")
parser.add_argument("--max_nodes", type=int, default=4, help="The maximum number of nodes.")
parser.add_argument("--channel", type=int, default=16, help="The number of channels.")
parser.add_argument("--num_cells", type=int, default=5, help="The number of cells in one stage.")
parser.add_argument("--affine", type=int, default=1, choices=[0, 1], help="Whether use affine=True or False in the BN layer.")
parser.add_argument("--track_running_stats", type=int, default=0, choices=[0, 1], help="Whether use track_running_stats or not in the BN layer.")

# log
parser.add_argument("--print_freq", type=int, default=200, help="print frequency (default: 200)")

# custom
parser.add_argument("--gpu", type=int, default=0, help="")
parser.add_argument("--workers", type=int, default=4, help="number of data loading workers")
#parser.add_argument("--api_data_path", type=str, default="/mnt/personal/tyblondr/NATS-tss-v1_0-3ffb9-full/data/NATS-tss-v1_0-3ffb9-full", help="")
parser.add_argument("--api_data_path", type=str, default="/mnt/personal/tyblondr/NATS-tss-v1_0-3ffb9-simple/", help="")
parser.add_argument("--save_dir", type=str, default='./results/tmp', help="Folder to save checkpoints and log.")
parser.add_argument('--zero_shot_score', type=str, default='vkdnw', choices=['az_nas','zico','zen','gradnorm','naswot','synflow','snip','grasp','te_nas','gradsign'])
parser.add_argument("--rand_seed", type=int, default=1, help="manual seed (we use 1-to-5)")


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


def search_find_best(xargs, xloader, train_loader, n_samples=None, archs=None):
    logger.log("Searching with {}".format(xargs.zero_shot_score.lower()))
    score_fn_name = "compute_{}_score".format(xargs.zero_shot_score.lower())
    score_fn = globals().get(score_fn_name)
    input_, target_ = next(iter(xloader))
    resolution = input_.size(2)
    batch_size = input_.size(0)
    zero_shot_score_dict = None
    arch_list = []
    if xargs.zero_shot_score.lower() in real_input_metrics:
        print('Use real images as inputs')
        trainloader = train_loader
    else:
        print('Use random inputs')
        trainloader = None

    if archs is None and n_samples is not None:
        all_time = []
        all_mem = []
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        for i in tqdm.tqdm(range(n_samples)):
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            # random sampling
            arch = random_genotype(xargs.max_nodes, search_space)
            network = TinyNetwork(xargs.channel, xargs.num_cells, arch, class_num)
            network = network.to(device)
            network.train()

            start.record()

            info_dict = score_fn.compute_nas_score(network, gpu=xargs.gpu, trainloader=trainloader,
                                                   resolution=resolution, batch_size=batch_size)

            end.record()
            torch.cuda.synchronize()
            all_time.append(start.elapsed_time(end))
            #             all_mem.append(torch.cuda.max_memory_reserved())
            all_mem.append(torch.cuda.max_memory_allocated())

            arch_list.append(arch)
            if zero_shot_score_dict is None:  # initialize dict
                zero_shot_score_dict = dict()
                for k in info_dict.keys():
                    zero_shot_score_dict[k] = []
            for k, v in info_dict.items():
                zero_shot_score_dict[k].append(v)

        logger.log("------Runtime------")
        logger.log("All: {:.5f} ms".format(np.mean(all_time)))
        logger.log("------Avg Mem------")
        logger.log("All: {:.5f} GB".format(np.mean(all_mem) / 1e9))
        logger.log("------Max Mem------")
        logger.log("All: {:.5f} GB".format(np.max(all_mem) / 1e9))

    elif archs is not None and n_samples is None:
        all_time = []
        all_mem = []
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        for arch in tqdm.tqdm(archs):
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            network = TinyNetwork(xargs.channel, xargs.num_cells, arch, class_num)
            network = network.to(device)
            network.train()

            start.record()

            info_dict = score_fn.compute_nas_score(network, gpu=xargs.gpu, trainloader=trainloader,
                                                   resolution=resolution, batch_size=batch_size)

            end.record()
            torch.cuda.synchronize()
            all_time.append(start.elapsed_time(end))
            #             all_mem.append(torch.cuda.max_memory_reserved())
            all_mem.append(torch.cuda.max_memory_allocated())

            arch_list.append(arch)
            if zero_shot_score_dict is None:  # initialize dict
                zero_shot_score_dict = dict()
                for k in info_dict.keys():
                    zero_shot_score_dict[k] = []
            for k, v in info_dict.items():
                zero_shot_score_dict[k].append(v)

        logger.log("------Runtime------")
        logger.log("All: {:.5f} ms".format(np.mean(all_time)))
        logger.log("------Avg Mem------")
        logger.log("All: {:.5f} GB".format(np.mean(all_mem) / 1e9))
        logger.log("------Max Mem------")
        logger.log("All: {:.5f} GB".format(np.max(all_mem) / 1e9))

    return arch_list, zero_shot_score_dict

def generate_all_archs(search_space, xargs):
    arch = random_genotype(xargs.max_nodes, search_space)
    archs = arch.gen_all(search_space, xargs.max_nodes, False)
    return archs

if __name__ == '__main__':

    args = parser.parse_args(args=[])

    if args.rand_seed is None or args.rand_seed < 0:
        args.rand_seed = random.randint(1, 100000)

    print(args.rand_seed)
    print(args)
    xargs = args

    assert torch.cuda.is_available(), "CUDA is not available."
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.set_num_threads(xargs.workers)
    prepare_seed(xargs.rand_seed)
    logger = prepare_logger(args)

    ## API
    api = create(xargs.api_data_path, xargs.search_space, fast_mode=True, verbose=False)
    logger.log("Create API = {:} done".format(api))

    ## data
    train_data, valid_data, xshape, class_num = get_datasets(xargs.dataset, xargs.data_path, -1)
    config = load_config(xargs.config_path, {"class_num": class_num, "xshape": xshape}, logger)
    search_loader, train_loader, valid_loader = get_nas_search_loaders(train_data,
                                                                       valid_data,
                                                                       xargs.dataset,
                                                                       "./configs/nas-benchmark/",
                                                                       (config.batch_size, config.test_batch_size),
                                                                       xargs.workers, )
    logger.log(
        "||||||| {:10s} ||||||| Search-Loader-Num={:}, Valid-Loader-Num={:}, batch size={:}".format(xargs.dataset,
                                                                                                    len(search_loader),
                                                                                                    len(valid_loader),
                                                                                                    config.batch_size))
    logger.log("||||||| {:10s} ||||||| Config={:}".format(xargs.dataset, config))

    ## model
    search_space = get_search_spaces(xargs.search_space, "nats-bench")
    logger.log("search space : {:}".format(search_space))

    device = torch.device('cuda:{}'.format(xargs.gpu))

    real_input_metrics = ['vkdnw', 'zico', 'snip', 'grasp', 'te_nas', 'gradsign']


    if os.path.exists("./tss_all_arch.pickle"):
        with open("./tss_all_arch.pickle", "rb") as fp:
            all_archs = pickle.load(fp)
    else:
        all_archs = generate_all_archs(search_space, xargs)
        with open("./tss_all_arch.pickle", "wb") as fp:
            pickle.dump(all_archs, fp)

    for zero_shot_score in ['vkdnw', 'az_nas', 'gradsign', 'zico', 'zen','gradnorm','naswot','synflow','snip','grasp','te_nas']:
        xargs.zero_shot_score = zero_shot_score
        result_path = "./{}_all_arch.pickle".format(xargs.zero_shot_score)
        if os.path.exists(result_path):
            print("results already exists")
            with open(result_path, "rb") as fp:
                results = pickle.load(fp)
            archs = all_archs
        else:
            archs, results = search_find_best(xargs, train_loader, train_loader, archs=all_archs)
            with open(result_path, "wb") as fp:
                pickle.dump(results, fp)