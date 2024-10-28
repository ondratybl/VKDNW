import os, sys, random, argparse
import numpy as np
import torch
import tqdm
import pickle
import wandb
from torch.utils.data import Dataset, DataLoader

# XAutoDL
from xautodl.config_utils import load_config
from xautodl.datasets import get_datasets, get_nas_search_loaders
from xautodl.procedures import (
    prepare_seed,
    prepare_logger,
)
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
parser.add_argument("--dataset", type=str, default='cifar10', choices=["cifar10", "cifar100", "ImageNet16-120"],
                    help="Choose between Cifar10/100 and ImageNet-16.")

# channels and number-of-cells
parser.add_argument("--search_space", type=str, default='tss', help="The search space name.")
parser.add_argument("--config_path", type=str, default='./configs/nas-benchmark/algos/weight-sharing.config',
                    help="The path to the configuration.")
parser.add_argument("--max_nodes", type=int, default=4, help="The maximum number of nodes.")
parser.add_argument("--channel", type=int, default=16, help="The number of channels.")
parser.add_argument("--num_cells", type=int, default=5, help="The number of cells in one stage.")
parser.add_argument("--affine", type=int, default=1, choices=[0, 1],
                    help="Whether use affine=True or False in the BN layer.")
parser.add_argument("--track_running_stats", type=int, default=0, choices=[0, 1],
                    help="Whether use track_running_stats or not in the BN layer.")

# log
parser.add_argument("--print_freq", type=int, default=200, help="print frequency (default: 200)")

# custom
parser.add_argument("--gpu", type=int, default=0, help="")
parser.add_argument("--workers", type=int, default=4, help="number of data loading workers")
# parser.add_argument("--api_data_path", type=str, default="/mnt/personal/tyblondr/NATS-tss-v1_0-3ffb9-full/data/NATS-tss-v1_0-3ffb9-full", help="")
parser.add_argument("--api_data_path", type=str, default="/mnt/personal/tyblondr/NATS-tss-v1_0-3ffb9-simple/",
                    help="")
parser.add_argument("--save_dir", type=str, default='./results/tmp', help="Folder to save checkpoints and log.")
parser.add_argument("--rand_seed", type=int, default=1, help="manual seed (we use 1-to-5)")
parser.add_argument('--wandb_key', required=True, type=str, help="Wandb key.")
parser.add_argument('--wandb_project', default='VKDNW')
parser.add_argument('--wandb_name', default='VKDNW')
parser.add_argument('--real_input', default=False, action='store_true')
parser.add_argument('--batch_size', type=int, default=64, help="Batch size.")
parser.add_argument('--params_grad_len', type=int, default=128, help="Number of layers to consider.")


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


class RandomDataset(Dataset):
    def __init__(self, num_samples, resolution, rand_seed=None):
        self.num_samples = num_samples
        self.resolution = resolution
        self.rand_seed = rand_seed
        if rand_seed is not None:
            torch.manual_seed(rand_seed)  # Set the seed for reproducibility

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Generate random data with the given resolution
        random_data = torch.rand(3, self.resolution, self.resolution)
        return random_data


def zero_shot_compute(xargs, data_loader, zero_shot_score_list=[], real_input_metrics=[], archs=None):
    all_time = []
    all_mem = []
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for arch in tqdm.tqdm(archs):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        network = TinyNetwork(xargs.channel, xargs.num_cells, arch, class_num)
        network = network.to(device)

        info_dict = {'arch': arch.tostr()}
        for zero_shot_score in zero_shot_score_list:

            start.record()
            score_fn_name = "compute_{}_score".format(zero_shot_score.lower())
            score_fn = globals().get(score_fn_name)

            network.train()
            network.zero_grad()

            if zero_shot_score in real_input_metrics:
                train_loader = data_loader
            else:
                #train_loader = DataLoader(
                #    RandomDataset(2*xargs.batch_size, xargs.resolution, xargs.rand_seed),  # we need two batches for ZEN
                #    batch_size=xargs.batch_size,
                #    shuffle=False,
                #)
                train_loader = None
            if zero_shot_score == 'vkdnw':
                info_dict.update(score_fn.compute_nas_score(
                    network, gpu=xargs.gpu, trainloader=train_loader, resolution=xargs.resolution,
                    batch_size=xargs.batch_size, params_grad_len=xargs.params_grad_len,
                ))
            else:
                info_dict.update(score_fn.compute_nas_score(
                    network, gpu=xargs.gpu, trainloader=train_loader, resolution=xargs.resolution, batch_size=xargs.batch_size,
                ))

            end.record()
            torch.cuda.synchronize()
        wandb.log(info_dict)

        all_time.append(start.elapsed_time(end))
        all_mem.append(torch.cuda.max_memory_allocated())

    logger.log("------Runtime------")
    logger.log("All: {:.5f} ms".format(np.mean(all_time)))
    logger.log("------Avg Mem------")
    logger.log("All: {:.5f} GB".format(np.mean(all_mem) / 1e9))
    logger.log("------Max Mem------")
    logger.log("All: {:.5f} GB".format(np.max(all_mem) / 1e9))


def generate_all_archs(search_space, xargs):
    arch = random_genotype(xargs.max_nodes, search_space)
    archs = arch.gen_all(search_space, xargs.max_nodes, False)
    return archs


if __name__ == '__main__':

    args = parser.parse_args()

    if args.rand_seed is None or args.rand_seed < 0:
        args.rand_seed = random.randint(1, 100000)

    print(args.rand_seed)
    print(args)
    xargs = args

    if xargs.dataset == 'ImageNet16-120':
        xargs.data_path += '/ImageNet16'

    # initialize wandb
    wandb.login(key=xargs.wandb_key)
    wandb.init(project=xargs.wandb_project,
               config=xargs, name=f'{xargs.dataset}_B{xargs.batch_size}_{xargs.wandb_name}',
               tags=['nb201', xargs.dataset, str(xargs.real_input)])

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
                                                                       (xargs.batch_size, xargs.batch_size),
                                                                       xargs.workers, )
    logger.log(
        "||||||| {:10s} ||||||| Search-Loader-Num={:}, Valid-Loader-Num={:}, batch size={:}".format(xargs.dataset,
                                                                                                    len(search_loader),
                                                                                                    len(valid_loader),
                                                                                                    xargs.batch_size))
    logger.log("||||||| {:10s} ||||||| Config={:}".format(xargs.dataset, config))

    search_space = get_search_spaces(xargs.search_space, "nats-bench")
    logger.log("search space : {:}".format(search_space))

    device = torch.device('cuda:{}'.format(xargs.gpu))

    if os.path.exists("./tss_all_arch.pickle"):
        with open("./tss_all_arch.pickle", "rb") as fp:
            archs = pickle.load(fp)
    else:
        archs = generate_all_archs(search_space, xargs)
        with open("./tss_all_arch.pickle", "wb") as fp:
            pickle.dump(archs, fp)

    zero_shot_score_list = ['vkdnw', 'az_nas', 'jacov', 'gradsign', 'zico', 'zen', 'gradnorm', 'naswot', 'synflow', 'snip', 'grasp', 'te_nas']
    if xargs.real_input:
        real_input_metrics = zero_shot_score_list
    else:
        #real_input_metrics = ['zico', 'snip', 'grasp', 'gradsign', 'jacov']  # these need target
        real_input_metrics = ['zico', 'snip', 'grasp', 'te_nas', 'gradsign', 'jacov']  # from AZ-NAS

    xargs.resolution = next(iter(train_loader))[0].size(2)

    zero_shot_compute(
        xargs, train_loader, zero_shot_score_list=zero_shot_score_list, real_input_metrics=real_input_metrics,
        archs=archs,
    )
