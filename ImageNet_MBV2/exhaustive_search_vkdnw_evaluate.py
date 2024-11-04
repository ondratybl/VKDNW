'''
Code modified from
ZenNAS: 'https://github.com/idstcv/ZenNAS/blob/d1d617e0352733d39890fb64ea758f9c85b28c1a/evolution_search.py'
ZiCo: 'https://github.com/SLDGroup/ZiCo/blob/3eeb517d51cd447685099c8a4351edee8e31e999/evolution_search.py'
'''

import os, sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append("../../apex/")

import argparse, random, logging
import torch
from torch import nn
import numpy as np
import global_utils
import Masternet
import wandb
from ZeroShotProxy.compute_vkdnw_score import compute_nas_score

working_dir = os.path.dirname(os.path.abspath(__file__))


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def none_or_int(value):
    if value.lower() == 'none':
        return None
    return int(value)


def parse_cmd_options(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--search_space', type=str, default='SearchSpace/search_space_IDW_fixfc.py',
                        help='.py file to specify the search space.')
    parser.add_argument('--evolution_max_iter', type=int, default=int(100000),
                        help='max iterations of evolution.')
    parser.add_argument('--budget_model_size', type=float, default=None,
                        help='budget of model size ( number of parameters), e.g., 1e6 means 1M params')
    parser.add_argument('--budget_flops', type=float, default=450e6,
                        help='budget of flops, e.g. , 1.8e6 means 1.8 GFLOPS')
    parser.add_argument('--max_layers', type=int, default=20, help='max number of layers of the network.')
    parser.add_argument('--batch_size', type=int, default=16, help='number of instances in one mini-batch.')
    parser.add_argument('--input_image_size', type=int, default=224,
                        help='resolution of input image, usually 32 for CIFAR and 224 for ImageNet.')
    parser.add_argument('--save_dir', type=str, default='./',
                        help='output directory')
    parser.add_argument('--dataset', type=str, default='ImageNet16-120',
                        help='type of dataset')
    parser.add_argument('--datapath', type=str, default='../NB201/cifar.python/ImageNet16',
                        help='root of path')
    parser.add_argument('--num_worker', type=int, default=40,
                        help='root of path')
    parser.add_argument('--maxbatch', type=int, default=2,
                        help='root of path')
    parser.add_argument('--search_no_res', type=str2bool, default=False, help='remove residual link in search phase')
    parser.add_argument('--seed', type=none_or_int, default=123)
    parser.add_argument('--wandb_key', default='109a132addff7ecca7b2a99e1126515e5fa66377')
    parser.add_argument('--wandb_project', default='VKDNW')
    parser.add_argument('--wandb_name', default='VKDNW_EXHAUSTIVE_EVALUATE')

    module_opt, _ = parser.parse_known_args(argv)
    return module_opt


def main(args):

    AnyPlainNet = Masternet.MasterNet

    import pandas as pd
    import wandb
    api_wandb = wandb.Api()
    runs_feasible = None
    for run_str in [
        'nazderaze/VKDNW/le027c6i',
        'nazderaze/VKDNW/e5ghgrna',
        'nazderaze/VKDNW/9t4z9soy',
        'nazderaze/VKDNW/xu12jwh3',
        'nazderaze/VKDNW/p4w49gbw',
        'nazderaze/VKDNW/8o8nv1bw',
        'nazderaze/VKDNW/a2ohrrxp',
        'nazderaze/VKDNW/uk275wky',
        'nazderaze/VKDNW/5v9c0ueh',
        'nazderaze/VKDNW/vt926kqu',
    ]:
        run = pd.DataFrame(api_wandb.run(run_str).scan_history())

        if runs_feasible is None:
            runs_feasible = run
        else:
            runs_feasible = pd.concat([runs_feasible, run], ignore_index=True)

    for net_str in runs_feasible['net_str']:

        net = AnyPlainNet(num_classes=10, plainnet_struct=net_str, no_create=False, no_reslink=False).cuda(args.gpu)
        # check the model size
        the_model_flops = net.get_FLOPs(args.input_image_size)
        the_model_layers = net.get_num_layers()
        the_model_size = net.get_model_size()
        if len(list(net.parameters())) < 128:
            print(f'Small VKDNW_dim {len(list(net.parameters()))}.')
            continue
        print(f'FLOPS rate: {(the_model_flops / args.budget_flops):.2f}, layers: {the_model_layers}')
        if args.max_layers <= the_model_layers:
            print('Something is wrong with layers.')
            continue
        if (the_model_flops < args.budget_flops*0.9) or (the_model_flops > args.budget_flops*1.1):
            print('Something is wrong with flops.')
            continue

        # compute proxy
        wandb_log = compute_nas_score(net, args.gpu, None, args.input_image_size, args.batch_size)

        del net
        torch.cuda.empty_cache()

        # log
        wandb_log['net_str'] = net_str
        wandb_log['flops'] = the_model_flops
        wandb_log['model_size'] = the_model_size
        wandb_log['num_layers'] = the_model_layers
        wandb.log(wandb_log)


if __name__ == '__main__':

    args = parse_cmd_options(sys.argv)
    log_fn = os.path.join(args.save_dir, 'evolution_search.log')
    global_utils.create_logging(log_fn)

    logging.info("The seed number is set to {}".format(args.seed))
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(args.seed)
    torch.cuda.set_device('cuda:{}'.format(args.gpu))

    wandb.login(key=args.wandb_key)
    wandb.init(project=args.wandb_project, config=args, name=args.wandb_name, tags=['exhaustive_search'])

    print(args)
    main(args)
