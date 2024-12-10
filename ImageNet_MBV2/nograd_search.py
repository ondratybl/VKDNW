'''
Code modified from
ZenNAS: 'https://github.com/idstcv/ZenNAS/blob/d1d617e0352733d39890fb64ea758f9c85b28c1a/evolution_search.py'
ZiCo: 'https://github.com/SLDGroup/ZiCo/blob/3eeb517d51cd447685099c8a4351edee8e31e999/evolution_search.py'
'''

import os, sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append("../../apex/")

import argparse, random, logging, time
import torch
import numpy as np
import global_utils
import Masternet
import time
import wandb
from collections import Counter

from ZeroShotProxy.compute_vkdnw_score import compute_nas_score as compute_nas

import nevergrad as ng

working_dir = os.path.dirname(os.path.abspath(__file__))


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
    parser.add_argument('--budget_latency', type=float, default=None,
                        help='latency of forward inference per mini-batch, e.g., 1e-3 means 1ms.')
    parser.add_argument('--max_layers', type=int, default=18, help='max number of layers of the network.')
    parser.add_argument('--batch_size', type=int, default=32, help='number of instances in one mini-batch.')
    parser.add_argument('--input_image_size', type=int, default=224,
                        help='resolution of input image, usually 32 for CIFAR and 224 for ImageNet.')
    parser.add_argument('--population_size', type=int, default=2048, help='population size of evolution.')
    parser.add_argument('--save_dir', type=str, default='./',
                        help='output directory')
    parser.add_argument('--gamma', type=float, default=1e-2,
                        help='noise perturbation coefficient')
    parser.add_argument('--num_classes', type=int, default=10,
                        help='number of classes')
    parser.add_argument('--dataset', type=str, default='ImageNet16-120',
                        help='type of dataset')
    parser.add_argument('--datapath', type=str, default='../NB201/cifar.python/ImageNet16',
                        help='root of path')
    parser.add_argument('--num_worker', type=int, default=1,
                        help='root of path')
    parser.add_argument('--maxbatch', type=int, default=2,
                        help='root of path')
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--wandb_key', default=None)
    parser.add_argument('--wandb_project', default='VKDNW')
    parser.add_argument('--wandb_name', default='VKDNW_NOGRAD')
    parser.add_argument('--init_net', default=None, type=str, help='init net string')
    parser.add_argument('--opt_budget', default=20, type=int, help='budget of optimization')
    parser.add_argument('--opt_optimizer', default='NgIohTuned', type=str, help='optimizer', choices=['NGOpt', 'NgIohTuned', 'TwoPointsDE', 'PortfolioDiscreteOnePlusOne', 'CMA', 'RandomSearch', 'BayesOptim'])

    module_opt, _ = parser.parse_known_args(argv)
    return module_opt


def get_splitted_structure_str(AnyPlainNet, structure_str, num_classes):
    the_net = AnyPlainNet(num_classes=num_classes, plainnet_struct=structure_str, no_create=True)
    assert hasattr(the_net, 'split')
    splitted_net_str = the_net.split(split_layer_threshold=6)
    return splitted_net_str


def compute_nas_score(AnyPlainNet, random_structure_str, gpu, args):
    the_model = AnyPlainNet(num_classes=args.num_classes, plainnet_struct=random_structure_str,
                            no_create=False, no_reslink=args.search_no_res)
    the_model = the_model.cuda(gpu)

    score_fn_name = "compute_{}_score".format(args.zero_shot_score.lower())
    score_fn = globals().get(score_fn_name)
    info = score_fn.compute_nas_score(model=the_model, gpu=gpu, trainloader=None,
                                      resolution=args.input_image_size,
                                      batch_size=args.batch_size)
    del the_model
    torch.cuda.empty_cache()
    return info


def create_net_str(conv_out, conv_stride, final_out, split_layer_threshold, res_blocks):
    """
    res_blocks: A list of dictionaries where each dictionary contains the parameters
                for a single residual block: {'e': expansion, 'k': kernel_size, 'ch_out': output_channels / 8,
                's': stride, 'b': bottleneck_channels / 8, 'l': num_layers}.
    """
    net_str = f'SuperConvK3BNRELU(3,{conv_out},{conv_stride},1)'

    for i, (_, block) in enumerate(res_blocks.items()):
        if i == 0:
            net_str += f"SuperResIDWE{block['e']}K{block['k']}({conv_out},{block['ch_out']},{block['s']},{block['b']},{block['l']})"
        else:
            net_str += f"SuperResIDWE{block['e']}K{block['k']}({ch_out_previous},{block['ch_out']},{block['s']},{block['b']},{block['l']})"
        ch_out_previous = block['ch_out']

    net_str += f'SuperConvK1BNRELU({ch_out_previous},{final_out},1,1)'

    net = Masternet.MasterNet(num_classes=10, plainnet_struct=net_str, no_create=True, no_reslink=False)
    assert hasattr(net, 'split')
    return net.split(split_layer_threshold=split_layer_threshold)


def loss(conv_out, conv_stride, final_out, split_layer_threshold, res_blocks):

    net_str = create_net_str(conv_out, conv_stride, final_out, split_layer_threshold, res_blocks)

    # Net characteristics
    wandb_log = {}
    net = Masternet.MasterNet(num_classes=1000, plainnet_struct=net_str, no_create=True, no_reslink=False)
    wandb_log['net_str'] = net_str
    wandb_log['flops'] = net.get_FLOPs(224)
    wandb_log['model_size'] = net.get_model_size()
    wandb_log['num_layers'] = net.get_num_layers()

    # Score
    net = Masternet.MasterNet(num_classes=10, plainnet_struct=net_str, no_create=False, no_reslink=False)
    net = net.cuda(0)

    score = compute_nas(model=net, gpu=0, trainloader=None, resolution=32, batch_size=16)['vkdnw']
    del net
    torch.cuda.empty_cache()

    wandb_log['score'] = score
    wandb.log(wandb_log)

    return -score

def main(args):
    import torch.multiprocessing as mp
    mp.set_start_method('spawn')  # Use 'spawn' method for multiprocessing

    gpu = args.gpu
    if gpu is not None:
        print(torch.cuda.device_count())
        torch.cuda.set_device('cuda:{}'.format(gpu))
        # torch.backends.cudnn.benchmark = True
    print(args)

    instrum = ng.p.Instrumentation(
        ng.p.Scalar(init=48, lower=24, upper=144).set_integer_casting(),  # conv_out
        ng.p.Choice([1, 2]),  # conv_stride
        ng.p.Scalar(init=2048, lower=512, upper=4096).set_integer_casting(),  # final_out
        ng.p.Scalar(init=6, lower=3, upper=8).set_integer_casting(),  # split_layer_threshold
        res_blocks=ng.p.Dict(
            res_block1=ng.p.Dict(
                e=ng.p.TransitionChoice([1, 2, 4, 6]),
                k=ng.p.TransitionChoice([3, 5, 7]),
                ch_out=ng.p.Scalar(init=48, lower=24, upper=1024).set_integer_casting(),
                s=ng.p.Choice([1, 2]),
                b=ng.p.Scalar(init=48, lower=24, upper=1024).set_integer_casting(),
                l=ng.p.Scalar(init=1, lower=1, upper=8).set_integer_casting(),
            ),
            res_block2=ng.p.Dict(
                e=ng.p.TransitionChoice([1, 2, 4, 6]),
                k=ng.p.TransitionChoice([3, 5, 7]),
                ch_out=ng.p.Scalar(init=48, lower=24, upper=1024).set_integer_casting(),
                s=ng.p.Choice([1, 2]),
                b=ng.p.Scalar(init=48, lower=24, upper=1024).set_integer_casting(),
                l=ng.p.Scalar(init=1, lower=1, upper=8).set_integer_casting(),
            ),
            res_block3=ng.p.Dict(
                e=ng.p.TransitionChoice([1, 2, 4, 6]),
                k=ng.p.TransitionChoice([3, 5, 7]),
                ch_out=ng.p.Scalar(init=48, lower=24, upper=1024).set_integer_casting(),
                s=ng.p.Choice([1, 2]),
                b=ng.p.Scalar(init=48, lower=24, upper=1024).set_integer_casting(),
                l=ng.p.Scalar(init=3, lower=1, upper=8).set_integer_casting(),
            ),
            res_block4=ng.p.Dict(
                e=ng.p.TransitionChoice([1, 2, 4, 6]),
                k=ng.p.TransitionChoice([3, 5, 7]),
                ch_out=ng.p.Scalar(init=48, lower=24, upper=1024).set_integer_casting(),
                s=ng.p.Choice([1, 2]),
                b=ng.p.Scalar(init=48, lower=24, upper=1024).set_integer_casting(),
                l=ng.p.Scalar(init=4, lower=1, upper=8).set_integer_casting(),
            ),
            res_block5=ng.p.Dict(
                e=ng.p.TransitionChoice([1, 2, 4, 6]),
                k=ng.p.TransitionChoice([3, 5, 7]),
                ch_out=ng.p.Scalar(init=48, lower=24, upper=2048).set_integer_casting(),
                s=ng.p.Choice([1, 2]),
                b=ng.p.Scalar(init=48, lower=24, upper=2048).set_integer_casting(),
                l=ng.p.Scalar(init=5, lower=1, upper=8).set_integer_casting(),
            ),
        )
    )
    optimizer = getattr(ng.optimizers, args.opt_optimizer)(parametrization=instrum, budget=args.opt_budget, num_workers=args.num_worker)
    violation_tracker = Counter()

    def constraint_size(val):
        conv_out, conv_stride, final_out, split_layer_threshold = val[0]
        res_blocks = val[1]['res_blocks']

        net_str = create_net_str(conv_out, conv_stride, final_out, split_layer_threshold, res_blocks)
        net = Masternet.MasterNet(num_classes=1000, plainnet_struct=net_str, no_create=True, no_reslink=False)
        net_flops = net.get_FLOPs(224)
        net_layers = net.get_num_layers()
        net_size = net.get_model_size()

        cond = (net_flops <= args.budget_flops) & (
                    net_layers <= args.max_layers) & (net_size >= args.budget_model_size)

        if not cond:
            violation_tracker["size_violations"] += 1

            if violation_tracker["size_violations"] % 100 == 0:
                print(f"Constraint violations reached {violation_tracker['size_violations']}!")
            return -1.
        else:
            violation_tracker['size_plausible'] += 1

            if violation_tracker["size_plausible"] % 100 == 0:
                print(f"Constraint plausible reached {violation_tracker['size_plausible']}!")
            return 1.

    def constraint_stride(val):
        _, conv_stride, _, _ = val[0]
        res_blocks = val[1]['res_blocks']

        strides_total = conv_stride - 1
        for block in res_blocks.values():
            strides_total += block['s'] - 1
        return float(5 - strides_total)

    optimizer.parametrization.register_cheap_constraint(constraint_size)
    optimizer.parametrization.register_cheap_constraint(constraint_stride)

    if args.num_worker == 1:
        recommendation = optimizer.minimize(loss)
    else:
        from concurrent import futures
        #with futures.ThreadPoolExecutor(max_workers=optimizer.num_workers) as executor:
        with futures.ProcessPoolExecutor(max_workers=optimizer.num_workers) as executor:
            recommendation = optimizer.minimize(loss, executor=executor, batch_mode=False)
    print(recommendation.value)
    print(f'Num. ask: {recommendation.num_ask}')
    print(f'Num. tell: {recommendation.num_tell}')


if __name__ == '__main__':
    args = parse_cmd_options(sys.argv)
    log_fn = os.path.join(args.save_dir, 'evolution_search.log')
    global_utils.create_logging(log_fn)

    wandb.login(key=args.wandb_key)
    wandb.init(project=args.wandb_project, config=args, name=args.wandb_name, tags=['nevergrad_search'])

    if args.seed is not None:
        logging.info("The seed number is set to {}".format(args.seed))
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ["PYTHONHASHSEED"] = str(args.seed)

    main(args)
