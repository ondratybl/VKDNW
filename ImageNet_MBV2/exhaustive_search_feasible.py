import os, sys

import argparse, random, time
import numpy as np
import Masternet
import wandb
import torch
import concurrent.futures

working_dir = os.path.dirname(os.path.abspath(__file__))


def parse_cmd_options(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--search_space', type=str, default='SearchSpace/search_space_IDW_fixfc.py',
                        help='.py file to specify the search space.')
    parser.add_argument('--budget_model_size', type=float, default=None,
                        help='budget of model size ( number of parameters), e.g., 1e6 means 1M params')
    parser.add_argument('--budget_flops', type=float, default=1000e6,
                        help='budget of flops, e.g. , 1.8e6 means 1.8 GFLOPS')
    parser.add_argument('--max_layers', type=int, default=19, help='max number of layers of the network.')
    parser.add_argument('--input_image_size', type=int, default=224,
                        help='resolution of input image, usually 32 for CIFAR and 224 for ImageNet.')
    parser.add_argument('--search_no_res', default=False, action='store_true', help='remove residual link in search phase')
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--wandb_key', default=None, type=str,)
    parser.add_argument('--wandb_project', default='VKDNW')
    parser.add_argument('--wandb_name', default='VKDNW_FEASIBLE')

    module_opt, _ = parser.parse_known_args(argv)
    return module_opt


def get_flops_distribution(net, resolution):

    flops_distribution = []
    for block in net.block_list:
        flops_distribution.append(block.get_FLOPs(resolution))
        resolution = block.get_output_resolution(resolution)

    flops_total = sum(flops_distribution)
    return [flops / flops_total for flops in flops_distribution]


def task(cpu_id, seed, budget_flops, input_image_size, max_layers):

    final_seed = seed*10000+cpu_id

    random.seed(final_seed)
    np.random.seed(final_seed)
    torch.manual_seed(final_seed)
    os.environ["PYTHONHASHSEED"] = str(final_seed)

    print(f'CPU {cpu_id}, final seed {final_seed}, FLOPS {budget_flops}, input image size {input_image_size}, max layers {max_layers}')

    AnyPlainNet = Masternet.MasterNet

    archs_all = []
    archs = []
    unsucessfull = 0
    start_time = time.time()
    while True:

        # propose net
        strides_total = 0
        net_len = random.choice([5, 6])
        out_channels = 8 * random.randint(4, 16)
        stride = random.choice([1, 2])
        if stride == 2:
            strides_total += 1
        net_str = f'SuperConvK3BNRELU(3,{out_channels},{stride},1)'
        for j in range(net_len):
            in_channels = out_channels
            out_channels = random.randint(int(in_channels * 0.9), min(2048, int(in_channels * 2.5)))
            layer_type = 3  # random.choice([1, 2, 3])
            kernel_size = random.choice([3, 5, 7])
            if strides_total > 6:
                stride = 1
            else:
                stride = random.choice([1, 2])
                if stride == 2:
                    strides_total += 1
            sub_layers = random.choices([1, 2, 3, 5], k=1)[0]
            bottleneck_channels = random.choice(range(int(in_channels / 2), min(2048, int(out_channels * 2))))
            if layer_type == 1:
                net_str += f'SuperResK{kernel_size}K{kernel_size}({in_channels},{out_channels},{stride},{bottleneck_channels},{sub_layers})'
            elif layer_type == 2:
                net_str += f'SuperResK1K{kernel_size}K1({in_channels},{out_channels},{stride},{bottleneck_channels},{sub_layers})'
            else:
                expansion = random.choice([1, 2, 4, 6])
                net_str += f'SuperResIDWE{expansion}K{kernel_size}({in_channels},{out_channels},{stride},{bottleneck_channels},{sub_layers})'
        net_str += f'SuperConvK1BNRELU({out_channels},2048,1,1)'

        # check if already computed
        if net_str in archs_all:
            continue
        archs_all.append(net_str)

        net = AnyPlainNet(num_classes=1000, plainnet_struct=net_str, no_create=False, no_reslink=False)
        # check the model size
        the_model_flops = net.get_FLOPs(input_image_size)
        if (the_model_flops < budget_flops*0.8) or (the_model_flops > budget_flops*1.2):
            unsucessfull += 1
            continue
        the_model_layers = net.get_num_layers()
        if the_model_layers > max_layers:
            unsucessfull += 1
            continue
        the_model_size = net.get_model_size()

        del net

        # log
        archs.append(net_str)
        wandb.log({
            'net_str': net_str,
            'flops': the_model_flops,
            'layers': the_model_layers,
            'size': the_model_size,
            'cpu_id': cpu_id,
            'final_seed': final_seed,
        })
        print(f'CPU {cpu_id}: no. of archs: {len(archs)}. Previously unsucessfull: {unsucessfull}. Took {time.time() - start_time:.2f}s.')
        unsucessfull = 0
        start_time = time.time()

    return archs


if __name__ == '__main__':

    args = parse_cmd_options(sys.argv)

    wandb.login(key=args.wandb_key)
    wandb.init(project=args.wandb_project, config=args, name=args.wandb_name, tags=['exhaustive_feasible'])

    cpu_count = 6  # 6CPUs = 2s/net; 2CPUs = 1s/net

    with concurrent.futures.ProcessPoolExecutor(max_workers=cpu_count) as executor:
        futures = [executor.submit(task, cpu_id, args.seed, args.budget_flops, args.input_image_size, args.max_layers) for cpu_id in range(cpu_count)]
        results = [future.result() for future in concurrent.futures.as_completed(futures)]

