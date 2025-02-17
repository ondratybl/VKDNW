import os, sys

from functorch import make_functional_with_buffers
from torch.func import functional_call
from scipy.stats import chisquare

from torch.func import jacrev, vmap

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from torch import nn
import numpy as np
import gc


def kaiming_normal_fanin_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
        if hasattr(m, 'bias') and m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
        if m.affine:
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)


def kaiming_normal_fanout_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if hasattr(m, 'bias') and m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
        if m.affine:
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)


def kaiming_uniform_fanin_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
        if hasattr(m, 'bias') and m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
        if m.affine:
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)


def kaiming_uniform_fanout_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight, mode='fan_out', nonlinearity='relu')
        if hasattr(m, 'bias') and m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
        if m.affine:
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)


def xavier_normal_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        if hasattr(m, 'bias') and m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
        if m.affine:
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)


def xavier_uniform_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if hasattr(m, 'bias') and m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
        if m.affine:
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)


def plain_normal_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, mean=0.0, std=0.1)
        if hasattr(m, 'bias') and m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
        if m.affine:
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)


def plain_uniform_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.uniform_(m.weight, a=-0.1, b=0.1)
        if hasattr(m, 'bias') and m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
        if m.affine:
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)


def init_model(model, method='kaiming_norm_fanin'):
    if method == 'kaiming_norm_fanin':
        model.apply(kaiming_normal_fanin_init)
    elif method == 'kaiming_norm_fanout':
        model.apply(kaiming_normal_fanout_init)
    elif method == 'kaiming_uni_fanin':
        model.apply(kaiming_uniform_fanin_init)
    elif method == 'kaiming_uni_fanout':
        model.apply(kaiming_uniform_fanout_init)
    elif method == 'xavier_norm':
        model.apply(xavier_normal_init)
    elif method == 'xavier_uni':
        model.apply(xavier_uniform_init)
    elif method == 'plain_norm':
        model.apply(plain_normal_init)
    elif method == 'plain_uni':
        model.apply(plain_uniform_init)
    else:
        raise NotImplementedError
    return model


def get_fisher(model, input, use_logits=True, params_grad_len=256, p=0.):

    model.eval()

    jacobian = get_jacobian_index(model, input, p, params_grad_len)
    if not use_logits:
        jacobian = torch.matmul(cholesky_covariance(model(input)[1]), jacobian).detach()

    fisher = torch.mean(torch.matmul(torch.transpose(jacobian, dim0=1, dim1=2), jacobian), dim=0).detach()

    del jacobian
    gc.collect()
    torch.cuda.empty_cache()

    return fisher


def cholesky_covariance(output):

    # Cholesky decomposition of covariance matrix (notation from Theorem 1 in https://sci-hub.se/10.2307/2345957)
    alpha = torch.tensor(0.05, dtype=torch.float16, device=output.device)
    prob = torch.nn.functional.softmax(output, dim=1) * (1 - alpha) + alpha / output.shape[1]
    q = torch.ones_like(prob) - torch.cumsum(prob, dim=1)
    q[:, -1] = torch.zeros_like(q[:, -1])
    q_shift = torch.roll(q, shifts=1, dims=1)
    q_shift[:, 0] = torch.ones_like(q_shift[:, 0])
    d = torch.sqrt(prob * q / q_shift)

    L = -torch.matmul(torch.unsqueeze(prob, dim=2), 1 / torch.transpose(torch.unsqueeze(q, dim=2), dim0=1, dim1=2))
    L = torch.nan_to_num(L, neginf=0.)
    L = L * (1 - torch.eye(L.shape[1], device=output.device, dtype=output.dtype).repeat(L.shape[0], 1, 1)) + \
        torch.eye(L.shape[1], device=output.device, dtype=output.dtype).repeat(L.shape[0], 1,
                                                                               1)  # replace diagonal elements by 1.
    L = L * (1 - torch.triu(torch.ones(L.shape[1], L.shape[2], device=output.device, dtype=output.dtype),
                            diagonal=1).repeat(L.shape[0], 1, 1))  # replace upper diagonal by 0
    L = torch.matmul(L, torch.diag_embed(d))  # multiply columns

    # Test
    cov_true = torch.diag_embed(prob) - torch.matmul(torch.unsqueeze(prob, dim=2),
                                                     torch.transpose(torch.unsqueeze(prob, dim=2), dim0=1, dim1=2))
    cov_cholesky = torch.matmul(L, torch.transpose(L, dim0=1, dim1=2))

    max_error = torch.abs(cov_true - cov_cholesky).max().item()
    if max_error > 1.0e-3:
        print(f'Cholesky decomposition back-test error with max error {max_error}')

    return L.detach()


def get_jacobian_index(model, input, p_count, params_grad_len):

    model.zero_grad()

    indices = [
        torch.linspace(0, len(v.flatten()) - 1, steps=int(p_count)).long().to(v.device).tolist()
        for _, v in model.named_parameters()
    ]

    params_grad = {
        k: v.flatten()[param_idx].detach()
        for param_idx, (k, v) in zip(indices, model.named_parameters())
    }
    buffers = {k: v.detach() for k, v in model.named_buffers()}

    if params_grad_len > 0:
        params_grad = dict(list(params_grad.items())[0:params_grad_len])
    else:
        params_grad = dict(list(params_grad.items())[0:(len(params_grad) // -params_grad_len)])

    def jacobian_sample(sample):
        def compute_prediction(params_grad_tmp):
            params = {k: v.detach() for k, v in model.named_parameters()}
            for param_idx, (k, v) in zip(indices, params_grad_tmp.items()):
                param_shape = params[k].shape
                param = params[k].flatten()
                param[param_idx] = v
                params[k] = param.reshape(param_shape)

            return functional_call(model, (params, buffers), (sample.unsqueeze(0),))[1].squeeze(0)

        return jacrev(compute_prediction)(params_grad)

    jacobian_dict = vmap(jacobian_sample)(input)

    ret = torch.cat([torch.flatten(v, start_dim=2, end_dim=-1) for v in jacobian_dict.values()], dim=2)

    return ret.detach()


def compute_nas_score(model, gpu, trainloader, resolution, batch_size, init_method='kaiming_norm_fanin', fp16=False, params_grad_len=128, p=0.):
    model.train()
    model.cuda()
    info = {}
    if gpu is not None:
        device = torch.device('cuda:{}'.format(gpu))
    else:
        device = torch.device('cpu')

    if fp16:
        dtype = torch.half
    else:
        dtype = torch.float32

    init_model(model, init_method)

    if trainloader == None:
        input_ = torch.randn(size=[batch_size, 3, resolution, resolution], device=device, dtype=dtype)
    else:
        input_ = next(iter(trainloader))[0].to(device)

    fisher_prob = get_fisher(model, input_, use_logits=False, params_grad_len=params_grad_len, p=p)

    try:
        lambdas = torch.svd(fisher_prob).S.detach()
    except RuntimeError as e:
        if ("CUDA error: an illegal memory access was encountered" in str(e)) or isinstance(e, torch._C._LinAlgError):
            print(e)
            lambdas = torch.zeros((2,), device=fisher_prob.device)
        else:
            raise  # re-raise the exception if it's not the specific RuntimeError you want to catch

    # Dimenson
    info['vkdnw_dim'] = float(len(list(model.named_parameters())))

    # Eigenvectors
    quantiles = torch.quantile(lambdas, torch.arange(0.1, 1., 0.1, device=lambdas.device))
    temp = quantiles / (torch.linalg.norm(quantiles, ord=1, keepdim=False).item() + 1e-10)
    info.update({'vkdnw_entropy': -(temp * torch.log(temp + 1e-10)).sum().cpu().numpy().item()})
    return info