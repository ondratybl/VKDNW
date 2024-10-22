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

def get_fisher(model, input, use_logits=True):

    model.eval()

    jacobian = get_jacobian_index(model, input, 0)
    if not use_logits:
        jacobian = torch.matmul(cholesky_covariance(model(input)), jacobian).detach()

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

"""
def get_jacobian_index(model, input, param_idx):
    # Convert model to functional form
    func_model, params, buffers = make_functional_with_buffers(model)

    # Extract the gradient parameter subset
    params_grad = {k: v.flatten()[param_idx:param_idx + 1].detach() for (k, v) in model.named_parameters()}
    params_grad = dict(list(params_grad.items())[:270])

    def jacobian_sample(sample):
        def compute_prediction(params_grad_tmp):
            # Copy the original parameters and modify the specified gradients
            params_copy = [p.clone() for p in params]
            for i, (k, v) in enumerate(params_grad_tmp.items()):
                param_shape = params_copy[i].shape
                param = params_copy[i].flatten()
                param[param_idx:param_idx + 1] = v
                params_copy[i] = param.view(param_shape)

            # Compute the prediction using the functional model
            return func_model(params_copy, buffers, sample.unsqueeze(0)).squeeze(0)

        return jacrev(compute_prediction)(params_grad)

    # Apply vmap to efficiently compute Jacobians for each input in the batch
    jacobian_dict = vmap(jacobian_sample)(input)

    # Concatenate the Jacobian results across parameters
    ret = torch.cat([torch.flatten(v, start_dim=2, end_dim=-1) for v in jacobian_dict.values()], dim=2)

    return ret.detach()
"""

def get_jacobian_index(model, input, param_idx):
    model.zero_grad()

    params_grad = {k: v.flatten()[param_idx:param_idx+1].detach() for k, v in model.named_parameters()}
    buffers = {k: v.detach() for k, v in model.named_buffers()}
    params_grad = dict(list(params_grad.items())[0:100])

    def jacobian_sample(sample):
        def compute_prediction(params_grad_tmp):
            params = {k: v.detach() for k, v in model.named_parameters()}
            for k, v in params_grad_tmp.items():
                param_shape = params[k].shape
                param = params[k].flatten()
                param[param_idx:param_idx+1] = v
                params[k] = param.reshape(param_shape)

            return functional_call(model, (params, buffers), (sample.unsqueeze(0),)).squeeze(0)

        return jacrev(compute_prediction)(params_grad)

    jacobian_dict = vmap(jacobian_sample)(input)

    ret = torch.cat([torch.flatten(v, start_dim=2, end_dim=-1) for v in jacobian_dict.values()], dim=2)

    return ret.detach()


def compute_nas_score(model, gpu, trainloader, resolution, batch_size, init_method='kaiming_norm_fanin', fp16=False):
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

    if trainloader is None:
        input_ = torch.randn(size=[batch_size, 3, resolution, resolution], device=device, dtype=dtype)
    else:
        input_ = next(iter(trainloader))
        if type(input_) == list:
            input_ = input_[0]
    if gpu is not None:
        input_ = input_.clone().cuda(device=device, non_blocking=True)
    else:
        input_ = input_.clone()


    if model.no_reslink:
        layer_features = model.extract_layer_features_nores(input_)
    else:
        layer_features, output = model.extract_layer_features_and_logit(input_)

    ################ expressivity & progressivity scores ################
    expressivity_scores = []
    for i in range(len(layer_features)):
        feat = layer_features[i].detach().clone()
        b, c, h, w = feat.size()
        feat = feat.permute(0, 2, 3, 1).contiguous().view(b * h * w, c)
        m = feat.mean(dim=0, keepdim=True)
        feat = feat - m
        sigma = torch.mm(feat.transpose(1, 0), feat) / (feat.size(0))
        s = torch.linalg.eigvalsh(
            sigma)  # faster version for computing eignevalues, can be adopted since sigma is symmetric
        prob_s = s / s.sum()
        score = (-prob_s) * torch.log(prob_s + 1e-8)
        score = score.sum().item()
        expressivity_scores.append(score)
    expressivity_scores = np.array(expressivity_scores)
    progressivity = np.min(expressivity_scores[1:] - expressivity_scores[:-1])
    expressivity = np.sum(expressivity_scores)
    #####################################################################

    ################ trainability score ##############
    scores = []
    for i in reversed(range(1, len(layer_features))):
        f_out = layer_features[i]
        f_in = layer_features[i - 1]
        if f_out.grad is not None:
            f_out.grad.zero_()
        if f_in.grad is not None:
            f_in.grad.zero_()

        g_out = torch.ones_like(f_out) * 0.5
        g_out = (torch.bernoulli(g_out) - 0.5) * 2
        g_in = torch.autograd.grad(outputs=f_out, inputs=f_in, grad_outputs=g_out, retain_graph=False)[0]
        if g_out.size() == g_in.size() and torch.all(g_in == g_out):
            scores.append(-np.inf)
        else:
            if g_out.size(2) != g_in.size(2) or g_out.size(3) != g_in.size(3):
                bo, co, ho, wo = g_out.size()
                bi, ci, hi, wi = g_in.size()
                stride = int(hi / ho)
                pixel_unshuffle = nn.PixelUnshuffle(stride)
                g_in = pixel_unshuffle(g_in)
            bo, co, ho, wo = g_out.size()
            bi, ci, hi, wi = g_in.size()
            ### straight-forward way
            # g_out = g_out.permute(0,2,3,1).contiguous().view(bo*ho*wo,1,co)
            # g_in = g_in.permute(0,2,3,1).contiguous().view(bi*hi*wi,ci,1)
            # mat = torch.bmm(g_in,g_out).mean(dim=0)
            ### efficient way # print(torch.allclose(mat, mat2, atol=1e-6))
            g_out = g_out.permute(0, 2, 3, 1).contiguous().view(bo * ho * wo, co)
            g_in = g_in.permute(0, 2, 3, 1).contiguous().view(bi * hi * wi, ci)
            mat = torch.mm(g_in.transpose(1, 0), g_out) / (bo * ho * wo)
            ### make faster on cpu
            if mat.size(0) < mat.size(1):
                mat = mat.transpose(0, 1)
            ###
            s = torch.linalg.svdvals(mat)
            scores.append(-s.max().item() - 1 / (s.max().item() + 1e-6) + 2)
    trainability = np.mean(scores)
    #################################################

    info['expressivity'] = float(expressivity) if not np.isnan(expressivity) else -np.inf
    info['progressivity'] = float(progressivity) if not np.isnan(progressivity) else -np.inf
    info['trainability'] = float(trainability) if not np.isnan(trainability) else -np.inf
    info['complexity'] = float(model.get_FLOPs(resolution))

    fisher_prob = get_fisher(model, input_, use_logits=False)

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

    #Chisquare
    bin_edges = [1]
    for i in range(18):
        bin_edges.append(bin_edges[-1] / 10)
    bin_edges.append(0)
    bin_edges = bin_edges[::-1]

    f_obs, _ = np.histogram(lambdas.cpu().numpy(), bins=bin_edges)
    info['vkdnw_chisquare'] = chisquare(f_obs).statistic

    # Eigenvectors
    quantiles = torch.quantile(lambdas, torch.arange(0.1, 1.0, 0.1, device=lambdas.device))
    info.update({'vkdnw_lambda_' + str(i): v.item() for (i, v) in enumerate(quantiles)})
    return info