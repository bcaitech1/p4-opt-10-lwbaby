import copy
import torch
import torch.nn as nn
import tensorly as tl
from tensorly.decomposition import parafac, partial_tucker
from typing import List

tl.set_backend('pytorch') # switch to the PyTorch backend


def get_tucker_decomposed(model):
    """InvertedResidual 같이 forward가 특수한 구조에 대해서는 사용할 수 없습니다."""
    def _decompose_DFS(module):
        nonlocal layers, curr_seq
        # 말단 모듈을 만난 경우, curr_seq에 저장
        if not module._modules.values():
            # conv 모듈을 만난 경우 (커널 사이즈가 2 이하면 오히려 MACs 증가)
            if (module.__class__ == nn.Conv2d) and (module.kernel_size[-1:][0] > 2):
                conv_layers = tucker_decomposition_conv_layer(module)
                curr_seq.append(conv_layers)
            else:
                curr_seq.append(module)
            return curr_seq
        # 말단 모듈이 아니면 하위 모듈에 대해 DFS 수행
        else:
            # if (module.__class__ == nn.Sequential):
            for sub_module in module._modules.values():
                _decompose_DFS(sub_module)
                # print(sub_module.__class__, len(curr_seq))
                if curr_seq and sub_module._modules.values():
                    layers.append(nn.Sequential(*curr_seq))
                    curr_seq = []
            if curr_seq:
                layers.append(nn.Sequential(*curr_seq))
                curr_seq = []
    
    layers = []
    curr_seq = []
    last_seq = _decompose_DFS(model)
    if last_seq:
        layers.extend(last_seq)
    return nn.Sequential(*layers)

def tucker_decomposition_conv_layer(
        layer: nn.Module,
        normed_rank: List[int] = [0.5, 0.5],
    ) -> nn.Module:
        """Gets a conv layer,
        returns a nn.Sequential object with the Tucker decomposition.
        The ranks are estimated with a Python implementation of VBMF
        https://github.com/CasvandenBogaard/VBMF
        """
        groups = 1
        if layer.groups != 1:
            groups = layer.groups
            sh = [v for v in layer.weight.shape] # [16, 1, 32, 32]
            sh[1] = layer.groups
            layer.weight.data = layer.weight.data.expand(sh) # [16, 16, 32, 32]
            layer.groups = 1
            normed_rank = [r/2 for r in normed_rank]

        if hasattr(layer, "rank"):
            normed_rank = getattr(layer, "rank")
        rank = [int(r * layer.weight.shape[i]) for i, r in enumerate(normed_rank)] # output channel * normalized rank
        rank = [max(r, 2) for r in rank]
        if groups != 1:
            groups = rank[0]

        core, [last, first] = partial_tucker(
            layer.weight.data,
            modes=[0, 1],
            n_iter_max=2000000,
            rank=rank,
            init="svd",
        )

        # A pointwise convolution that reduces the channels from S to R3
        first_layer = nn.Conv2d(
            in_channels=first.shape[0],
            out_channels=first.shape[1],
            kernel_size=1,
            stride=1,
            padding=0,
            dilation=layer.dilation,
            groups=groups,
            bias=False,
        )

        # A regular 2D convolution layer with R3 input channels
        # and R3 output channels
        core_layer = nn.Conv2d(
            in_channels=core.shape[1],
            out_channels=core.shape[0],
            kernel_size=layer.kernel_size,
            stride=layer.stride,
            padding=layer.padding,
            dilation=layer.dilation,
            groups=groups,
            bias=False,
        )

        # A pointwise convolution that increases the channels from R4 to T
        last_layer = nn.Conv2d(
            in_channels=last.shape[1],
            out_channels=last.shape[0],
            kernel_size=1,
            stride=1,
            padding=0,
            dilation=layer.dilation,
            groups=groups,
            bias=True,
        )

        if hasattr(layer, "bias") and layer.bias is not None:
            last_layer.bias.data = layer.bias.data

        first_layer.weight.data = (
            torch.transpose(first, 1, 0).unsqueeze(-1).unsqueeze(-1)
        )
        last_layer.weight.data = last.unsqueeze(-1).unsqueeze(-1)

        if groups != 1:
            w_dim = max(first_layer.in_channels//first_layer.out_channels, 1)
            first_layer.weight.data = first_layer.weight.data[:, :w_dim, :, :]
            w_dim = max(last_layer.in_channels//last_layer.out_channels, 1)
            last_layer.weight.data = last_layer.weight.data[:, :w_dim, :, :]
            w_dim = max(core_layer.in_channels//core_layer.out_channels, 1)
            core_layer.weight.data = core[:, :w_dim, :, :]
        else:
            core_layer.weight.data = core
            
        new_layers = [first_layer, core_layer, last_layer]
        return nn.Sequential(*new_layers)

def tucker_decomposition_dwconv_layer(
        layer: nn.Module,
        normed_rank: int = 0.5,
    ) -> nn.Module:
        rank = int(normed_rank * layer.weight.shape[0])
        rank = max(rank, 2)
        groups = rank

        core, (last, *first) = partial_tucker(
            layer.weight.data,
            modes=[0],
            n_iter_max=2000000,
            rank=[rank],
            init="svd",
        )

        # A pointwise convolution that reduces the channels from S to R3
        first_layer = nn.Conv2d(
            in_channels=first.shape[0],
            out_channels=first.shape[1],
            kernel_size=1,
            stride=1,
            padding=0,
            dilation=layer.dilation,
            groups=groups,
            bias=False,
        )

        core_layer = nn.Conv2d(
            in_channels=groups,#core.shape[0],
            out_channels=groups,#core.shape[0],
            kernel_size=layer.kernel_size,
            stride=layer.stride,
            padding=layer.padding,
            dilation=layer.dilation,
            groups=groups,
            bias=False,
        )

        first_layer.weight.data = (
            torch.transpose(first, 1, 0).unsqueeze(-1).unsqueeze(-1)
        )

        first_layer.weight.data = first_layer.weight.data[:, :groups, :, :]
        core_layer.weight.data = core
        new_layers = [first_layer, core_layer]
        return nn.Sequential(*new_layers)
