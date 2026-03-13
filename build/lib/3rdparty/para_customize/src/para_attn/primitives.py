from typing import List, Optional, Tuple, Union

import torch
import torch.distributed as dist

if dist.is_available():
    import torch.distributed._functional_collectives as ft_c
    import torch.distributed.distributed_c10d as c10d
else:
    ft_c = None
    c10d = None


def get_group(group=None):
    if group is None:
        group = c10d._get_default_group()

    if isinstance(group, dist.ProcessGroup):
        pg: Union[dist.ProcessGroup, List[dist.ProcessGroup]] = group
    else:
        pg = group.get_group()

    return pg


def get_world_size(group=None):
    pg = get_group(group)
    return dist.get_world_size(pg)


def get_rank(group=None):
    pg = get_group(group)
    return dist.get_rank(pg)


def _maybe_wait(tensor: torch.Tensor) -> torch.Tensor:
    """
    When tracing the code, the result tensor is not an AsyncCollectiveTensor,
    so we cannot call ``wait()``.
    """
    if isinstance(tensor, ft_c.AsyncCollectiveTensor):
        return tensor.wait()
    return tensor


def all_gather_tensor_sync(x, *args, group=None, **kwargs):
    group = get_group(group)
    x_shape = x.shape
    x = x.flatten()
    x_numel = x.numel()
    x = ft_c.all_gather_tensor(x, *args, group=group, **kwargs)
    x = _maybe_wait(x)
    x_shape = list(x_shape)
    x_shape[0] *= x.numel() // x_numel
    x = x.reshape(x_shape)
    return x


def all_gather_tensor_autograd_sync(x, *args, group=None, **kwargs):
    group = get_group(group)
    x_shape = x.shape
    x = x.flatten()
    x_numel = x.numel()
    x = ft_c.all_gather_tensor_autograd(x, *args, group=group, **kwargs)
    x = _maybe_wait(x)
    x_shape = list(x_shape)
    x_shape[0] *= x.numel() // x_numel
    x = x.reshape(x_shape)
    return x


def all_to_all_single_sync(x, *args, **kwargs):
    x_shape = x.shape
    x = x.flatten()
    x = ft_c.all_to_all_single(x, *args, **kwargs)
    x = _maybe_wait(x)
    x = x.reshape(x_shape)
    return x


def all_to_all_single_autograd_sync(x, *args, **kwargs):
    x_shape = x.shape
    x = x.flatten()
    x = ft_c.all_to_all_single_autograd(x, *args, **kwargs)
    x = _maybe_wait(x)
    x = x.reshape(x_shape)
    return x


def all_reduce_sync(x, *args, group=None, **kwargs):
    group = get_group(group)
    x = ft_c.all_reduce(x, *args, group=group, **kwargs)
    x = _maybe_wait(x)
    return x


def get_buffer(
    shape_or_tensor: Union[Tuple[int], torch.Tensor],
    *,
    repeats: int = 1,
    dim: int = 0,
    dtype: Optional[torch.dtype] = None,
    device: Optional[torch.device] = None,
    group=None,
) -> torch.Tensor:
    if repeats is None:
        repeats = get_world_size(group)

    if isinstance(shape_or_tensor, torch.Tensor):
        shape = shape_or_tensor.shape
        dtype = shape_or_tensor.dtype
        device = shape_or_tensor.device

    assert dtype is not None
    assert device is not None

    shape = list(shape)
    if repeats > 1:
        shape[dim] *= repeats

    buffer = torch.empty(shape, dtype=dtype, device=device)
    return buffer


def get_assigned_chunk(
    tensor: torch.Tensor,
    dim: int = 0,
    idx: Optional[int] = None,
    group=None,
) -> torch.Tensor:
    if idx is None:
        idx = get_rank(group)
    world_size = get_world_size(group)
    total_size = tensor.shape[dim]
    # # 计算能被 world_size 整除的最大尺寸
    # divisible_size = (total_size // world_size) * world_size
    
    # # 如果有余数，截断 tensor
    # if divisible_size < total_size:
    #     # 构造 slice 对象，在 dim 维度上截取 [0:divisible_size]
    #     slices = [slice(None)] * tensor.ndim
    #     slices[dim] = slice(0, divisible_size)
    #     tensor = tensor[tuple(slices)]
    
    assert total_size % world_size == 0, f"tensor.shape[{dim}]={total_size} is not divisible by world_size={world_size}"
    # chunk_size = (total_size + world_size - 1) // world_size
    # pad_size = chunk_size * world_size - total_size
    # if pad_size > 0:
    #     # 在 dim 维度后面 pad
    #     pad_shape = list(tensor.shape)
    #     pad_shape[dim] = pad_size
    #     pad_tensor = torch.zeros(pad_shape, dtype=tensor.dtype, device=tensor.device)
    #     tensor = torch.cat([tensor, pad_tensor], dim=dim)
    return tensor.chunk(world_size, dim=dim)[idx]


def get_complete_tensor(
    tensor: torch.Tensor,
    *,
    dim: int = 0,
    group=None,
) -> torch.Tensor:
    tensor = tensor.transpose(0, dim).contiguous()
    output_tensor = all_gather_tensor_sync(tensor, gather_dim=0, group=group)
    output_tensor = output_tensor.transpose(0, dim)
    return output_tensor


def aggregate_hybrid_density(local_density, ulysses_pg, ring_pg):
    """
    将混合切分的 head_density 聚合成全局 tensor。
    
    Args:
        local_density: [B, H_local, S_Q_local, S_K_global] 
                       例如你的例子: [1, 10, 296, 592]
        ulysses_pg: Ulysses 进程组 (负责 Head 维度)
        ring_pg: Ring 进程组 (负责 Sequence 维度)
        
    Returns:
        global_density: [B, H_global, S_Q_global, S_K_global]
                        例如你的目标: [1, 40, 592, 592]
    """
    # 1. Ulysses 聚合: 沿着 Head 维度 (dim=1) 拼接
    # 输入: [B, 10, 296, 592] -> 输出: [B, 40, 296, 592]
    density_stage1 = _all_gather_and_cat(local_density, dim=1, group=ulysses_pg)
    
    # 2. Ring 聚合: 沿着 Query Sequence 维度 (dim=2) 拼接
    # 输入: [B, 40, 296, 592] -> 输出: [B, 40, 592, 592]
    global_density = _all_gather_and_cat(density_stage1, dim=2, group=ring_pg)
    
    return global_density

def _all_gather_and_cat(tensor, dim, group):
    """辅助函数: 指定维度的 All-Gather"""
    if group is None or dist.get_world_size(group) == 1:
        return tensor
        
    world_size = dist.get_world_size(group)
    # 创建收集列表
    tensor_list = [torch.empty_like(tensor) for _ in range(world_size)]
    
    # 执行通信
    dist.all_gather(tensor_list, tensor, group=group)
    
    # 拼接
    return torch.cat(tensor_list, dim=dim)