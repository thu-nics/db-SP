import inspect
import math
import logging
from typing import Optional
# from SageAttention.sageattention.triton import attn_qk_int8_block_varlen
import omegaconf
from omegaconf import OmegaConf

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.models.attention import Attention
import torch.distributed as dist
# from paroattention.core import paroattn
# import paroattention._qattn_sm80 as qattn
# from paroattention.quant import per_block_int8 as per_block_int8_cuda
# from paroattention.quant import per_warp_int8 as per_warp_int8_cuda
# import sageattention._qattn_sm80 as qattn_sage
from spas_sage_attn import customize_spas_sage_attn_meansim_cuda
# from spas_sage_attn import block_sparse_sage2_attn_cuda
from paroattention import paroattn_qk_int8_pv_fp16_cuda
import torch.cuda as cuda
# import plotly.subplots as sp
# import plotly.graph_objects as go
from torch.profiler import profile, record_function, ProfilerActivity

def all_to_all_4D(
    input: torch.tensor, scatter_idx: int = 2, gather_idx: int = 1, group=None, use_sync: bool = False
) -> torch.tensor:
    """
    all-to-all for QKV

    Args:
        input (torch.tensor): a tensor sharded along dim scatter dim
        scatter_idx (int): default 1
        gather_idx (int): default 2
        group : torch process group
        use_sync (bool): whether to synchronize after all-to-all

    Returns:
        torch.tensor: resharded tensor (bs, seqlen/P, hc, hs)
    """
    assert (
        input.dim() == 4
    ), f"input must be 4D tensor, got {input.dim()} and shape {input.shape}"

    seq_world_size = dist.get_world_size(group)

    if scatter_idx == 2 and gather_idx == 1:
        # input (torch.tensor): a tensor sharded along dim 1 (bs, seqlen/P, hc, hs) output: (bs, seqlen, hc/P, hs)
        bs, shard_seqlen, hc, hs = input.shape
        seqlen = shard_seqlen * seq_world_size
        shard_hc = hc // seq_world_size

        # transpose groups of heads with the seq-len parallel dimension, so that we can scatter them!
        # (bs, seqlen/P, hc, hs) -reshape-> (bs, seq_len/P, P, hc/P, hs) -transpose(0,2)-> (P, seq_len/P, bs, hc/P, hs)
        input_t = (
            input.reshape(bs, shard_seqlen, seq_world_size, shard_hc, hs)
            .transpose(0, 2)
            .contiguous()
        )

        output = torch.empty_like(input_t)
        # https://pytorch.org/docs/stable/distributed.html#torch.distributed.all_to_all_single
        # (P, seq_len/P, bs, hc/P, hs) scatter seqlen -all2all-> (P, seq_len/P, bs, hc/P, hs) scatter head

        if seq_world_size > 1:
            dist.all_to_all_single(output, input_t, group=group) # , async_op=True
            if use_sync:
                torch.cuda.synchronize()
        else:
            output = input_t
        # if scattering the seq-dim, transpose the heads back to the original dimension
        output = output.reshape(seqlen, bs, shard_hc, hs)

        # (seq_len, bs, hc/P, hs) -reshape-> (bs, seq_len, hc/P, hs)
        output = output.transpose(0, 1).contiguous().reshape(bs, seqlen, shard_hc, hs)

        return output

    elif scatter_idx == 1 and gather_idx == 2:
        # input (torch.tensor): a tensor sharded along dim 1 (bs, seqlen, hc/P, hs) output: (bs, seqlen/P, hc, hs)
        bs, seqlen, shard_hc, hs = input.shape
        hc = shard_hc * seq_world_size
        shard_seqlen = seqlen // seq_world_size
        seq_world_size = dist.get_world_size(group)

        # transpose groups of heads with the seq-len parallel dimension, so that we can scatter them!
        # (bs, seqlen, hc/P, hs) -reshape-> (bs, P, seq_len/P, hc/P, hs) -transpose(0, 3)-> (hc/P, P, seqlen/P, bs, hs) -transpose(0, 1) -> (P, hc/P, seqlen/P, bs, hs)
        input_t = (
            input.reshape(bs, seq_world_size, shard_seqlen, shard_hc, hs)
            .transpose(0, 3)
            .transpose(0, 1)
            .contiguous()
            .reshape(seq_world_size, shard_hc, shard_seqlen, bs, hs)
        )

        output = torch.empty_like(input_t)
        # https://pytorch.org/docs/stable/distributed.html#torch.distributed.all_to_all_single
        # (P, bs x hc/P, seqlen/P, hs) scatter seqlen -all2all-> (P, bs x seq_len/P, hc/P, hs) scatter head
        if seq_world_size > 1:
            dist.all_to_all_single(output, input_t, group=group)
            if use_sync:
                torch.cuda.synchronize()
        else:
            output = input_t

        # if scattering the seq-dim, transpose the heads back to the original dimension
        output = output.reshape(hc, shard_seqlen, bs, hs)

        # (hc, seqlen/N, bs, hs) -tranpose(0,2)-> (bs, seqlen/N, hc, hs)
        output = output.transpose(0, 2).contiguous().reshape(bs, shard_seqlen, hc, hs)

        return output
    else:
        raise RuntimeError("scatter_idx must be 1 or 2 and gather_idx must be 1 or 2")


class PARO_PARALLEL_WanAttnProcessor2_0:
    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("WanAttnProcessor2_0 requires PyTorch 2.0. To use it, please upgrade PyTorch to 2.0.")
        self.events = {
            'total_start': cuda.Event(enable_timing=True),
            'total_end': cuda.Event(enable_timing=True),
        }
        self.time_accum = {
            'total': 0.0,
        }
        self.call_count = 0

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        sparse: Optional[torch.Tensor] = None,
        head_perm_idx:Optional[torch.Tensor]=None,
        head_deperm_idx:Optional[torch.Tensor]=None,
        new_row_perm_idx:Optional[torch.Tensor]=None,
        new_col_perm_idx:Optional[torch.Tensor]=None,
        transpose_matrix_q:Optional[torch.Tensor]=None,
        transpose_matrix_k:Optional[torch.Tensor]=None,
        new_row_deperm_idx:Optional[torch.Tensor]=None,
        ulysses_pg: Optional[torch.distributed.ProcessGroup] = None,
        ring_pg: Optional[torch.distributed.ProcessGroup] = None,

        rotary_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        encoder_hidden_states_img = None
        if attn.add_k_proj is not None:
            encoder_hidden_states_img = encoder_hidden_states[:, :257]
            encoder_hidden_states = encoder_hidden_states[:, 257:]
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states

        query = attn.to_q(hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        # print(torch.cuda.memory_allocated() / 1024**2, "MB allocated after qkv")

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        query = query.unflatten(2, (attn.heads, -1)).transpose(1, 2)
        key = key.unflatten(2, (attn.heads, -1)).transpose(1, 2)
        value = value.unflatten(2, (attn.heads, -1)).transpose(1, 2)

        if rotary_emb is not None:

            def apply_rotary_emb(hidden_states: torch.Tensor, freqs: torch.Tensor):
                x_rotated = torch.view_as_complex(hidden_states.to(torch.float64).unflatten(3, (-1, 2)))
                x_out = torch.view_as_real(x_rotated * freqs).flatten(3, 4)
                return x_out.type_as(hidden_states)

            query = apply_rotary_emb(query, rotary_emb)
            key = apply_rotary_emb(key, rotary_emb)

        # I2V task
        hidden_states_img = None
        if encoder_hidden_states_img is not None:
            key_img = attn.add_k_proj(encoder_hidden_states_img)
            key_img = attn.norm_added_k(key_img)
            value_img = attn.add_v_proj(encoder_hidden_states_img)

            key_img = key_img.unflatten(2, (attn.heads, -1)).transpose(1, 2)
            value_img = value_img.unflatten(2, (attn.heads, -1)).transpose(1, 2)

            hidden_states_img = F.scaled_dot_product_attention(
                query, key_img, value_img, attn_mask=None, dropout_p=0.0, is_causal=False
            )
            hidden_states_img = hidden_states_img.transpose(1, 2).flatten(2, 3)
            hidden_states_img = hidden_states_img.type_as(query)

        """
        INFO: the PAROAttention, replace the original scaled_dot_product_attention with PAROAttention.
        """
        # support prefetch and not.
        # permute_qk(query, key, value, self.permute_plan, self.i_block)

        self.events['total_start'].record()
        
        
        # with profile(
        #     activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        #     record_shapes=True,
        #     profile_memory=True,
        #     with_stack=True
        # ) as prof:
            
        # head-wise reorder [batch_size, num_heads, seq_len, head_dim]
        if head_perm_idx is not None and head_deperm_idx is not None:
            query = query.index_select(1, head_perm_idx)
            query = all_to_all_4D(query, scatter_idx=1, gather_idx=2, group=ulysses_pg)
            key = key.index_select(1, head_perm_idx)
            key = all_to_all_4D(key, scatter_idx=1, gather_idx=2, group=ulysses_pg)
            value = value.index_select(1, head_perm_idx)
            value = all_to_all_4D(value, scatter_idx=1, gather_idx=2, group=ulysses_pg)
            sparse = all_to_all_4D(sparse, scatter_idx=1, gather_idx=2, group=ulysses_pg)

        else: 
            query = all_to_all_4D(query, scatter_idx=1, gather_idx=2, group=ulysses_pg)
            key = all_to_all_4D(key, scatter_idx=1, gather_idx=2, group=ulysses_pg )
            value = all_to_all_4D(value, scatter_idx=1, gather_idx=2, group=ulysses_pg)
            sparse = all_to_all_4D(sparse, scatter_idx=1, gather_idx=2, group=ulysses_pg)

        cuda.synchronize()
        rank = dist.get_rank()
        ulysses_world_size = dist.get_world_size(ulysses_pg) if ulysses_pg is not None else 1
        ring_world_size = dist.get_world_size(ring_pg) if ring_pg is not None else 1
        batch_size, nheads, seqlen, d = query.shape

        original_seqlen = seqlen
        if seqlen % (64*ulysses_world_size) != 0:
            pad_length = (64*ulysses_world_size) - (seqlen % (64*ulysses_world_size))
            query = F.pad(query, (0, 0, 0, pad_length), "constant", 0)
            key = F.pad(key, (0, 0, 0, pad_length), "constant", 0)
            value = F.pad(value, (0, 0, 0, pad_length), "constant", 0)
            seqlen = query.shape[2]

        if transpose_matrix_q is not None and transpose_matrix_k is not None: 
            query=query.reshape(batch_size, nheads, seqlen//64, 64, d).transpose(0,2)
            key=key.reshape(batch_size, nheads, seqlen//64, 64, d).transpose(0,2)
            value=value.reshape(batch_size, nheads, seqlen//64, 64, d).transpose(0,2)

            # print(f"{rank},new_row_perm_idx: {new_row_perm_idx[rank%ring_world_size]}, {new_row_perm_idx[rank%ring_world_size].shape}, q shape: {query.shape}")
            query=query.index_select(0, new_row_perm_idx[rank%ring_world_size]).contiguous()
            dist.all_to_all_single(query,query,transpose_matrix_q.T[rank%ring_world_size].tolist(),transpose_matrix_q[rank%ring_world_size].tolist(), group=ring_pg) #, async_op=True


            key=key.index_select(0, new_col_perm_idx[rank%ring_world_size]).contiguous()
            dist.all_to_all_single(key,key,transpose_matrix_k.T[rank%ring_world_size].tolist(), transpose_matrix_k[rank%ring_world_size].tolist(), group=ring_pg)

            value=value.index_select(0, new_col_perm_idx[rank%ring_world_size]).contiguous()
            dist.all_to_all_single(value,value,transpose_matrix_k.T[rank%ring_world_size].tolist(),transpose_matrix_k[rank%ring_world_size].tolist(), group=ring_pg)

            query=query.transpose(0,2).reshape(batch_size, nheads, seqlen, d).contiguous()
            key=key.transpose(0,2).reshape(batch_size, nheads, seqlen, d).contiguous()
            value=value.transpose(0,2).reshape(batch_size, nheads, seqlen, d).contiguous()
            cuda.synchronize()

        elif new_row_perm_idx is not None:
            query=query.reshape(batch_size, nheads, seqlen//64, 64, d).transpose(0,2)
            key=key.reshape(batch_size, nheads, seqlen//64, 64, d).transpose(0,2)
            value=value.reshape(batch_size, nheads, seqlen//64, 64, d).transpose(0,2)
            query=query.index_select(0, new_row_perm_idx[rank%ring_world_size]).contiguous()
            key=key.index_select(0, new_col_perm_idx[rank%ring_world_size]).contiguous()
            value=value.index_select(0, new_col_perm_idx[rank%ring_world_size]).contiguous()
            query=query.transpose(0,2).reshape(batch_size, nheads, seqlen, d).contiguous()
            key=key.transpose(0,2).reshape(batch_size, nheads, seqlen, d).contiguous()
            value=value.transpose(0,2).reshape(batch_size, nheads, seqlen, d).contiguous()
            cuda.synchronize()
                
        if ring_world_size == 1:
            # print(sparse.float().sum())
            hidden_states = paroattn_qk_int8_pv_fp16_cuda(query, key, value, sparse=sparse)
            # print(f"{dist.get_rank()},{sparse.sum()},{sparse.shape}")
        # hidden_states_real = F.scaled_dot_product_attention(
        #         query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        #     )
        else:
            from yunchang.ring import ring_flash_attn_func
            from yunchang.kernels import AttnType
            hidden_states = ring_flash_attn_func(query, key, value, group=ring_pg, attn_type=AttnType.PARO, attn_processor=None, sparse=sparse)
        
        if transpose_matrix_q is not None:
            # print(f"{rank}, hidden states before transpose shape: {hidden_states.shape}, transpose_matrix_q: {transpose_matrix_q[rank%ring_world_size]},{transpose_matrix_q[rank%ring_world_size].shape}, new_row_deperm_idx: {new_row_deperm_idx[rank%ring_world_size].shape}")
            hidden_states=hidden_states.reshape(batch_size, nheads, seqlen//64, 64, d).transpose(0,2).contiguous()
            dist.all_to_all_single(hidden_states,hidden_states,transpose_matrix_q[rank%ring_world_size].tolist(),transpose_matrix_q.T[rank%ring_world_size].tolist(), group=ring_pg)
            hidden_states=hidden_states.index_select(0, new_row_deperm_idx[rank%ring_world_size]).transpose(0,2).reshape(batch_size, nheads, seqlen, d).contiguous()
            
        elif new_row_deperm_idx is not None:
            hidden_states=hidden_states.reshape(batch_size, nheads, seqlen//64, 64, d).transpose(0,2).contiguous()
            hidden_states=hidden_states.index_select(0, new_row_deperm_idx[rank%ring_world_size]).transpose(0,2).reshape(batch_size, nheads, seqlen, d).contiguous()

        cuda.synchronize()

        if original_seqlen != seqlen:
            hidden_states = hidden_states[:, :, :original_seqlen, :]
            
        hidden_states = all_to_all_4D(hidden_states, scatter_idx=2, gather_idx=1, group=ulysses_pg)
        
        # head-wise reorder
        if head_perm_idx is not None and head_deperm_idx is not None:
            hidden_states = hidden_states.index_select(1, head_deperm_idx)

        self.events['total_end'].record()
        cuda.synchronize()
        total_time = self.events['total_start'].elapsed_time(self.events['total_end'])
        self.time_accum['total'] = total_time
        # print(f"rank {torch.cuda.current_device()}, total_time: {total_time} ms")

        # print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))
        # prof.export_chrome_trace(f"rank_{torch.cuda.current_device()}_ulysses{ulysses_world_size}_ring_{ring_world_size}_onlyhead.json")  # 可选：导出火焰图

        
        hidden_states = hidden_states.transpose(1, 2).flatten(2, 3)
        hidden_states = hidden_states.type_as(query)

        # print(torch.cuda.memory_allocated() / 1024**2, "MB allocated after dot product attention")

        if hidden_states_img is not None:
            hidden_states = hidden_states + hidden_states_img

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        return hidden_states
    
    def get_time_stats(self):
        return {
            'total_ms': self.time_accum['total']
        }


class NEW_WanAttnProcessor2_0:
    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("WanAttnProcessor2_0 requires PyTorch 2.0. To use it, please upgrade PyTorch to 2.0.")

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        rotary_emb: Optional[torch.Tensor] = None,
        perm_idx: Optional[torch.Tensor] = None,
        deperm_idx: Optional[torch.Tensor] = None,
        layer_idx: int = 0,
    ) -> torch.Tensor:
        encoder_hidden_states_img = None
        if attn.add_k_proj is not None:
            encoder_hidden_states_img = encoder_hidden_states[:, :257]
            encoder_hidden_states = encoder_hidden_states[:, 257:]
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states

        query = attn.to_q(hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        query = query.unflatten(2, (attn.heads, -1)).transpose(1, 2)
        key = key.unflatten(2, (attn.heads, -1)).transpose(1, 2)
        value = value.unflatten(2, (attn.heads, -1)).transpose(1, 2)

        if rotary_emb is not None:

            def apply_rotary_emb(hidden_states: torch.Tensor, freqs: torch.Tensor):
                x_rotated = torch.view_as_complex(hidden_states.to(torch.float64).unflatten(3, (-1, 2)))
                x_out = torch.view_as_real(x_rotated * freqs).flatten(3, 4)
                return x_out.type_as(hidden_states)

            query = apply_rotary_emb(query, rotary_emb)
            key = apply_rotary_emb(key, rotary_emb)

        # I2V task
        hidden_states_img = None
        if encoder_hidden_states_img is not None:
            key_img = attn.add_k_proj(encoder_hidden_states_img)
            key_img = attn.norm_added_k(key_img)
            value_img = attn.add_v_proj(encoder_hidden_states_img)

            key_img = key_img.unflatten(2, (attn.heads, -1)).transpose(1, 2)
            value_img = value_img.unflatten(2, (attn.heads, -1)).transpose(1, 2)

            hidden_states_img = F.scaled_dot_product_attention(
                query, key_img, value_img, attn_mask=None, dropout_p=0.0, is_causal=False
            )
            hidden_states_img = hidden_states_img.transpose(1, 2).flatten(2, 3)
            hidden_states_img = hidden_states_img.type_as(query)

        # head-wise reorder
        if perm_idx is not None and deperm_idx is not None:
            query = query.index_select(1, perm_idx[layer_idx])
            key = key.index_select(1, perm_idx[layer_idx])
            value = value.index_select(1, perm_idx[layer_idx])
        
        # torch.Size([1, 40, 32760, 128])
        # import torch.distributed as dist
        # if dist.get_rank() == 0:
        #     import ipdb; ipdb.set_trace() 
        hidden_states, head_density = customize_spas_sage_attn_meansim_cuda(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False, simthreshd1=0.1, cdfthreshd=0.9, pvthreshd=10000, return_sparsity=True
        ) 

        # hidden_states, head_density = F.scaled_dot_product_attention(
        #     query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        # )

        # head-wise reorder
        if perm_idx is not None and deperm_idx is not None:
            hidden_states = hidden_states.index_select(1, deperm_idx[layer_idx])

        # import ipdb; ipdb.set_trace();

        hidden_states = hidden_states.transpose(1, 2).flatten(2, 3)
        hidden_states = hidden_states.type_as(query)

        if hidden_states_img is not None:
            hidden_states = hidden_states + hidden_states_img

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        return hidden_states, head_density

class NEW_PARALLEL_WanAttnProcessor2_0:
    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("WanAttnProcessor2_0 requires PyTorch 2.0. To use it, please upgrade PyTorch to 2.0.")
        self.events = {
            'total_start': cuda.Event(enable_timing=True),
            'total_end': cuda.Event(enable_timing=True),
        }
        self.time_accum = {
            'total': 0.0,
        }
        self.call_count = 0

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        rotary_emb: Optional[torch.Tensor] = None,
        head_perm_idx:Optional[torch.Tensor]=None,
        head_deperm_idx:Optional[torch.Tensor]=None,
        new_row_perm_idx:Optional[torch.Tensor]=None,
        new_col_perm_idx:Optional[torch.Tensor]=None,
        transpose_matrix_q:Optional[torch.Tensor]=None,
        transpose_matrix_k:Optional[torch.Tensor]=None,
        new_row_deperm_idx:Optional[torch.Tensor]=None,
        ulysses_pg: Optional[torch.distributed.ProcessGroup] = None,
        ring_pg: Optional[torch.distributed.ProcessGroup] = None,
    ) -> torch.Tensor:
        encoder_hidden_states_img = None
        if attn.add_k_proj is not None:
            encoder_hidden_states_img = encoder_hidden_states[:, :257]
            encoder_hidden_states = encoder_hidden_states[:, 257:]
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states

        query = attn.to_q(hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        query = query.unflatten(2, (attn.heads, -1)).transpose(1, 2)
        key = key.unflatten(2, (attn.heads, -1)).transpose(1, 2)
        value = value.unflatten(2, (attn.heads, -1)).transpose(1, 2)

        if rotary_emb is not None:

            def apply_rotary_emb(hidden_states: torch.Tensor, freqs: torch.Tensor):
                x_rotated = torch.view_as_complex(hidden_states.to(torch.float64).unflatten(3, (-1, 2)))
                x_out = torch.view_as_real(x_rotated * freqs).flatten(3, 4)
                return x_out.type_as(hidden_states)

            query = apply_rotary_emb(query, rotary_emb)
            key = apply_rotary_emb(key, rotary_emb)

        # I2V task
        hidden_states_img = None
        if encoder_hidden_states_img is not None:
            key_img = attn.add_k_proj(encoder_hidden_states_img)
            key_img = attn.norm_added_k(key_img)
            value_img = attn.add_v_proj(encoder_hidden_states_img)

            key_img = key_img.unflatten(2, (attn.heads, -1)).transpose(1, 2)
            value_img = value_img.unflatten(2, (attn.heads, -1)).transpose(1, 2)

            hidden_states_img = F.scaled_dot_product_attention(
                query, key_img, value_img, attn_mask=None, dropout_p=0.0, is_causal=False
            )
            hidden_states_img = hidden_states_img.transpose(1, 2).flatten(2, 3)
            hidden_states_img = hidden_states_img.type_as(query)

        # import torch.distributed as dist
        # if dist.get_rank() == 0:
        #     import ipdb; ipdb.set_trace()

        self.events['total_start'].record()

        if head_perm_idx is not None and head_deperm_idx is not None:
            query = query.index_select(1, head_perm_idx)
            query = all_to_all_4D(query, scatter_idx=1, gather_idx=2, group=ulysses_pg)
            key = key.index_select(1, head_perm_idx)
            key = all_to_all_4D(key, scatter_idx=1, gather_idx=2, group=ulysses_pg)
            value = value.index_select(1, head_perm_idx)
            value = all_to_all_4D(value, scatter_idx=1, gather_idx=2, group=ulysses_pg)
            
        else: 
            query = all_to_all_4D(query, scatter_idx=1, gather_idx=2, group=ulysses_pg)
            key = all_to_all_4D(key, scatter_idx=1, gather_idx=2, group=ulysses_pg )
            value = all_to_all_4D(value, scatter_idx=1, gather_idx=2, group=ulysses_pg)
            

        cuda.synchronize()
        rank = dist.get_rank(group=ring_pg)
        ulysses_world_size = dist.get_world_size(ulysses_pg) if ulysses_pg is not None else 1
        ring_world_size = dist.get_world_size(ring_pg) if ring_pg is not None else 1
        batch_size, nheads, seqlen, d = query.shape

        original_seqlen = seqlen
        if seqlen % (64*ulysses_world_size) != 0:
            pad_length = (64*ulysses_world_size) - (seqlen % (64*ulysses_world_size))
            query = F.pad(query, (0, 0, 0, pad_length), "constant", 0)
            key = F.pad(key, (0, 0, 0, pad_length), "constant", 0)
            value = F.pad(value, (0, 0, 0, pad_length), "constant", 0)
            seqlen = query.shape[2]

        if transpose_matrix_q is not None and transpose_matrix_k is not None:
            # print(f"query:{query.shape}, key:{key.shape}, value:{value.shape},transpose_matrix_q:{transpose_matrix_q},transpose_matrix_k:{transpose_matrix_k},new_row_perm_idx:{new_row_perm_idx.shape}, new_col_perm_idx:{new_col_perm_idx.shape}") 

            query=query.reshape(batch_size, nheads, seqlen//128, 128, d).transpose(0,2)
            key=key.reshape(batch_size, nheads, seqlen//64, 64, d).transpose(0,2)
            value=value.reshape(batch_size, nheads, seqlen//64, 64, d).transpose(0,2)

            query=query.index_select(0, new_row_perm_idx[rank%ring_world_size]).contiguous()
            dist.all_to_all_single(query,query,transpose_matrix_q.T[rank%ring_world_size].tolist(),transpose_matrix_q[rank%ring_world_size].tolist(), group=ring_pg) #, async_op=True


            key=key.index_select(0, new_col_perm_idx[rank%ring_world_size]).contiguous()
            dist.all_to_all_single(key,key,transpose_matrix_k.T[rank%ring_world_size].tolist(), transpose_matrix_k[rank%ring_world_size].tolist(), group=ring_pg)

            value=value.index_select(0, new_col_perm_idx[rank%ring_world_size]).contiguous()
            dist.all_to_all_single(value,value,transpose_matrix_k.T[rank%ring_world_size].tolist(),transpose_matrix_k[rank%ring_world_size].tolist(), group=ring_pg)

            query=query.transpose(0,2).reshape(batch_size, nheads, -1, d).contiguous()
            key=key.transpose(0,2).reshape(batch_size, nheads, -1, d).contiguous()
            value=value.transpose(0,2).reshape(batch_size, nheads, -1, d).contiguous()
            cuda.synchronize()

        # elif new_row_perm_idx is not None:
        #     query=query.reshape(batch_size, nheads, seqlen//128, 128, d).transpose(0,2)
        #     key=key.reshape(batch_size, nheads, seqlen//64, 64, d).transpose(0,2)
        #     value=value.reshape(batch_size, nheads, seqlen//64, 64, d).transpose(0,2)
        #     query=query.index_select(0, new_row_perm_idx[rank%ring_world_size]).contiguous()
        #     key=key.index_select(0, new_col_perm_idx[rank%ring_world_size]).contiguous()
        #     value=value.index_select(0, new_col_perm_idx[rank%ring_world_size]).contiguous()
        #     query=query.transpose(0,2).reshape(batch_size, nheads, seqlen, d).contiguous()
        #     key=key.transpose(0,2).reshape(batch_size, nheads, seqlen, d).contiguous()
        #     value=value.transpose(0,2).reshape(batch_size, nheads, seqlen, d).contiguous()
        #     cuda.synchronize()

        if ring_world_size == 1:
            hidden_states, head_density = customize_spas_sage_attn_meansim_cuda(
                query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False, simthreshd1=0.1, cdfthreshd=0.9, pvthreshd=10000, return_sparsity=True
            ) 
            cuda.synchronize()
        else:
            # print("Using ring flash attention kernel!")
            from yunchang.ring import ring_flash_attn_func
            from yunchang.kernels import AttnType
            hidden_states, head_density = ring_flash_attn_func(query, key, value, group=ring_pg, attn_type=AttnType.SPARGE, attn_processor=None)

        if transpose_matrix_q is not None:
            hidden_states=hidden_states.reshape(batch_size, nheads, seqlen//128, 128, d).transpose(0,2).contiguous()
            dist.all_to_all_single(hidden_states,hidden_states,transpose_matrix_q[rank%ring_world_size].tolist(),transpose_matrix_q.T[rank%ring_world_size].tolist(), group=ring_pg)
            hidden_states=hidden_states.index_select(0, new_row_deperm_idx[rank%ring_world_size]).transpose(0,2).reshape(batch_size, nheads, -1, d).contiguous()
            
        # elif new_row_deperm_idx is not None:
        #     hidden_states=hidden_states.reshape(batch_size, nheads, seqlen//128, 128, d).transpose(0,2).contiguous()
        #     hidden_states=hidden_states.index_select(0, new_row_deperm_idx[rank%ring_world_size]).transpose(0,2).reshape(batch_size, nheads, seqlen, d).contiguous()

        cuda.synchronize()
        
        if original_seqlen != seqlen:
            hidden_states = hidden_states[:, :, :original_seqlen, :]
            
        hidden_states = all_to_all_4D(hidden_states, scatter_idx=2, gather_idx=1, group=ulysses_pg)
        
        # head-wise reorder
        if head_perm_idx is not None and head_deperm_idx is not None:
            hidden_states = hidden_states.index_select(1, head_deperm_idx)

        self.events['total_end'].record()
        cuda.synchronize()
        total_time = self.events['total_start'].elapsed_time(self.events['total_end']) 
        self.time_accum['total'] = total_time

        # hidden_states, head_density = F.scaled_dot_product_attention(
        #     query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        # )


        # import ipdb; ipdb.set_trace();

        hidden_states = hidden_states.transpose(1, 2).flatten(2, 3)
        hidden_states = hidden_states.type_as(query)

        if hidden_states_img is not None:
            hidden_states = hidden_states + hidden_states_img

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        return hidden_states, head_density

    def get_time_stats(self):
        return {
            'total_ms': self.time_accum['total']
        }
    
"""
INFO: For wan, there are attn1 and attn2, so we replace the transfomer block instead

"""

import math
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import omegaconf
from omegaconf import OmegaConf

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.utils import USE_PEFT_BACKEND, logging, scale_lora_layers, unscale_lora_layers
from diffusers.models.attention import FeedForward
from diffusers.models.attention_processor import Attention
from diffusers.models.cache_utils import CacheMixin
from diffusers.models.embeddings import PixArtAlphaTextProjection, TimestepEmbedding, Timesteps, get_1d_rotary_pos_embed
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.normalization import FP32LayerNorm

class WanImageEmbedding(torch.nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()

        self.norm1 = FP32LayerNorm(in_features)
        self.ff = FeedForward(in_features, out_features, mult=1, activation_fn="gelu")
        self.norm2 = FP32LayerNorm(out_features)

    def forward(self, encoder_hidden_states_image: torch.Tensor) -> torch.Tensor:
        hidden_states = self.norm1(encoder_hidden_states_image)
        hidden_states = self.ff(hidden_states)
        hidden_states = self.norm2(hidden_states)
        return hidden_states
    
class Org_PARALLEL_WanAttnProcessor2_0:
    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("WanAttnProcessor2_0 requires PyTorch 2.0. To use it, please upgrade PyTorch to 2.0.")

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        rotary_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        encoder_hidden_states_img = None
        if attn.add_k_proj is not None:
            encoder_hidden_states_img = encoder_hidden_states[:, :257]
            encoder_hidden_states = encoder_hidden_states[:, 257:]
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states

        query = attn.to_q(hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        query = query.unflatten(2, (attn.heads, -1)).transpose(1, 2)
        key = key.unflatten(2, (attn.heads, -1)).transpose(1, 2)
        value = value.unflatten(2, (attn.heads, -1)).transpose(1, 2)

        if rotary_emb is not None:

            def apply_rotary_emb(hidden_states: torch.Tensor, freqs: torch.Tensor):
                x_rotated = torch.view_as_complex(hidden_states.to(torch.float64).unflatten(3, (-1, 2)))
                x_out = torch.view_as_real(x_rotated * freqs).flatten(3, 4)
                return x_out.type_as(hidden_states)

            query = apply_rotary_emb(query, rotary_emb)
            key = apply_rotary_emb(key, rotary_emb)

        # I2V task
        hidden_states_img = None
        if encoder_hidden_states_img is not None:
            key_img = attn.add_k_proj(encoder_hidden_states_img)
            key_img = attn.norm_added_k(key_img)
            value_img = attn.add_v_proj(encoder_hidden_states_img)

            key_img = key_img.unflatten(2, (attn.heads, -1)).transpose(1, 2)
            value_img = value_img.unflatten(2, (attn.heads, -1)).transpose(1, 2)

            hidden_states_img = F.scaled_dot_product_attention(
                query, key_img, value_img, attn_mask=None, dropout_p=0.0, is_causal=False
            )
            hidden_states_img = hidden_states_img.transpose(1, 2).flatten(2, 3)
            hidden_states_img = hidden_states_img.type_as(query)

        query = all_to_all_4D(query, scatter_idx=1, gather_idx=2)
        key = all_to_all_4D(key, scatter_idx=1, gather_idx=2)
        value = all_to_all_4D(value, scatter_idx=1, gather_idx=2)
        cuda.synchronize()

        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        cuda.synchronize()
        hidden_states = all_to_all_4D(hidden_states, scatter_idx=2, gather_idx=1)
        
        hidden_states = hidden_states.transpose(1, 2).flatten(2, 3)
        hidden_states = hidden_states.type_as(query)

        if hidden_states_img is not None:
            hidden_states = hidden_states + hidden_states_img

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        return hidden_states 


class WanTimeTextImageEmbedding(nn.Module):
    def __init__(
        self,
        dim: int,
        time_freq_dim: int,
        time_proj_dim: int,
        text_embed_dim: int,
        image_embed_dim: Optional[int] = None,
    ):
        super().__init__()

        self.timesteps_proj = Timesteps(num_channels=time_freq_dim, flip_sin_to_cos=True, downscale_freq_shift=0)
        self.time_embedder = TimestepEmbedding(in_channels=time_freq_dim, time_embed_dim=dim)
        self.act_fn = nn.SiLU()
        self.time_proj = nn.Linear(dim, time_proj_dim)
        self.text_embedder = PixArtAlphaTextProjection(text_embed_dim, dim, act_fn="gelu_tanh")

        self.image_embedder = None
        if image_embed_dim is not None:
            self.image_embedder = WanImageEmbedding(image_embed_dim, dim)

    def forward(
        self,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_hidden_states_image: Optional[torch.Tensor] = None,
    ):
        timestep = self.timesteps_proj(timestep)

        time_embedder_dtype = next(iter(self.time_embedder.parameters())).dtype
        if timestep.dtype != time_embedder_dtype and time_embedder_dtype != torch.int8:
            timestep = timestep.to(time_embedder_dtype)
        temb = self.time_embedder(timestep).type_as(encoder_hidden_states)
        timestep_proj = self.time_proj(self.act_fn(temb))

        encoder_hidden_states = self.text_embedder(encoder_hidden_states)
        if encoder_hidden_states_image is not None:
            encoder_hidden_states_image = self.image_embedder(encoder_hidden_states_image)

        return temb, timestep_proj, encoder_hidden_states, encoder_hidden_states_image


class WanRotaryPosEmbed(nn.Module):
    def __init__(
        self, attention_head_dim: int, patch_size: Tuple[int, int, int], max_seq_len: int, theta: float = 10000.0
    ):
        super().__init__()

        self.attention_head_dim = attention_head_dim
        self.patch_size = patch_size
        self.max_seq_len = max_seq_len

        h_dim = w_dim = 2 * (attention_head_dim // 6)
        t_dim = attention_head_dim - h_dim - w_dim

        freqs = []
        for dim in [t_dim, h_dim, w_dim]:
            freq = get_1d_rotary_pos_embed(
                dim, max_seq_len, theta, use_real=False, repeat_interleave_real=False, freqs_dtype=torch.float64
            )
            freqs.append(freq)
        self.freqs = torch.cat(freqs, dim=1)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, num_channels, num_frames, height, width = hidden_states.shape
        p_t, p_h, p_w = self.patch_size
        ppf, pph, ppw = num_frames // p_t, height // p_h, width // p_w

        self.freqs = self.freqs.to(hidden_states.device)
        freqs = self.freqs.split_with_sizes(
            [
                self.attention_head_dim // 2 - 2 * (self.attention_head_dim // 6),
                self.attention_head_dim // 6,
                self.attention_head_dim // 6,
            ],
            dim=1,
        )

        freqs_f = freqs[0][:ppf].view(ppf, 1, 1, -1).expand(ppf, pph, ppw, -1)
        freqs_h = freqs[1][:pph].view(1, pph, 1, -1).expand(ppf, pph, ppw, -1)
        freqs_w = freqs[2][:ppw].view(1, 1, ppw, -1).expand(ppf, pph, ppw, -1)
        freqs = torch.cat([freqs_f, freqs_h, freqs_w], dim=-1).reshape(1, 1, ppf * pph * ppw, -1)
        return freqs


class NEW_WanTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        ffn_dim: int,
        num_heads: int,
        qk_norm: str = "rms_norm_across_heads",
        cross_attn_norm: bool = False,
        eps: float = 1e-6,
        added_kv_proj_dim: Optional[int] = None,
        layer_idx: int = 0,
    ):
        super().__init__()

        # 1. Self-attention
        self.norm1 = FP32LayerNorm(dim, eps, elementwise_affine=False)
        self.attn1 = Attention(
            query_dim=dim,
            heads=num_heads,
            kv_heads=num_heads,
            dim_head=dim // num_heads,
            qk_norm=qk_norm,
            eps=eps,
            bias=True,
            cross_attention_dim=None,
            out_bias=True,
            processor=NEW_PARALLEL_WanAttnProcessor2_0(),
        )

        # 2. Cross-attention
        self.attn2 = Attention(
            query_dim=dim,
            heads=num_heads,
            kv_heads=num_heads,
            dim_head=dim // num_heads,
            qk_norm=qk_norm,
            eps=eps,
            bias=True,
            cross_attention_dim=None,
            out_bias=True,
            added_kv_proj_dim=added_kv_proj_dim,
            added_proj_bias=True,
            processor=Org_PARALLEL_WanAttnProcessor2_0(),
        )
        self.norm2 = FP32LayerNorm(dim, eps, elementwise_affine=True) if cross_attn_norm else nn.Identity()

        # 3. Feed-forward
        self.ffn = FeedForward(dim, inner_dim=ffn_dim, activation_fn="gelu-approximate")
        self.norm3 = FP32LayerNorm(dim, eps, elementwise_affine=False)

        self.scale_shift_table = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        rotary_emb: torch.Tensor,
        head_perm_idx:Optional[torch.Tensor]=None,
        head_deperm_idx:Optional[torch.Tensor]=None,
        new_row_perm_idx:Optional[torch.Tensor]=None,
        new_col_perm_idx:Optional[torch.Tensor]=None,
        transpose_matrix_q:Optional[torch.Tensor]=None,
        transpose_matrix_k:Optional[torch.Tensor]=None,
        new_row_deperm_idx:Optional[torch.Tensor]=None,
        ulysses_pg: Optional[torch.distributed.ProcessGroup] = None,
        ring_pg: Optional[torch.distributed.ProcessGroup] = None,
        layer_idx: Optional[int] = 0,
    ) -> torch.Tensor:
        shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = (
            self.scale_shift_table + temb.float()
        ).chunk(6, dim=1)

        # 1. Self-attention
        norm_hidden_states = (self.norm1(hidden_states.float()) * (1 + scale_msa) + shift_msa).type_as(hidden_states)
        attn_output, head_density = self.attn1(hidden_states=norm_hidden_states, rotary_emb=rotary_emb, head_perm_idx=head_perm_idx if head_perm_idx is not None else None,
                    head_deperm_idx=head_deperm_idx if head_deperm_idx is not None else None, new_row_perm_idx=new_row_perm_idx if new_row_perm_idx is not None else None, new_col_perm_idx=new_col_perm_idx if new_col_perm_idx is not None else None, transpose_matrix_q=transpose_matrix_q if transpose_matrix_q is not None else None, transpose_matrix_k=transpose_matrix_k if transpose_matrix_k is not None else None, new_row_deperm_idx=new_row_deperm_idx if new_row_deperm_idx is not None else None, ulysses_pg=ulysses_pg, ring_pg=ring_pg)
        hidden_states = (hidden_states.float() + attn_output * gate_msa).type_as(hidden_states)

        # 2. Cross-attention
        norm_hidden_states = self.norm2(hidden_states.float()).type_as(hidden_states)
        attn_output = self.attn2(hidden_states=norm_hidden_states, encoder_hidden_states=encoder_hidden_states)
        hidden_states = hidden_states + attn_output

        # 3. Feed-forward
        norm_hidden_states = (self.norm3(hidden_states.float()) * (1 + c_scale_msa) + c_shift_msa).type_as(
            hidden_states
        )
        ff_output = self.ffn(norm_hidden_states)
        hidden_states = (hidden_states.float() + ff_output.float() * c_gate_msa).type_as(hidden_states)

        return hidden_states, head_density

class PARO_WanTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        ffn_dim: int,
        num_heads: int,
        qk_norm: str = "rms_norm_across_heads",
        cross_attn_norm: bool = False,
        eps: float = 1e-6,
        added_kv_proj_dim: Optional[int] = None,
        layer_idx: int = 0,
    ):
        super().__init__()

        # 1. Self-attention
        self.norm1 = FP32LayerNorm(dim, eps, elementwise_affine=False)
        self.attn1 = Attention(
            query_dim=dim,
            heads=num_heads,
            kv_heads=num_heads,
            dim_head=dim // num_heads,
            qk_norm=qk_norm,
            eps=eps,
            bias=True,
            cross_attention_dim=None,
            out_bias=True,
            processor=PARO_PARALLEL_WanAttnProcessor2_0(),
        )

        # 2. Cross-attention
        self.attn2 = Attention(
            query_dim=dim,
            heads=num_heads,
            kv_heads=num_heads,
            dim_head=dim // num_heads,
            qk_norm=qk_norm,
            eps=eps,
            bias=True,
            cross_attention_dim=None,
            out_bias=True,
            added_kv_proj_dim=added_kv_proj_dim,
            added_proj_bias=True,
            processor=Org_PARALLEL_WanAttnProcessor2_0(),
        )
        self.norm2 = FP32LayerNorm(dim, eps, elementwise_affine=True) if cross_attn_norm else nn.Identity()

        # 3. Feed-forward
        self.ffn = FeedForward(dim, inner_dim=ffn_dim, activation_fn="gelu-approximate")
        self.norm3 = FP32LayerNorm(dim, eps, elementwise_affine=False)

        self.scale_shift_table = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        rotary_emb: torch.Tensor,
        sparse: Optional[torch.Tensor] = None,
        head_perm_idx:Optional[torch.Tensor]=None,
        head_deperm_idx:Optional[torch.Tensor]=None,
        new_row_perm_idx:Optional[torch.Tensor]=None,
        new_col_perm_idx:Optional[torch.Tensor]=None,
        transpose_matrix_q:Optional[torch.Tensor]=None,
        transpose_matrix_k:Optional[torch.Tensor]=None,
        new_row_deperm_idx:Optional[torch.Tensor]=None,
        ulysses_pg: Optional[torch.distributed.ProcessGroup] = None,
        ring_pg: Optional[torch.distributed.ProcessGroup] = None,
        layer_idx: Optional[int] = 0,
    ) -> torch.Tensor:
        shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = (
            self.scale_shift_table + temb.float()
        ).chunk(6, dim=1)

        layer_idx = self.layer_idx if hasattr(self, 'layer_idx') else layer_idx
        # 1. Self-attention
        norm_hidden_states = (self.norm1(hidden_states.float()) * (1 + scale_msa) + shift_msa).type_as(hidden_states)
        attn_output = self.attn1(hidden_states=norm_hidden_states, rotary_emb=rotary_emb, 
                    sparse=sparse if sparse is not None else None,
                    head_perm_idx=head_perm_idx if head_perm_idx is not None else None,
                    head_deperm_idx=head_deperm_idx if head_deperm_idx is not None else None, new_row_perm_idx=new_row_perm_idx if new_row_perm_idx is not None else None, new_col_perm_idx=new_col_perm_idx if new_col_perm_idx is not None else None, transpose_matrix_q=transpose_matrix_q if transpose_matrix_q is not None else None, transpose_matrix_k=transpose_matrix_k if transpose_matrix_k is not None else None, new_row_deperm_idx=new_row_deperm_idx if new_row_deperm_idx is not None else None, ulysses_pg=ulysses_pg, ring_pg=ring_pg)
        hidden_states = (hidden_states.float() + attn_output * gate_msa).type_as(hidden_states)

        # 2. Cross-attention
        norm_hidden_states = self.norm2(hidden_states.float()).type_as(hidden_states)
        attn_output = self.attn2(hidden_states=norm_hidden_states, encoder_hidden_states=encoder_hidden_states)
        hidden_states = hidden_states + attn_output

        # 3. Feed-forward
        norm_hidden_states = (self.norm3(hidden_states.float()) * (1 + c_scale_msa) + c_shift_msa).type_as(
            hidden_states
        )
        ff_output = self.ffn(norm_hidden_states)
        hidden_states = (hidden_states.float() + ff_output.float() * c_gate_msa).type_as(hidden_states)

        return hidden_states

class NEW_WanTransformer3DModel(ModelMixin, ConfigMixin, CacheMixin):
    r"""
    A Transformer model for video-like data used in the Wan model.

    Args:
        patch_size (`Tuple[int]`, defaults to `(1, 2, 2)`):
            3D patch dimensions for video embedding (t_patch, h_patch, w_patch).
        num_attention_heads (`int`, defaults to `40`):
            Fixed length for text embeddings.
        attention_head_dim (`int`, defaults to `128`):
            The number of channels in each head.
        in_channels (`int`, defaults to `16`):
            The number of channels in the input.
        out_channels (`int`, defaults to `16`):
            The number of channels in the output.
        text_dim (`int`, defaults to `512`):
            Input dimension for text embeddings.
        freq_dim (`int`, defaults to `256`):
            Dimension for sinusoidal time embeddings.
        ffn_dim (`int`, defaults to `13824`):
            Intermediate dimension in feed-forward network.
        num_layers (`int`, defaults to `40`):
            The number of layers of transformer blocks to use.
        window_size (`Tuple[int]`, defaults to `(-1, -1)`):
            Window size for local attention (-1 indicates global attention).
        cross_attn_norm (`bool`, defaults to `True`):
            Enable cross-attention normalization.
        qk_norm (`bool`, defaults to `True`):
            Enable query/key normalization.
        eps (`float`, defaults to `1e-6`):
            Epsilon value for normalization layers.
        add_img_emb (`bool`, defaults to `False`):
            Whether to use img_emb.
        added_kv_proj_dim (`int`, *optional*, defaults to `None`):
            The number of channels to use for the added key and value projections. If `None`, no projection is used.
    """

    _supports_gradient_checkpointing = True
    _skip_layerwise_casting_patterns = ["patch_embedding", "condition_embedder", "norm"]
    _no_split_modules = ["WanTransformerBlock"]
    _keep_in_fp32_modules = ["time_embedder", "scale_shift_table", "norm1", "norm2", "norm3"]
    _keys_to_ignore_on_load_unexpected = ["norm_added_q"]

    @register_to_config
    def __init__(
        self,
        patch_size: Tuple[int] = (1, 2, 2),
        num_attention_heads: int = 40,
        attention_head_dim: int = 128,
        in_channels: int = 16,
        out_channels: int = 16,
        text_dim: int = 4096,
        freq_dim: int = 256,
        ffn_dim: int = 13824,
        num_layers: int = 40,
        cross_attn_norm: bool = True,
        qk_norm: Optional[str] = "rms_norm_across_heads",
        eps: float = 1e-6,
        image_dim: Optional[int] = None,
        added_kv_proj_dim: Optional[int] = None,
        rope_max_seq_len: int = 1024,
        quant_config = None,
    ) -> None:
        super().__init__()

        inner_dim = num_attention_heads * attention_head_dim
        out_channels = out_channels or in_channels

        # 1. Patch & position embedding
        self.rope = WanRotaryPosEmbed(attention_head_dim, patch_size, rope_max_seq_len)
        self.patch_embedding = nn.Conv3d(in_channels, inner_dim, kernel_size=patch_size, stride=patch_size)

        # 2. Condition embeddings
        # image_embedding_dim=1280 for I2V model
        self.condition_embedder = WanTimeTextImageEmbedding(
            dim=inner_dim,
            time_freq_dim=freq_dim,
            time_proj_dim=inner_dim * 6,
            text_embed_dim=text_dim,
            image_embed_dim=image_dim,
        )

        # 3. Transformer blocks
        self.blocks = nn.ModuleList(
            [
                NEW_WanTransformerBlock(
                    inner_dim, ffn_dim, num_attention_heads, qk_norm, cross_attn_norm, eps, added_kv_proj_dim, layer
                )
                for layer in range(num_layers)
            ]
        )
        # TODO: Check for attn1 and attn2

        # 4. Output norm & projection
        self.norm_out = FP32LayerNorm(inner_dim, eps, elementwise_affine=False)
        self.proj_out = nn.Linear(inner_dim, out_channels * math.prod(patch_size))
        self.scale_shift_table = nn.Parameter(torch.randn(1, 2, inner_dim) / inner_dim**0.5)

        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.LongTensor,
        encoder_hidden_states: torch.Tensor,
        encoder_hidden_states_image: Optional[torch.Tensor] = None,
        return_dict: bool = True,
        perm_idx: Optional[list] = None,
        deperm_idx: Optional[list] = None,
        attention_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        if attention_kwargs is not None:
            attention_kwargs = attention_kwargs.copy()
            lora_scale = attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0

        if USE_PEFT_BACKEND:
            # weight the lora layers by setting `lora_scale` for each PEFT layer
            scale_lora_layers(self, lora_scale)
        else:
            if attention_kwargs is not None and attention_kwargs.get("scale", None) is not None:
                RuntimeError("`scale` argument is not supported when not using PEFT backend. Please set `USE_PEFT_BACKEND=True` in your config.")

        batch_size, num_channels, num_frames, height, width = hidden_states.shape
        p_t, p_h, p_w = self.config.patch_size
        post_patch_num_frames = num_frames // p_t
        post_patch_height = height // p_h
        post_patch_width = width // p_w

        rotary_emb = self.rope(hidden_states)

        hidden_states = self.patch_embedding(hidden_states)
        hidden_states = hidden_states.flatten(2).transpose(1, 2)

        temb, timestep_proj, encoder_hidden_states, encoder_hidden_states_image = self.condition_embedder(
            timestep, encoder_hidden_states, encoder_hidden_states_image
        )
        timestep_proj = timestep_proj.unflatten(1, (6, -1))

        if encoder_hidden_states_image is not None:
            encoder_hidden_states = torch.concat([encoder_hidden_states_image, encoder_hidden_states], dim=1)

        # 4. Transformer blocks
        if torch.is_grad_enabled() and self.gradient_checkpointing:
            all_head_density = []
            for block in self.blocks:
                hidden_states, head_density = self._gradient_checkpointing_func(
                    block, hidden_states, encoder_hidden_states, timestep_proj, rotary_emb
                )
                all_head_density.append(head_density)
            all_head_density = torch.stack(all_head_density, dim=0)
        else:
            all_head_density = []
            # import ipdb; ipdb.set_trace();
            for block in self.blocks:
                hidden_states, head_density = block(hidden_states, encoder_hidden_states, timestep_proj, rotary_emb, perm_idx, deperm_idx) 
                all_head_density.append(head_density)
            all_head_density = torch.stack(all_head_density, dim=0)  # shape: [num_blocks, num_heads]
            # torch.save(all_head_density, f"/mnt/public/chensiqi/ParaAttention/first_block_cache_examples/profile/head_density_{timestep.item()}.pt")
        print(all_head_density.shape) 

        # 5. Output norm, projection & unpatchify
        shift, scale = (self.scale_shift_table + temb.unsqueeze(1)).chunk(2, dim=1)

        # Move the shift and scale tensors to the same device as hidden_states.
        # When using multi-GPU inference via accelerate these will be on the
        # first device rather than the last device, which hidden_states ends up
        # on.
        shift = shift.to(hidden_states.device)
        scale = scale.to(hidden_states.device)

        hidden_states = (self.norm_out(hidden_states.float()) * (1 + scale) + shift).type_as(hidden_states)
        hidden_states = self.proj_out(hidden_states)

        hidden_states = hidden_states.reshape(
            batch_size, post_patch_num_frames, post_patch_height, post_patch_width, p_t, p_h, p_w, -1
        )
        hidden_states = hidden_states.permute(0, 7, 1, 4, 2, 5, 3, 6)
        output = hidden_states.flatten(6, 7).flatten(4, 5).flatten(2, 3)

        if USE_PEFT_BACKEND:
            # remove `lora_scale` from each PEFT layer
            unscale_lora_layers(self, lora_scale)

        if not return_dict:
            return output, all_head_density

        return Transformer2DModelOutput(sample=output), all_head_density

class PARO_WanTransformer3DModel(ModelMixin, ConfigMixin, CacheMixin):
    r"""
    A Transformer model for video-like data used in the Wan model.

    Args:
        patch_size (`Tuple[int]`, defaults to `(1, 2, 2)`):
            3D patch dimensions for video embedding (t_patch, h_patch, w_patch).
        num_attention_heads (`int`, defaults to `40`):
            Fixed length for text embeddings.
        attention_head_dim (`int`, defaults to `128`):
            The number of channels in each head.
        in_channels (`int`, defaults to `16`):
            The number of channels in the input.
        out_channels (`int`, defaults to `16`):
            The number of channels in the output.
        text_dim (`int`, defaults to `512`):
            Input dimension for text embeddings.
        freq_dim (`int`, defaults to `256`):
            Dimension for sinusoidal time embeddings.
        ffn_dim (`int`, defaults to `13824`):
            Intermediate dimension in feed-forward network.
        num_layers (`int`, defaults to `40`):
            The number of layers of transformer blocks to use.
        window_size (`Tuple[int]`, defaults to `(-1, -1)`):
            Window size for local attention (-1 indicates global attention).
        cross_attn_norm (`bool`, defaults to `True`):
            Enable cross-attention normalization.
        qk_norm (`bool`, defaults to `True`):
            Enable query/key normalization.
        eps (`float`, defaults to `1e-6`):
            Epsilon value for normalization layers.
        add_img_emb (`bool`, defaults to `False`):
            Whether to use img_emb.
        added_kv_proj_dim (`int`, *optional*, defaults to `None`):
            The number of channels to use for the added key and value projections. If `None`, no projection is used.
    """

    _supports_gradient_checkpointing = True
    _skip_layerwise_casting_patterns = ["patch_embedding", "condition_embedder", "norm"]
    _no_split_modules = ["WanTransformerBlock"]
    _keep_in_fp32_modules = ["time_embedder", "scale_shift_table", "norm1", "norm2", "norm3"]
    _keys_to_ignore_on_load_unexpected = ["norm_added_q"]

    @register_to_config
    def __init__(
        self,
        patch_size: Tuple[int] = (1, 2, 2),
        num_attention_heads: int = 40,
        attention_head_dim: int = 128,
        in_channels: int = 16,
        out_channels: int = 16,
        text_dim: int = 4096,
        freq_dim: int = 256,
        ffn_dim: int = 13824,
        num_layers: int = 40,
        cross_attn_norm: bool = True,
        qk_norm: Optional[str] = "rms_norm_across_heads",
        eps: float = 1e-6,
        image_dim: Optional[int] = None,
        added_kv_proj_dim: Optional[int] = None,
        rope_max_seq_len: int = 1024,
        quant_config = None,
    ) -> None:
        super().__init__()

        inner_dim = num_attention_heads * attention_head_dim
        out_channels = out_channels or in_channels

        # 1. Patch & position embedding
        self.rope = WanRotaryPosEmbed(attention_head_dim, patch_size, rope_max_seq_len)
        self.patch_embedding = nn.Conv3d(in_channels, inner_dim, kernel_size=patch_size, stride=patch_size)

        # 2. Condition embeddings
        # image_embedding_dim=1280 for I2V model
        self.condition_embedder = WanTimeTextImageEmbedding(
            dim=inner_dim,
            time_freq_dim=freq_dim,
            time_proj_dim=inner_dim * 6,
            text_embed_dim=text_dim,
            image_embed_dim=image_dim,
        )

        # 3. Transformer blocks
        self.blocks = nn.ModuleList(
            [
                PARO_WanTransformerBlock(
                    inner_dim, ffn_dim, num_attention_heads, qk_norm, cross_attn_norm, eps, added_kv_proj_dim, layer
                )
                for layer in range(num_layers)
            ]
        )
        # TODO: Check for attn1 and attn2

        # 4. Output norm & projection
        self.norm_out = FP32LayerNorm(inner_dim, eps, elementwise_affine=False)
        self.proj_out = nn.Linear(inner_dim, out_channels * math.prod(patch_size))
        self.scale_shift_table = nn.Parameter(torch.randn(1, 2, inner_dim) / inner_dim**0.5)

        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.LongTensor,
        encoder_hidden_states: torch.Tensor,
        encoder_hidden_states_image: Optional[torch.Tensor] = None,
        return_dict: bool = True,
        sparse: Optional[torch.Tensor] = None,
        attention_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        if attention_kwargs is not None:
            attention_kwargs = attention_kwargs.copy()
            lora_scale = attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0

        if USE_PEFT_BACKEND:
            # weight the lora layers by setting `lora_scale` for each PEFT layer
            scale_lora_layers(self, lora_scale)
        else:
            if attention_kwargs is not None and attention_kwargs.get("scale", None) is not None:
                RuntimeError("`scale` argument is not supported when not using PEFT backend. Please set `USE_PEFT_BACKEND=True` in your config.")

        batch_size, num_channels, num_frames, height, width = hidden_states.shape
        p_t, p_h, p_w = self.config.patch_size
        post_patch_num_frames = num_frames // p_t
        post_patch_height = height // p_h
        post_patch_width = width // p_w

        rotary_emb = self.rope(hidden_states)

        hidden_states = self.patch_embedding(hidden_states)
        hidden_states = hidden_states.flatten(2).transpose(1, 2)

        temb, timestep_proj, encoder_hidden_states, encoder_hidden_states_image = self.condition_embedder(
            timestep, encoder_hidden_states, encoder_hidden_states_image
        )
        timestep_proj = timestep_proj.unflatten(1, (6, -1))

        if encoder_hidden_states_image is not None:
            encoder_hidden_states = torch.concat([encoder_hidden_states_image, encoder_hidden_states], dim=1)

        # 4. Transformer blocks
        if torch.is_grad_enabled() and self.gradient_checkpointing:
            for block in self.blocks:
                hidden_states= self._gradient_checkpointing_func(
                    block, hidden_states, encoder_hidden_states, timestep_proj, rotary_emb
                )
        else:
            # import ipdb; ipdb.set_trace();
            for block in self.blocks:
                hidden_states = block(hidden_states, encoder_hidden_states, timestep_proj, rotary_emb, sparse=sparse) 

        # 5. Output norm, projection & unpatchify
        shift, scale = (self.scale_shift_table + temb.unsqueeze(1)).chunk(2, dim=1)

        # Move the shift and scale tensors to the same device as hidden_states.
        # When using multi-GPU inference via accelerate these will be on the
        # first device rather than the last device, which hidden_states ends up
        # on.
        shift = shift.to(hidden_states.device)
        scale = scale.to(hidden_states.device)

        hidden_states = (self.norm_out(hidden_states.float()) * (1 + scale) + shift).type_as(hidden_states)
        hidden_states = self.proj_out(hidden_states)

        hidden_states = hidden_states.reshape(
            batch_size, post_patch_num_frames, post_patch_height, post_patch_width, p_t, p_h, p_w, -1
        )
        hidden_states = hidden_states.permute(0, 7, 1, 4, 2, 5, 3, 6)
        output = hidden_states.flatten(6, 7).flatten(4, 5).flatten(2, 3)

        if USE_PEFT_BACKEND:
            # remove `lora_scale` from each PEFT layer
            unscale_lora_layers(self, lora_scale)

        if not return_dict:
            return output

        return Transformer2DModelOutput(sample=output)

