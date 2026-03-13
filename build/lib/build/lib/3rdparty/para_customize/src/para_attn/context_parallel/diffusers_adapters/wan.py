import functools
from typing import Any, Dict, List, Optional, Union

import torch
import torch.distributed as dist
from diffusers import DiffusionPipeline, WanTransformer3DModel
from parallel_examples.wan import NEW_WanTransformer3DModel
from parallel_examples.wan import PARO_WanTransformer3DModel
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.utils import logging, scale_lora_layers, unscale_lora_layers, USE_PEFT_BACKEND

import para_attn.primitives as DP
from para_attn.context_parallel import init_context_parallel_mesh
from para_attn.para_attn_interface import UnifiedAttnMode

logger = logging.get_logger(__name__)


def parallelize_transformer(transformer: WanTransformer3DModel, *, mesh=None):
    if getattr(transformer, "_is_parallelized", False):
        return transformer

    mesh = init_context_parallel_mesh(transformer.device.type, mesh=mesh)
    batch_mesh = mesh["batch"]
    seq_mesh = mesh["ring", "ulysses"]._flatten()

    @functools.wraps(transformer.__class__.forward)
    def new_forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.LongTensor,
        encoder_hidden_states: torch.Tensor,
        encoder_hidden_states_image: Optional[torch.Tensor] = None,
        return_dict: bool = True,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs,
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
                logger.warning("Passing `scale` via `attention_kwargs` when not using the PEFT backend is ineffective.")

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

        timestep_proj = DP.get_assigned_chunk(timestep_proj, dim=0, group=batch_mesh)
        temb = DP.get_assigned_chunk(temb, dim=0, group=batch_mesh)
        hidden_states = DP.get_assigned_chunk(hidden_states, dim=0, group=batch_mesh)
        hidden_states = DP.get_assigned_chunk(hidden_states, dim=-2, group=seq_mesh)
        encoder_hidden_states = DP.get_assigned_chunk(encoder_hidden_states, dim=0, group=batch_mesh)

        # rotary_emb is broadcast across the batch dimension
        rotary_emb = DP.get_assigned_chunk(rotary_emb, dim=-2, group=seq_mesh)

        with UnifiedAttnMode(mesh):
            hidden_states = self.call_transformer_blocks(
                hidden_states, encoder_hidden_states, timestep_proj, rotary_emb
            )

        # 5. Output norm, projection & unpatchify
        shift, scale = (self.scale_shift_table + temb.unsqueeze(1)).chunk(2, dim=1)
        hidden_states = (self.norm_out(hidden_states.float()) * (1 + scale) + shift).type_as(hidden_states)
        hidden_states = self.proj_out(hidden_states)

        hidden_states = DP.get_complete_tensor(hidden_states, dim=-2, group=seq_mesh)
        hidden_states = DP.get_complete_tensor(hidden_states, dim=0, group=batch_mesh)

        hidden_states = hidden_states.reshape(
            batch_size, post_patch_num_frames, post_patch_height, post_patch_width, p_t, p_h, p_w, -1
        )

        hidden_states = hidden_states.permute(0, 7, 1, 4, 2, 5, 3, 6)
        output = hidden_states.flatten(6, 7).flatten(4, 5).flatten(2, 3)

        if USE_PEFT_BACKEND:
            # remove `lora_scale` from each PEFT layer
            unscale_lora_layers(self, lora_scale)

        if not return_dict:
            return (output,)

        return Transformer2DModelOutput(sample=output)

    transformer.forward = new_forward.__get__(transformer)

    def call_transformer_blocks(self, hidden_states, encoder_hidden_states, *args, **kwargs):
        if torch.is_grad_enabled() and self.gradient_checkpointing:

            def create_custom_forward(module, return_dict=None):
                def custom_forward(*inputs):
                    if return_dict is not None:
                        return module(*inputs, return_dict=return_dict)
                    else:
                        return module(*inputs)

                return custom_forward

            ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False}

            for block in self.blocks:
                hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    hidden_states,
                    encoder_hidden_states,
                    *args,
                    **kwargs,
                    **ckpt_kwargs,
                )

        else:
            for block in self.blocks:
                hidden_states = block(hidden_states, encoder_hidden_states, *args, **kwargs)

        return hidden_states

    transformer.call_transformer_blocks = call_transformer_blocks.__get__(transformer)

    transformer._is_parallelized = True

    return transformer


def parallelize_pipe(pipe: DiffusionPipeline, *, shallow_patch: bool = False, **kwargs):
    if not getattr(pipe, "_is_parallelized", False):
        original_call = pipe.__class__.__call__

        @functools.wraps(original_call)
        def new_call(self, *args, generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None, **kwargs):
            if generator is None and getattr(self, "_is_parallelized", False):
                seed_t = torch.randint(0, torch.iinfo(torch.int64).max, [1], dtype=torch.int64, device=self.device)
                seed_t = DP.get_complete_tensor(seed_t, dim=0)
                seed_t = DP.get_assigned_chunk(seed_t, dim=0, idx=0)
                seed = seed_t.item()
                seed -= torch.iinfo(torch.int64).min
                generator = torch.Generator(self.device).manual_seed(seed)
            return original_call(self, *args, generator=generator, **kwargs)

        new_call._is_parallelized = True

        pipe.__class__.__call__ = new_call
        pipe.__class__._is_parallelized = True

    if not shallow_patch:
        parallelize_transformer(pipe.transformer, **kwargs)

    return pipe

def parallelize_transformer_sparge(transformer: NEW_WanTransformer3DModel, *, mesh=None):
    if getattr(transformer, "_is_parallelized", False):
        return transformer

    mesh = init_context_parallel_mesh(transformer.device.type, mesh=mesh)
    batch_mesh = mesh["batch"]
    seq_mesh = mesh["ring", "ulysses"]._flatten()

    transformer._total_attention_time = 0.0
    
    @functools.wraps(transformer.__class__.forward)
    def new_forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.LongTensor,
        encoder_hidden_states: torch.Tensor,
        encoder_hidden_states_image: Optional[torch.Tensor] = None,
        return_dict: bool = True,
        head_perm_idx:Optional[torch.Tensor]=None,
        head_deperm_idx:Optional[torch.Tensor]=None,
        new_row_perm_idx:Optional[list]=None,
        new_col_perm_idx:Optional[list]=None,
        transpose_matrix_q:Optional[list]=None,
        transpose_matrix_k:Optional[list]=None,
        new_row_deperm_idx:Optional[list]=None,
        ulysses_pg: Optional[torch.distributed.ProcessGroup] = None,
        ring_pg: Optional[torch.distributed.ProcessGroup] = None,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs,
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
                logger.warning("Passing `scale` via `attention_kwargs` when not using the PEFT backend is ineffective.")

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

        timestep_proj = DP.get_assigned_chunk(timestep_proj, dim=0, group=batch_mesh)
        temb = DP.get_assigned_chunk(temb, dim=0, group=batch_mesh)
        hidden_states = DP.get_assigned_chunk(hidden_states, dim=0, group=batch_mesh)
        hidden_states = DP.get_assigned_chunk(hidden_states, dim=-2, group=seq_mesh)
        encoder_hidden_states = DP.get_assigned_chunk(encoder_hidden_states, dim=0, group=batch_mesh)

        # rotary_emb is broadcast across the batch dimension
        rotary_emb = DP.get_assigned_chunk(rotary_emb, dim=-2, group=seq_mesh)

        # with UnifiedAttnMode(mesh):
        #     hidden_states, all_head_density = self.call_transformer_blocks(
        #         hidden_states, encoder_hidden_states, timestep_proj, rotary_emb
        #     )

        # import torch.distributed as dist
        # if(dist.get_rank()==0):
        #     import ipdb; ipdb.set_trace();

        total_time = 0.0
        if torch.is_grad_enabled() and self.gradient_checkpointing:
            all_head_density = []
            for layer_idx, block in enumerate(self.blocks):
                hidden_states, head_density = self._gradient_checkpointing_func(
                    block, hidden_states, encoder_hidden_states, timestep_proj, rotary_emb, head_perm_idx=head_perm_idx[layer_idx] if head_perm_idx is not None else None,
                    head_deperm_idx=head_deperm_idx[layer_idx] if head_deperm_idx is not None else None, new_row_perm_idx=new_row_perm_idx[layer_idx] if new_row_perm_idx is not None else None, new_col_perm_idx=new_col_perm_idx[layer_idx] if new_col_perm_idx is not None else None, transpose_matrix_q=transpose_matrix_q[layer_idx] if transpose_matrix_q is not None else None, transpose_matrix_k=transpose_matrix_k[layer_idx] if transpose_matrix_k is not None else None, new_row_deperm_idx=new_row_deperm_idx[layer_idx] if new_row_deperm_idx is not None else None, ulysses_pg=ulysses_pg, ring_pg=ring_pg, layer_idx=layer_idx,
                )
                all_head_density.append(head_density.squeeze(0))
                if hasattr(block.attn1.processor, "get_time_stats"):
                    total_time += block.attn1.processor.get_time_stats()['total_ms']
            all_head_density = torch.stack(all_head_density, dim=0)
        else:
            all_head_density = []
            # import ipdb; ipdb.set_trace();
            for layer_idx, block in enumerate(self.blocks):
                hidden_states, head_density = block(hidden_states, encoder_hidden_states, timestep_proj, rotary_emb, head_perm_idx=head_perm_idx[layer_idx] if head_perm_idx is not None else None,
                    head_deperm_idx=head_deperm_idx[layer_idx] if head_deperm_idx is not None else None, new_row_perm_idx=new_row_perm_idx[layer_idx] if new_row_perm_idx is not None else None, new_col_perm_idx=new_col_perm_idx[layer_idx] if new_col_perm_idx is not None else None, transpose_matrix_q=transpose_matrix_q[layer_idx] if transpose_matrix_q is not None else None, transpose_matrix_k=transpose_matrix_k[layer_idx] if transpose_matrix_k is not None else None, new_row_deperm_idx=new_row_deperm_idx[layer_idx] if new_row_deperm_idx is not None else None, ulysses_pg=ulysses_pg, ring_pg=ring_pg, layer_idx=layer_idx,)
                all_head_density.append(head_density.squeeze(0))
                if hasattr(block.attn1.processor, "get_time_stats"):
                    total_time += block.attn1.processor.get_time_stats()['total_ms']
            all_head_density = torch.stack(all_head_density, dim=0)
        
        # import torch.distributed as dist
        # if dist.get_rank() == 0:
        #     print(f"Transformer total attention time: {total_time} ms")
        self._total_attention_time = total_time

        # for ulysses
        if dist.get_world_size(ring_pg) == 1:
            head_density = DP.get_complete_tensor(all_head_density, dim=-1, group=seq_mesh)
        # for hybrid
        else:
            density_step1 = DP._all_gather_and_cat(all_head_density, dim=-3, group=ulysses_pg)
            head_density = DP._all_gather_and_cat(density_step1, dim=-2, group=ring_pg)


        # 5. Output norm, projection & unpatchify
        shift, scale = (self.scale_shift_table + temb.unsqueeze(1)).chunk(2, dim=1)
        hidden_states = (self.norm_out(hidden_states.float()) * (1 + scale) + shift).type_as(hidden_states)
        hidden_states = self.proj_out(hidden_states)

        hidden_states = DP.get_complete_tensor(hidden_states, dim=-2, group=seq_mesh)
        hidden_states = DP.get_complete_tensor(hidden_states, dim=0, group=batch_mesh)

        hidden_states = hidden_states.reshape(
            batch_size, post_patch_num_frames, post_patch_height, post_patch_width, p_t, p_h, p_w, -1
        )

        hidden_states = hidden_states.permute(0, 7, 1, 4, 2, 5, 3, 6)
        output = hidden_states.flatten(6, 7).flatten(4, 5).flatten(2, 3)

        if USE_PEFT_BACKEND:
            # remove `lora_scale` from each PEFT layer
            unscale_lora_layers(self, lora_scale)

        if not return_dict:
            return output, head_density

        return Transformer2DModelOutput(sample=output), head_density

    transformer.forward = new_forward.__get__(transformer)

    def get_total_attention_time(self):
        return getattr(self, "_total_attention_time", 0.0)

    transformer.get_total_attention_time = get_total_attention_time.__get__(transformer)

    def call_transformer_blocks(self, hidden_states, encoder_hidden_states, perm_idx, deperm_idx, *args, **kwargs):
        if torch.is_grad_enabled() and self.gradient_checkpointing:

            def create_custom_forward(module, return_dict=None):
                def custom_forward(*inputs):
                    if return_dict is not None:
                        return module(*inputs, return_dict=return_dict)
                    else:
                        return module(*inputs)

                return custom_forward

            ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False}

            all_head_density = []
            for block in self.blocks:
                hidden_states, head_density = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    hidden_states,
                    encoder_hidden_states,
                    perm_idx,
                    deperm_idx,
                    *args,
                    **kwargs,
                    **ckpt_kwargs,
                )
                all_head_density.append(head_density)
            all_head_density = torch.stack(all_head_density, dim=0)

        else:
            all_head_density = []
            for block in self.blocks:
                hidden_states, head_density = block(hidden_states, encoder_hidden_states, perm_idx, deperm_idx, *args, **kwargs)
                all_head_density.append(head_density)
            all_head_density = torch.stack(all_head_density, dim=0)

        return hidden_states, all_head_density

    transformer.call_transformer_blocks = call_transformer_blocks.__get__(transformer)

    transformer._is_parallelized = True

    return transformer


def parallelize_pipe_sparge(pipe: DiffusionPipeline, *, shallow_patch: bool = False, **kwargs):
    if not getattr(pipe, "_is_parallelized", False):
        original_call = pipe.__class__.__call__

        @functools.wraps(original_call)
        def new_call(self, *args, generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None, **kwargs):
            if generator is None and getattr(self, "_is_parallelized", False):
                seed_t = torch.randint(0, torch.iinfo(torch.int64).max, [1], dtype=torch.int64, device=self.device)
                seed_t = DP.get_complete_tensor(seed_t, dim=0)
                seed_t = DP.get_assigned_chunk(seed_t, dim=0, idx=0)
                seed = seed_t.item()
                seed -= torch.iinfo(torch.int64).min
                generator = torch.Generator(self.device).manual_seed(seed)
            return original_call(self, *args, generator=generator, **kwargs)

        new_call._is_parallelized = True

        pipe.__class__.__call__ = new_call
        pipe.__class__._is_parallelized = True

    if not shallow_patch:
        parallelize_transformer_sparge(pipe.transformer, **kwargs)

    return pipe

def parallelize_transformer_paro(transformer: PARO_WanTransformer3DModel, *, mesh=None):
    if getattr(transformer, "_is_parallelized", False):
        return transformer

    mesh = init_context_parallel_mesh(transformer.device.type, mesh=mesh)
    batch_mesh = mesh["batch"]
    seq_mesh = mesh["ring", "ulysses"]._flatten()

    transformer._total_attention_time = 0.0
    
    @functools.wraps(transformer.__class__.forward)
    def new_forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.LongTensor,
        encoder_hidden_states: torch.Tensor,
        encoder_hidden_states_image: Optional[torch.Tensor] = None,
        return_dict: bool = True,
        sparse: Optional[torch.Tensor] = None,
        head_perm_idx:Optional[torch.Tensor]=None,
        head_deperm_idx:Optional[torch.Tensor]=None,
        new_row_perm_idx:Optional[list]=None,
        new_col_perm_idx:Optional[list]=None,
        transpose_matrix_q:Optional[list]=None,
        transpose_matrix_k:Optional[list]=None,
        new_row_deperm_idx:Optional[list]=None,
        ulysses_pg: Optional[torch.distributed.ProcessGroup] = None,
        ring_pg: Optional[torch.distributed.ProcessGroup] = None,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs,
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
                logger.warning("Passing `scale` via `attention_kwargs` when not using the PEFT backend is ineffective.")

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

        timestep_proj = DP.get_assigned_chunk(timestep_proj, dim=0, group=batch_mesh)
        temb = DP.get_assigned_chunk(temb, dim=0, group=batch_mesh)
        hidden_states = DP.get_assigned_chunk(hidden_states, dim=0, group=batch_mesh)
        hidden_states = DP.get_assigned_chunk(hidden_states, dim=-2, group=seq_mesh)
        encoder_hidden_states = DP.get_assigned_chunk(encoder_hidden_states, dim=0, group=batch_mesh)

        # rotary_emb is broadcast across the batch dimension
        rotary_emb = DP.get_assigned_chunk(rotary_emb, dim=-2, group=seq_mesh)

        sparse = DP.get_assigned_chunk(sparse, dim=-2, group=seq_mesh)

        # with UnifiedAttnMode(mesh):
        #     hidden_states, all_head_density = self.call_transformer_blocks(
        #         hidden_states, encoder_hidden_states, timestep_proj, rotary_emb
        #     )

        # import torch.distributed as dist
        # if(dist.get_rank()==0):
        #     import ipdb; ipdb.set_trace();

        total_time = 0.0
        if torch.is_grad_enabled() and self.gradient_checkpointing:
            for layer_idx, block in enumerate(self.blocks):
                hidden_states = self._gradient_checkpointing_func(
                    block, hidden_states, encoder_hidden_states, timestep_proj, rotary_emb, sparse=sparse[layer_idx].unsqueeze(0), head_perm_idx=head_perm_idx[layer_idx] if head_perm_idx is not None else None,
                    head_deperm_idx=head_deperm_idx[layer_idx] if head_deperm_idx is not None else None, new_row_perm_idx=new_row_perm_idx[layer_idx] if new_row_perm_idx is not None else None, new_col_perm_idx=new_col_perm_idx[layer_idx] if new_col_perm_idx is not None else None, transpose_matrix_q=transpose_matrix_q[layer_idx] if transpose_matrix_q is not None else None, transpose_matrix_k=transpose_matrix_k[layer_idx] if transpose_matrix_k is not None else None, new_row_deperm_idx=new_row_deperm_idx[layer_idx] if new_row_deperm_idx is not None else None, ulysses_pg=ulysses_pg, ring_pg=ring_pg, layer_idx=layer_idx,
                )
                if hasattr(block.attn1.processor, "get_time_stats"):
                    total_time += block.attn1.processor.get_time_stats()['total_ms']
  
        else:
            # import ipdb; ipdb.set_trace();
            for layer_idx, block in enumerate(self.blocks):
                hidden_states = block(hidden_states, encoder_hidden_states, timestep_proj, rotary_emb, sparse=sparse[layer_idx].unsqueeze(0), head_perm_idx=head_perm_idx[layer_idx] if head_perm_idx is not None else None,
                head_deperm_idx=head_deperm_idx[layer_idx] if head_deperm_idx is not None else None, 
                new_row_perm_idx=new_row_perm_idx[layer_idx] if new_row_perm_idx is not None else None, new_col_perm_idx=new_col_perm_idx[layer_idx] if new_col_perm_idx is not None else None, transpose_matrix_q=transpose_matrix_q[layer_idx] if transpose_matrix_q is not None else None, transpose_matrix_k=transpose_matrix_k[layer_idx] if transpose_matrix_k is not None else None, new_row_deperm_idx=new_row_deperm_idx[layer_idx] if new_row_deperm_idx is not None else None, ulysses_pg=ulysses_pg, ring_pg=ring_pg)
                if hasattr(block.attn1.processor, "get_time_stats"):
                    total_time += block.attn1.processor.get_time_stats()['total_ms']

        # import torch.distributed as dist
        # if dist.get_rank() == 0:
        #     print(f"Transformer total attention time: {total_time} ms")
        self._total_attention_time = total_time

        # 5. Output norm, projection & unpatchify
        shift, scale = (self.scale_shift_table + temb.unsqueeze(1)).chunk(2, dim=1)
        hidden_states = (self.norm_out(hidden_states.float()) * (1 + scale) + shift).type_as(hidden_states)
        hidden_states = self.proj_out(hidden_states)

        hidden_states = DP.get_complete_tensor(hidden_states, dim=-2, group=seq_mesh)
        hidden_states = DP.get_complete_tensor(hidden_states, dim=0, group=batch_mesh)

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

    transformer.forward = new_forward.__get__(transformer)

    def get_total_attention_time(self):
        return getattr(self, "_total_attention_time", 0.0)

    transformer.get_total_attention_time = get_total_attention_time.__get__(transformer)

    def call_transformer_blocks(self, hidden_states, encoder_hidden_states,sparse, head_perm_idx, head_deperm_idx, new_row_perm_idx, new_col_perm_idx, transpose_matrix_q, transpose_matrix_k, new_row_deperm_idx, ulysses_pg, ring_pg, *args, **kwargs):
        if torch.is_grad_enabled() and self.gradient_checkpointing:

            def create_custom_forward(module, return_dict=None):
                def custom_forward(*inputs):
                    if return_dict is not None:
                        return module(*inputs, return_dict=return_dict)
                    else:
                        return module(*inputs)

                return custom_forward

            ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False}

            for block in self.blocks:
                hidden_states= torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    hidden_states,
                    encoder_hidden_states,
                    sparse,
                    head_perm_idx, 
                    head_deperm_idx, 
                    new_row_perm_idx, 
                    new_col_perm_idx, 
                    transpose_matrix_q, 
                    transpose_matrix_k, 
                    new_row_deperm_idx,
                    ulysses_pg, 
                    ring_pg,
                    *args,
                    **kwargs,
                    **ckpt_kwargs,
                )

        else:
            for block in self.blocks:
                hidden_states= block(hidden_states, encoder_hidden_states, sparse, head_perm_idx, head_deperm_idx, new_row_perm_idx, new_col_perm_idx, transpose_matrix_q, transpose_matrix_k, new_row_deperm_idx, ulysses_pg, ring_pg, *args, **kwargs)

        return hidden_states

    transformer.call_transformer_blocks = call_transformer_blocks.__get__(transformer)

    transformer._is_parallelized = True

    return transformer


def parallelize_pipe_paro(pipe: DiffusionPipeline, *, shallow_patch: bool = False, **kwargs):
    if not getattr(pipe, "_is_parallelized", False):
        original_call = pipe.__class__.__call__

        @functools.wraps(original_call)
        def new_call(self, *args, generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None, **kwargs):
            if generator is None and getattr(self, "_is_parallelized", False):
                seed_t = torch.randint(0, torch.iinfo(torch.int64).max, [1], dtype=torch.int64, device=self.device)
                seed_t = DP.get_complete_tensor(seed_t, dim=0)
                seed_t = DP.get_assigned_chunk(seed_t, dim=0, idx=0)
                seed = seed_t.item()
                seed -= torch.iinfo(torch.int64).min
                generator = torch.Generator(self.device).manual_seed(seed)
            return original_call(self, *args, generator=generator, **kwargs)

        new_call._is_parallelized = True

        pipe.__class__.__call__ = new_call
        pipe.__class__._is_parallelized = True

    if not shallow_patch:
        parallelize_transformer_paro(pipe.transformer, **kwargs)

    return pipe
