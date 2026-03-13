import torch
# from flash_attn.utils.benchmark import benchmark_forward
from spas_sage_attn.utils import hyperparameter_check, get_block_map_meansim, get_block_map_meansim_fuse_quant
import spas_sage_attn._qattn as qattn
import torch.nn.functional as F
# import sageattention._qattn_sm80 as qattn
from torch.nn.functional import scaled_dot_product_attention as sdpa
import argparse
# from sageattention.triton.attn_qk_int8_per_block import forward
# from sageattention.triton.attn_qk_int8_per_block_causal import forward as forward_causal


parser = argparse.ArgumentParser(description='Benchmark QK Int8 PV FP16')
parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
parser.add_argument('--num_heads', type=int, default=48, help='Number of heads')
parser.add_argument('--head_dim', type=int, default=64, help='Head dimension')
# parser.add_argument('--quant_gran', type=str, default='per_warp', choices=['per_warp', 'per_thread'], help='Quantization granularity')
# parser.add_argument('--pv_accum_dtype', type=str, default='fp16', choices=['fp16', 'fp16+fp32', 'fp32'])
parser.add_argument('--method', type=str, default='sparge', choices=['sparge', 'fa2','sageattn'])
parser.add_argument('--use_calib_data', type=str, default='none', choices=['cogvideox', 'hyvideo','none'])
args = parser.parse_args()
method = args.method
use_calib_data = args.use_calib_data

is_causal = False
_is_causal = 1 if is_causal else 0
print(f"is_causal: {is_causal}")
device = torch.device("cuda")

#load calib_data
if(use_calib_data != 'none'):
    if(use_calib_data == 'cogvideox'):
        data = torch.load("/home/chensiqi/qkv_5_1.pth", map_location='cuda')
        q_list = []
        k_list = []
        v_list = []
        for block_q in data['q']:
            selected = block_q[0, 0, 0, :, :17536, :]  # [48, 17550, 64]
            q_list.append(selected)
        for block_k in data['k']:
            selected = block_k[0, 0, 0, :, :17536, :]  # [48, 17536, 64]
            k_list.append(selected)
        for block_v in data['v']:
            selected = block_v[0, 0, 0, :, :17536, :]  # [48, 17536, 64]
            v_list.append(selected)

        q = torch.stack(q_list, dim=0)  # [42, 48, 17536, 64]
        k = torch.stack(k_list, dim=0)  # [42, 48, 17536, 64]
        v = torch.stack(v_list, dim=0)  # [42, 48, 17536, 64]
        q=q.to(device)
        k=k.to(device)
        v=v.to(device).to(torch.float16)
        dtype = q.dtype
        if dtype == torch.float32 or dtype == torch.float16:
            q, k, v = q.contiguous().to(torch.float16), k.contiguous().to(torch.float16), v.contiguous().to(torch.float16)
        else:
            q, k, v = q.contiguous().to(torch.bfloat16), k.contiguous().to(torch.bfloat16), v.contiguous().to(torch.float16)
    
    elif(use_calib_data == 'hyvideo'):
        data = torch.load("/home/zhangyichong/Sparse-VideoGen/attention_matrices.pt")
        q = data['query'].to(device)[:, :, :119040, :] #torch.Size([1, 24, 119056, 128]) change it to [1, 24, 119040, 128]
        k = data['key'].to(device)[:, :, :119040, :]
        v = data['value'].to(device)[:, :, :119040, :]
        dtype = q.dtype
        if dtype == torch.float32 or dtype == torch.float16:
            q, k, v = q.contiguous().to(torch.float16), k.contiguous().to(torch.float16), v.contiguous().to(torch.float16)
        else:
            q, k, v = q.contiguous().to(torch.bfloat16), k.contiguous().to(torch.bfloat16), v.contiguous().to(torch.float16)
    
    # print(q.dtype)
    # print(k.dtype)
    # print(v.dtype)
    batch = q.size(0)
    head = q.size(1)
    headdim = q.size(3)
    WARP_Q = 32 if (headdim == 64) else 16
    WARP_K = 64
    # kernel calculate
    lut, valid_block_num, q_int8, q_scale, k_int8, k_scale = get_block_map_meansim_fuse_quant(q, k, None, is_causal=is_causal, simthreshd1=0.1, cdfthreshd=0.85, return_lut=True, attention_sink=False)  
    print(f"lut size: {lut.size()}, valid_block_num size: {valid_block_num.size()}")
    pvthreshd = 0.05
    pvthreshd = hyperparameter_check(pvthreshd, q.size(-3), q.device)
    qk_sparsity = 0


    # generate random data

for seq_len in {17792}: # seq_len must be the multiple of 128
    print(f'seq_len: {seq_len}')
    if(use_calib_data == 'none'):
        batch_size=args.batch_size
        num_heads=args.num_heads
        head_dim=args.head_dim
        head = num_heads
        batch = batch_size
        headdim = head_dim
        if(method == 'sageattn' or method == 'sparge'):
            q = torch.randint(-100, 100, (batch_size, num_heads, seq_len, head_dim), dtype=torch.int8, device='cuda')
            k = torch.randint(-100, 100, (batch_size, num_heads, seq_len, head_dim), dtype=torch.int8, device='cuda')
            v = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=torch.float16, device='cuda')
        elif(method == 'fa2'):
            q = torch.randint(-100, 100, (batch_size, num_heads, seq_len, head_dim), dtype=torch.bfloat16, device='cuda').transpose(1,2).contiguous()
            k = torch.randint(-100, 100, (batch_size, num_heads, seq_len, head_dim), dtype=torch.bfloat16, device='cuda').transpose(1,2).contiguous()
            v = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=torch.bfloat16, device='cuda').transpose(1,2).contiguous()

        q_scale = torch.randn(batch_size, num_heads, (seq_len // 128), 1, dtype=torch.float16, device='cuda')
        k_scale = torch.randn(batch_size, num_heads, (seq_len // 64), 1, dtype=torch.float16, device='cuda')

    if(method=='sparge'):
        flops = 4 * head * batch * headdim * seq_len * seq_len / (2 if is_causal else 1)
        print(f"SpargeAttention: QK Int8 PV FP16 Benchmark")
        print(f"batch: {batch}, head: {head}, headdim: {headdim}")
        kernel = qattn.qk_int8_sv_f16_accum_f16_block_sparse_attn_inst_buf_with_pv_threshold 
        # _qk_quant_gran = 3 if args.quant_gran == 'per_thread' else 2
        o = torch.empty_like(q)
        sm_scale = 1 / (headdim ** 0.5)
        
        for i in range(5):
            kernel(q_int8, k_int8, v, o, lut, valid_block_num, pvthreshd, q_scale, k_scale, 1, _is_causal, 1, sm_scale, 0)
            torch.cuda.synchronize()
        _, time = benchmark_forward(kernel, q_int8, k_int8, v, o, lut, valid_block_num, pvthreshd, q_scale, k_scale, 1, _is_causal, 1, sm_scale, 0, repeats=100, verbose=False, desc='Triton')
        if is_causal is False:
            qk_sparsity = (valid_block_num.float().sum()) / (lut.size(3) * lut.size(2) * lut.size(0) * lut.size(1))
        else:
            qk_sparsity = (valid_block_num.float().sum()) / ((lut.size(3) + 2) // 2 * lut.size(2) * lut.size(0) * lut.size(1))
        print(f'sparge attention sparsity:{qk_sparsity.item()}, flops:{flops/time.mean*1e-12}, latency: {time.mean*1e3}')
        print("*sparse ratio means the ratio of calculated elements to all elements in the attention map")
    if(method=='fa2'):
        lut, valid_block_num, q_int8, q_scale, k_int8, k_scale = get_block_map_meansim_fuse_quant(q, k, None, is_causal=is_causal, simthreshd1=0.1, cdfthreshd=0.85, return_lut=True, attention_sink=False)  
        print(f"lut size: {lut.shape}, valid_block_num size: {valid_block_num.squeeze(0).sum(dim=-1)}")

        print("baseline:flashattention2")
        batch = q.size(0)
        head = q.size(1)
        headdim = q.size(3)
        flops = 4 * head * batch * headdim * seq_len * seq_len / (2 if is_causal else 1)
        print(f"batch: {batch}, head: {head}, headdim: {headdim}")
        for i in range(5): sdpa(q, k, v.to(torch.bfloat16), is_causal=is_causal)
        torch.cuda.synchronize()
        _, time = benchmark_forward(sdpa, q, k, v.to(torch.bfloat16), is_causal=is_causal, repeats=100, verbose=False, desc='Triton')
        print(f'flashattention flops:{flops/time.mean*1e-12}, latency:{time.mean*1e3}')

    if(method=='sageattn'):
        print(f"SageAttention: QK Int8 PV FP16 Benchmark")
        batch = q.size(0)
        head = q.size(1)
        headdim = q.size(3)
        print(f"batch: {batch}, head: {head}, headdim: {headdim}")
        flops = 4 * head * batch * headdim * seq_len * seq_len / (2 if is_causal else 1)
        q = torch.randint(-100, 100, (batch, head, seq_len, headdim), dtype=torch.int8, device='cuda')
        k = torch.randint(-100, 100, (batch, head, seq_len, headdim), dtype=torch.int8, device='cuda')
        v = torch.randn(batch, head, seq_len, headdim, dtype=torch.float16, device='cuda')

        q_scale = torch.randn(batch, head, (seq_len // 128), 1, dtype=torch.float16, device='cuda')
        k_scale = torch.randn(batch, head, (seq_len // 64), 1, dtype=torch.float16, device='cuda')
        for i in range(5): forward(q, k, v, q_scale, k_scale, output_dtype=torch.bfloat16)
        torch.cuda.synchronize()
        _, time = benchmark_forward(forward, q, k, v, q_scale, k_scale, output_dtype=torch.bfloat16, repeats=100, verbose=False, desc='Triton')
        print(f'sageattention flops:{flops/time.mean*1e-12}, latency:{time.mean*1e3}')