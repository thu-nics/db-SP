torchrun --nproc_per_node=8 ./test/ae.py --sp_ulysses_degree 8 --ring_impl_type "basic" --attn_impl paro --head_idx 31 --output_csv ./attention_speedup_ulysses.csv
torchrun --nproc_per_node=8 ./test/ae.py --sp_ulysses_degree 8 --ring_impl_type "basic" --attn_impl paro --head_idx 17 --output_csv ./attention_speedup_ulysses.csv
torchrun --nproc_per_node=8 ./test/ae.py --sp_ulysses_degree 8 --ring_impl_type "basic" --attn_impl paro --head_idx 39 --output_csv ./attention_speedup_ulysses.csv
torchrun --nproc_per_node=8 ./test/ae.py --sp_ulysses_degree 8 --ring_impl_type "basic" --attn_impl paro --head_idx 35 --output_csv ./attention_speedup_ulysses.csv
torchrun --nproc_per_node=8 ./test/ae.py --sp_ulysses_degree 8 --ring_impl_type "basic" --attn_impl paro --head_idx 15 --output_csv ./attention_speedup_ulysses.csv
torchrun --nproc_per_node=8 ./test/ae.py --sp_ulysses_degree 8 --ring_impl_type "basic" --attn_impl paro --head_idx 33 --output_csv ./attention_speedup_ulysses.csv
torchrun --nproc_per_node=8 ./test/ae.py --sp_ulysses_degree 8 --ring_impl_type "basic" --attn_impl paro --head_idx 36 --output_csv ./attention_speedup_ulysses.csv