# DyLLM

DyLLM selects salient tokens after attention to remove redundant computations in FFN and use approximate attention enlightening the attention operation. Without hurting the accuracy of the original implementation, DyLLM achieve ~9.6x higher throughput.

## How to install 

```
conda create --name dyllm python=3.10 -y
conda activate dyllm
bash setup_env.sh
```

## How to run

```
python run.py
```

### DiffusionGemma

DiffusionGemma checkpoints are supported with Transformers 4.57.6 and newer,
including the native DiffusionGemma implementation in Transformers 5:

```python
from dyllm import SamplingParams, dLLM

model_path = "/data/models/diffusiongemma-26B-A4B-it"
engine = dLLM(model_path, threshold=None, tensor_parallel_size=1)
prompt = engine.tokenizer.apply_chat_template(
    [{"role": "user", "content": "Reply with only: OK"}],
    tokenize=False,
    add_generation_prompt=True,
)
outputs = engine.generate(
    [prompt],
    SamplingParams(max_new_tokens=256),
)
engine.exit()
```

DiffusionGemma currently supports one GPU. Set `threshold` to a cosine
similarity threshold such as `0.99` to enable the saliency/sparse-attention
path; use `None` for dense attention.

### Tensor and expert parallelism

For dense LLaDA and Dream models, `tensor_parallel_size` shards attention,
dense MLP, embeddings, and the LM head. Sparse-attention cosine decisions are
made from an all-reduce of three FP32 statistics per token (dot product and two
squared norms), so TP ranks select exactly the same salient rows without
gathering context vectors.

LLaDA-MoE follows vLLM's two MoE layouts:

```python
# Every expert is tensor-parallel across the two ranks.
engine = dLLM(model_path, threshold=0.99, tensor_parallel_size=2)

# Experts are placed contiguously across the same two ranks (32 of 64 per GPU).
engine = dLLM(
    model_path,
    threshold=0.99,
    tensor_parallel_size=2,
    expert_parallel_size=2,
)
```

`expert_parallel_size` defaults to `1`. Since DyLLM does not yet expose a
separate data-parallel dimension, EP is either disabled (`1`) or must equal
`tensor_parallel_size`.

## Algorithm

![approximate attetion](assets/approximate_attention.png)

After attention context operation, DyLLM compares the cosine similarity of context activation of each token with the same activation from the previous step.
If the similarity is smaller than the given $\tau$, the token is selected as **salient token**.
Only the salient tokens are computed in FFN significantly reducing the computational overhead.

We further reduce the runtime by focusing more on repsonse tokens. 
DyLLM basically picks salient tokens from the response tokens and attends the whole sentence periodically.


### Overall Comparison

![result table](assets/result_table.png)

![scalability](assets/eight_plots.png)

### Commands to reproduce 

```
bash ./scripts/run_gsm8k_acc_llada.sh # accuracy test
bash ./scripts/run_gsm8k_llada.sh # throughput test
```

## Citation

If you find our code useful, please cite our paper.

```bibtex
@inproceedings{dyllm2026,
    title={Dy{LLM}: Efficient Diffusion {LLM} inference via saliency-based token selection and partial attention},
    author={Younjoo Lee and Seungkyun Dan and Junghoo Lee and Jaiyoung Park and {Jung Ho} Ahn},
    booktitle={Forty-third International Conference on Machine Learning},
    year={2026},
    url={https://openreview.net/forum?id=0azUrmsSyA}
}
```
