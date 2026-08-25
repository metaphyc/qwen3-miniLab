"""实验 1：观察 Qwen3-0.6B-Base 的静态结构。"""

from pathlib import Path

import torch
from transformers import AutoModelForCausalLM


MODEL_PATH = Path(__file__).parent.parent / "models" / "Qwen3-0.6B-Base"


def main() -> None:
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        dtype=torch.bfloat16,
    ).cpu().eval()
    config = model.config
    layer = model.model.layers[0]

    print("=== 顶层结构 ===")
    print(f"模型类: {type(model).__name__}")
    print(f"总参数量: {sum(parameter.numel() for parameter in model.parameters()):,}")
    print(f"词表大小: {config.vocab_size:,}")
    print(f"隐藏向量维度: {config.hidden_size}")
    print(f"Transformer block 数量: {config.num_hidden_layers}")
    print(f"注意力头: Q={config.num_attention_heads}, KV={config.num_key_value_heads}")
    print(f"每头维度: {config.head_dim}")
    print(f"MLP 中间维度: {config.intermediate_size}")
    print()

    print("Qwen3ForCausalLM")
    print("├── model: Qwen3Model")
    print(f"│   ├── embed_tokens: Embedding({config.vocab_size}, {config.hidden_size})")
    print(f"│   ├── layers: ModuleList × {len(model.model.layers)}")
    print(f"│   └── norm: {model.model.norm}")
    print(f"└── lm_head: Linear({config.hidden_size}, {config.vocab_size}, bias=False)")
    print()

    print("=== 第 0 个 Transformer block ===")
    print(layer)
    print()

    print("=== 关键权重形状 ===")
    parameters = (
        ("token embedding", model.model.embed_tokens.weight),
        ("Q projection", layer.self_attn.q_proj.weight),
        ("K projection", layer.self_attn.k_proj.weight),
        ("V projection", layer.self_attn.v_proj.weight),
        ("attention output projection", layer.self_attn.o_proj.weight),
        ("MLP gate projection", layer.mlp.gate_proj.weight),
        ("MLP up projection", layer.mlp.up_proj.weight),
        ("MLP down projection", layer.mlp.down_proj.weight),
        ("language-model head", model.lm_head.weight),
    )
    for name, parameter in parameters:
        print(f"{name:28} {tuple(parameter.shape)}")

    shared_weights = model.model.embed_tokens.weight.data_ptr() == model.lm_head.weight.data_ptr()
    print()
    print(f"输入 embedding 与 lm_head 是否共享同一块权重: {shared_weights}")


if __name__ == "__main__":
    main()
