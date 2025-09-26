import torch
from torch import nn
import sys
import numpy as np
from transformers import BertForQuestionAnswering, BertTokenizer, BertModel
import bert_binding
import os

def simple_souffle_test():
    # 重置CUDA
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    # 加载原始模型
    model_name = "bert-base-uncased"
    print(f"Loading model: {model_name}")

    tokenizer = BertTokenizer.from_pretrained(model_name, torch_dtype=torch.float16)
    original_model = BertModel.from_pretrained(model_name, torch_dtype=torch.float16)
    original_model.eval()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if torch.cuda.is_available():
        original_model = original_model.cuda()

    text = "Hello, I am a sentence for BERT."
    inputs = tokenizer(text, return_tensors='pt', max_length=384, truncation=True, padding='max_length')
    if torch.cuda.is_available():
        inputs = {k: v.to('cuda') for k, v in inputs.items()}
    with torch.no_grad():
        input_embeddings = original_model.embeddings(
            input_ids=inputs["input_ids"],
            token_type_ids=inputs.get("token_type_ids"),
        ).to(device)

    # =========================================================================
    # START: 关键修改部分 - 将输入矩阵替换为所有元素为 0.125 的矩阵
    # =========================================================================
    print("\n--- 调试模式: 替换 Input Embeddings 为常数矩阵 (0.125) ---")

    # 使用 torch.full_like() 创建一个与 input_embeddings 形状、dtype、device 相同，
    # 但所有元素值为 0.125 的新张量
    input_embeddings = torch.full_like(
        input_embeddings,
        0.125,
        dtype=input_embeddings.dtype,
        device=input_embeddings.device
    )
    # =========================================================================
    # END: 关键修改部分
    # =========================================================================

    layer_6 = original_model.encoder.layer[5]

    attention = layer_6.attention.self

    # 提取权重 (PyTorch Linear层格式: [out_features, in_features])
    q_weight = attention.query.weight.data.clone()  # [768, 768]
    k_weight = attention.key.weight.data.clone()  # [768, 768]
    v_weight = attention.value.weight.data.clone()  # [768, 768]

    # ----------------------------------------------
    # 关键修改：将所有元素原地设置为 0.01
    # ----------------------------------------------
    q_weight.fill_(0.01)
    k_weight.fill_(0.01)
    v_weight.fill_(0.01)

    attn_output_weight = layer_6.attention.output.dense.weight.data.clone()  # [768, 768]

    ff_fc1_weight = layer_6.intermediate.dense.weight.data.clone()  # [3072, 768]
    ff_fc2_weight = layer_6.output.dense.weight.data.clone()  # [768, 3072]

    # 运行原始模型第一层获取参考输出
    print("\n--- Running Original Model Layer 5 ---")
    with torch.no_grad():
        test_configs = [

            {
                'name': 'Config 8',
                'qkv': torch.stack([q_weight.t(), k_weight.t(), v_weight], dim=0),
                'attn_fc': attn_output_weight,
                'ff1': ff_fc1_weight.t(),
                'ff2': ff_fc2_weight.t()
            },

        ]

        for config in test_configs:
            print(f"\n--- Testing {config['name']} ---")

            # 转换权重到FP16和正确设备
            qkv_w = config['qkv'].to(torch.float16).to(device)
            attn_w = config['attn_fc'].to(torch.float16).to(device)
            ff1_w = config['ff1'].to(torch.float16).to(device)
            ff2_w = config['ff2'].to(torch.float16).to(device)

            # 调用 souffle_bert_layer
            souffle_output = bert_binding.souffle_bert_layer(
                input_embeddings,
                qkv_w,
                attn_w,
                ff1_w,
                ff2_w,
                4  # opt_level
            )

            final_output = souffle_output[1]

simple_souffle_test()