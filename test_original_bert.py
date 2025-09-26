import torch
import torch.nn as nn
from transformers import BertForQuestionAnswering, BertTokenizer, BertModel
import numpy as np


def print_tensor_info(tensor, name):
    """打印tensor的详细信息"""
    print(f"=== {name} ===")
    print(f"Shape: {list(tensor.shape)}")
    print(f"Dtype: {tensor.dtype}")
    print(f"Device: {tensor.device}")

    # 转换为float32进行统计计算
    tensor_fp32 = tensor.float()
    print(f"Min: {tensor_fp32.min().item():.6f}")
    print(f"Max: {tensor_fp32.max().item():.6f}")
    print(f"Mean: {tensor_fp32.mean().item():.6f}")
    print(f"Std: {tensor_fp32.std().item():.6f}")

    # 打印前10个值
    flattened = tensor_fp32.flatten()
    print(f"First 10 values: {flattened[:10].tolist()}")
    print()


class DebugBertSelfAttention(nn.Module):
    """带调试输出的BERT Self-Attention"""

    def __init__(self, original_attn):
        super().__init__()
        self.query = original_attn.query
        self.key = original_attn.key
        self.value = original_attn.value
        self.dropout = original_attn.dropout
        self.num_attention_heads = original_attn.num_attention_heads
        self.attention_head_size = original_attn.attention_head_size
        self.all_head_size = original_attn.all_head_size

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask=None, head_mask=None, output_attentions=False):
        print("\n========== Self-Attention Forward Pass ==========")
        print_tensor_info(hidden_states, "Input Hidden States")

        # 1. Linear projections for Q, K, V
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        print_tensor_info(mixed_query_layer, "Query Linear Output (before reshape)")
        print_tensor_info(mixed_key_layer, "Key Linear Output (before reshape)")
        print_tensor_info(mixed_value_layer, "Value Linear Output (before reshape)")

        # 2. Reshape for multi-head attention
        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # 1. 替换 query_layer
        query_layer = torch.full_like(
            query_layer,
            0.01,
            dtype=query_layer.dtype,
            device=query_layer.device
        )

        # 2. 替换 key_layer
        key_layer = torch.full_like(
            key_layer,
            0.01,
            dtype=key_layer.dtype,
            device=key_layer.device
        )

        # 3. 替换 value_layer
        value_layer = torch.full_like(
            value_layer,
            0.01,
            dtype=value_layer.dtype,
            device=value_layer.device
        )

        print_tensor_info(query_layer, "Query Layer (after multi-head reshape)")
        print_tensor_info(key_layer, "Key Layer (after multi-head reshape)")
        print_tensor_info(value_layer, "Value Layer (after multi-head reshape)")

        # 3. Attention scores (Q * K^T)
        print_tensor_info(key_layer.transpose(-1, -2), "key_layer.transpose(-1, -2)")
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        print_tensor_info(attention_scores, "torch.matmul(query_layer, key_layer.transpose(-1, -2))")

        attention_scores = attention_scores / (self.attention_head_size ** 0.5)

        print_tensor_info(attention_scores, "Attention Scores (Q*K^T / sqrt(d_k))")

        # # 4. Apply attention mask if provided
        # if attention_mask is not None:
        #     attention_scores = attention_scores + attention_mask
        #     print_tensor_info(attention_scores, "Attention Scores (after mask)")

        # 5. Softmax
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        print_tensor_info(attention_probs, "Attention Probabilities (after softmax)")

        # # 6. Dropout
        # attention_probs = self.dropout(attention_probs)
        # print_tensor_info(attention_probs, "Attention Probabilities (after dropout)")

        # 7. Apply attention to values
        context_layer = torch.matmul(attention_probs, value_layer)
        print_tensor_info(context_layer, "Context Layer (Attention * Value)")

        # 8. Reshape back
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        print_tensor_info(context_layer, "Context Layer (after reshape)")

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)
        return outputs


class DebugBertSelfOutput(nn.Module):
    """带调试输出的BERT Self-Output"""

    def __init__(self, original_output):
        super().__init__()
        self.dense = original_output.dense
        self.LayerNorm = original_output.LayerNorm
        self.dropout = original_output.dropout

    def forward(self, hidden_states, input_tensor):
        print("\n========== Self-Output Forward Pass ==========")
        print_tensor_info(hidden_states, "Self-Attention Output (before output projection)")
        print_tensor_info(input_tensor, "Input Tensor (for residual connection)")

        # 1. Linear projection
        hidden_states = self.dense(hidden_states)
        print_tensor_info(hidden_states, "After Output Linear Projection")

        # 2. Dropout
        hidden_states = self.dropout(hidden_states)
        print_tensor_info(hidden_states, "After Dropout")

        # 3. Residual connection + LayerNorm
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        print_tensor_info(hidden_states, "After Residual + LayerNorm")

        return hidden_states


class DebugBertIntermediate(nn.Module):
    """带调试输出的BERT Intermediate (FFN第一层)"""

    def __init__(self, original_intermediate):
        super().__init__()
        self.dense = original_intermediate.dense
        self.intermediate_act_fn = original_intermediate.intermediate_act_fn

    def forward(self, hidden_states):
        print("\n========== Intermediate (FFN FC1) Forward Pass ==========")
        print_tensor_info(hidden_states, "Input to FFN FC1")

        # 1. Linear transformation
        hidden_states = self.dense(hidden_states)
        print_tensor_info(hidden_states, "After FFN FC1 Linear")

        # 2. Activation function (usually GELU)
        hidden_states = self.intermediate_act_fn(hidden_states)
        print_tensor_info(hidden_states, "After FFN FC1 Activation (GELU)")

        return hidden_states


class DebugBertOutput(nn.Module):
    """带调试输出的BERT Output (FFN第二层)"""

    def __init__(self, original_output):
        super().__init__()
        self.dense = original_output.dense
        self.LayerNorm = original_output.LayerNorm
        self.dropout = original_output.dropout

    def forward(self, hidden_states, input_tensor):
        print("\n========== Output (FFN FC2) Forward Pass ==========")
        print_tensor_info(hidden_states, "Input to FFN FC2")
        print_tensor_info(input_tensor, "Input Tensor (for residual connection)")

        # 1. Linear transformation
        hidden_states = self.dense(hidden_states)
        print_tensor_info(hidden_states, "After FFN FC2 Linear")

        # 2. Dropout
        hidden_states = self.dropout(hidden_states)
        print_tensor_info(hidden_states, "After Dropout")

        # 3. Residual connection + LayerNorm
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        print_tensor_info(hidden_states, "Final Layer Output (after Residual + LayerNorm)")

        return hidden_states


class DebugBertLayer(nn.Module):
    """带调试输出的完整BERT Layer"""

    def __init__(self, original_layer):
        super().__init__()
        self.attention = nn.Module()
        self.attention.self = DebugBertSelfAttention(original_layer.attention.self)
        self.attention.output = DebugBertSelfOutput(original_layer.attention.output)
        self.intermediate = DebugBertIntermediate(original_layer.intermediate)
        self.output = DebugBertOutput(original_layer.output)

    def forward(self, hidden_states, attention_mask=None, head_mask=None, output_attentions=False):
        print(f"\n{'=' * 60}")
        print("BERT LAYER FORWARD PASS START")
        print(f"{'=' * 60}")

        # Self-attention
        self_attention_outputs = self.attention.self(
            hidden_states, attention_mask, head_mask, output_attentions
        )
        attention_output = self_attention_outputs[0]

        # Self-attention output (with residual + layernorm)
        attention_output = self.attention.output(attention_output, hidden_states)

        # Feed-forward network
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)

        print(f"\n{'=' * 60}")
        print("BERT LAYER FORWARD PASS END")
        print(f"{'=' * 60}")

        outputs = (layer_output,) + self_attention_outputs[1:]
        return outputs


def test_bert_layer_with_debug():
    """测试带调试输出的BERT layer"""
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

    print_tensor_info(input_embeddings, "Input Embeddings")

    # 创建调试版本的第6层 (index 5)
    original_layer = original_model.encoder.layer[5]
    with torch.no_grad():
        for name, param in original_layer.named_parameters():
            # 检查参数名称是否包含 'bias'
            if 'bias' in name:
                print(f"正在将参数 {name} 的值设置为 0。")
                # 使用 .zero_() 方法将 Tensor 的所有元素原地设置为 0
                param.zero_()
    query_bias_name = f'attention.self.query.bias'
    query_bias_tensor = original_layer.state_dict()[query_bias_name]

    print(f"\n修改后的 {query_bias_name} 矩阵:")
    print(query_bias_tensor)
    debug_layer = DebugBertLayer(original_layer)

    print("\n--- Running Debug BERT Layer ---")
    with torch.no_grad():
        debug_output = debug_layer(input_embeddings)[0]

    print_tensor_info(debug_output, "Final Debug Layer Output")

    # 同时运行原始layer作为对比
    print("\n--- Running Original BERT Layer for Comparison ---")
    with torch.no_grad():
        original_output = original_layer(input_embeddings)[0]

    print_tensor_info(original_output, "Final Original Layer Output")

    # 验证输出是否相同
    print(f"\nOutputs match: {torch.allclose(debug_output, original_output, rtol=1e-5, atol=1e-5)}")

    return debug_output, original_output, input_embeddings


def extract_weights_from_bert_layer(model, layer_idx=5):
    """从BERT模型中提取权重用于souffle对比"""
    layer = model.encoder.layer[layer_idx]

    print(f"\n========== BERT Layer {layer_idx} Weight Information ==========")

    # 提取QKV权重
    query_weight = layer.attention.self.query.weight.data
    key_weight = layer.attention.self.key.weight.data
    value_weight = layer.attention.self.value.weight.data
    qkv_weight = torch.stack([query_weight, key_weight, value_weight], dim=0)

    # 提取其他权重
    attn_output_weight = layer.attention.output.dense.weight.data
    ffn_fc1_weight = layer.intermediate.dense.weight.data
    ffn_fc2_weight = layer.output.dense.weight.data

    print_tensor_info(qkv_weight, "QKV Weight (stacked)")
    print_tensor_info(attn_output_weight, "Attention Output Weight")
    print_tensor_info(ffn_fc1_weight, "FFN FC1 Weight")
    print_tensor_info(ffn_fc2_weight, "FFN FC2 Weight")

    return {
        'qkv_weight': qkv_weight,
        'attn_output_weight': attn_output_weight,
        'ffn_fc1_weight': ffn_fc1_weight,
        'ffn_fc2_weight': ffn_fc2_weight
    }


if __name__ == "__main__":
    debug_output, original_output, input_embeddings = test_bert_layer_with_debug()

    # 加载原始模型用于权重提取
    model_name = "bert-base-uncased"
    model = BertModel.from_pretrained(model_name, torch_dtype=torch.float16)
    if torch.cuda.is_available():
        model = model.cuda()

    weights = extract_weights_from_bert_layer(model, layer_idx=5)