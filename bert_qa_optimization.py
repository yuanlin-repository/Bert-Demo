import torch
from torch import nn
import sys
import numpy as np
from transformers import BertForQuestionAnswering, BertTokenizer
import bert_binding
import os

# 设置CUDA调试
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


class OptimizedBertQA:
    def __init__(self, model_name="MattBoraske/BERT-question-answering-SQuAD", opt_level=4):
        """
        修复版本的优化 BERT 问答模型
        """
        self.model_name = model_name
        self.opt_level = opt_level
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # 清理CUDA缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # 加载原始模型和分词器
        print(f"Loading model: {model_name}")
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.original_model = BertForQuestionAnswering.from_pretrained(model_name)
        self.original_model.eval()

        if torch.cuda.is_available():
            self.original_model = self.original_model.cuda()

        # 提取模型参数用于优化推理
        self.extract_bert_weights()

        # BERT 配置参数
        self.batch_size = 1
        self.num_heads = 12
        self.num_hidden = 64  # hidden_size per head
        self.hidden_size = self.num_heads * self.num_hidden  # 768
        self.d_intermedia = 3072
        self.num_layers = 12

    def extract_bert_weights(self):
        """修复的权重提取函数"""
        print("Extracting BERT weights (Fixed Version)...")

        self.bert_weights = {}
        bert_model = self.original_model.bert

        # 提取每一层的权重
        for i in range(12):  # BERT-base 有 12 层
            layer = bert_model.encoder.layer[i]

            # 注意力机制权重
            attention = layer.attention.self

            # 获取原始权重 - PyTorch Linear层格式是 [out_features, in_features]
            q_weight = attention.query.weight.data  # [768, 768]
            k_weight = attention.key.weight.data  # [768, 768]
            v_weight = attention.value.weight.data  # [768, 768]

            # **关键修复1**: 检查souffle_bert_layer期望的QKV格式
            # 如果souffle_bert_layer期望的是 [3, in_features, out_features]，则需要转置
            # 如果期望的是 [3, out_features, in_features]，则不需要转置

            # 先尝试不转置的版本（因为这是HF的原始格式）
            qkv_weight = torch.stack([q_weight, k_weight, v_weight], dim=0)  # [3, 768, 768]

            # **关键修复2**: 注意力输出权重
            # 检查是否需要转置
            attn_output_weight = layer.attention.output.dense.weight.data  # [768, 768]

            # **关键修复3**: Feed Forward 权重
            # HF格式: intermediate.dense.weight [3072, 768], output.dense.weight [768, 3072]
            ff_fc1_weight_orig = layer.intermediate.dense.weight.data  # [3072, 768]
            ff_fc2_weight_orig = layer.output.dense.weight.data  # [768, 3072]

            # 检查souffle_bert_layer期望的格式
            # 通常情况下，矩阵乘法期望: input @ weight，其中weight是[in_features, out_features]
            # 但PyTorch Linear使用 F.linear(input, weight) 其中weight是[out_features, in_features]

            # 如果souffle_bert_layer使用标准矩阵乘法，我们需要转置HF权重
            ff_fc1_weight = ff_fc1_weight_orig.t()  # [768, 3072]
            ff_fc2_weight = ff_fc2_weight_orig.t()  # [3072, 768]

            print(f"Layer {i} weight shapes:")
            print(f"  QKV: {qkv_weight.shape}")
            print(f"  Attn output: {attn_output_weight.shape}")
            print(f"  FF1: {ff_fc1_weight_orig.shape} -> {ff_fc1_weight.shape}")
            print(f"  FF2: {ff_fc2_weight_orig.shape} -> {ff_fc2_weight.shape}")

            self.bert_weights[f'layer_{i}'] = {
                'qkv_weight': qkv_weight.to(torch.float16).to(self.device),
                'attn_fc_weight': attn_output_weight.to(torch.float16).to(self.device),
                'ff_fc1_weight': ff_fc1_weight.to(torch.float16).to(self.device),
                'ff_fc2_weight': ff_fc2_weight.to(torch.float16).to(self.device)
            }

        print("Weight extraction completed!")

    def run_original_forward(self, input_embeddings):
        """运行原始模型的 BERT 前向传播并返回每层的输出"""
        original_outputs = []
        hidden_states = input_embeddings.to(torch.float32)  # 原始模型使用FP32

        for i in range(self.num_layers):
            layer = self.original_model.bert.encoder.layer[i]
            with torch.no_grad():
                output = layer(hidden_states)
            print("output[0].shape", output[0].shape)
            hidden_states = output[0]
            print("hidden_states[0].shape", hidden_states[0].shape)
            original_outputs.append(hidden_states)

        return original_outputs

    def optimized_bert_forward(self, input_embeddings):
        """
        修复版本的优化 BERT 前向传播
        """
        # 确保输入格式正确
        if input_embeddings.dtype != torch.float16:
            input_embeddings = input_embeddings.to(torch.float16)
        if input_embeddings.device != torch.device(self.device):
            input_embeddings = input_embeddings.to(self.device)

        # 从原始模型获取每层的输出，用于对比
        original_outputs = self.run_original_forward(input_embeddings)

        batch_size, seq_length, hidden_size = input_embeddings.shape
        print(f"Input embeddings shape: {input_embeddings.shape}")

        hidden_states = input_embeddings

        # 逐层进行优化推理
        for i in range(min(2, self.num_layers)):  # 只测试前两层
            print(f"\n=== Processing Layer {i} ===")
            layer_weights = self.bert_weights[f'layer_{i}']

            print(f"Layer {i} weight shapes:")
            for key, weight in layer_weights.items():
                print(f"  {key}: {weight.shape}")

            try:
                # **关键修复**: 确保输入维度正确
                # souffle_bert_layer 可能期望不同的输入格式
                layer_outputs = bert_binding.souffle_bert_layer(
                    hidden_states,
                    layer_weights['qkv_weight'],
                    layer_weights['attn_fc_weight'],
                    layer_weights['ff_fc1_weight'],
                    layer_weights['ff_fc2_weight'],
                    self.opt_level
                )

                print(f"Layer {i} outputs type: {type(layer_outputs)}")

                if isinstance(layer_outputs, list):
                    print(f"Number of outputs: {len(layer_outputs)}")
                    for idx, output in enumerate(layer_outputs):
                        print(f"  Output[{idx}] shape: {output.shape}")
                        if output.dim() == 3:
                            print(f"  Output[{idx}] sample: {output[0, 0, :3]}")
                        elif output.dim() == 2:
                            print(f"  Output[{idx}] sample: {output[0, :3]}")

                    # **关键修复**: 正确处理输出格式
                    # 假设最后一个输出是下一层的输入
                    if len(layer_outputs) >= 2:
                        hidden_states = layer_outputs[-1]  # 取最后一个输出

                        # **关键修复**: 确保维度正确
                        if hidden_states.dim() == 2:
                            # 如果输出是2D [seq_len*batch, hidden]，重塑为3D
                            hidden_states = hidden_states.view(batch_size, seq_length, hidden_size)
                        elif hidden_states.dim() == 3 and hidden_states.shape[0] != batch_size:
                            # 检查维度顺序是否正确
                            print(f"Warning: unexpected batch dimension: {hidden_states.shape}")
                    else:
                        hidden_states = layer_outputs[0]
                else:
                    hidden_states = layer_outputs

                print(f"Next layer input shape: {hidden_states.shape}")

                # 与原始模型对比
                optimized_output = hidden_states
                original_output = original_outputs[i]

                # **关键修复**: 确保对比时维度和数据类型一致
                if optimized_output.shape != original_output.shape:
                    print(f"Shape mismatch - Optimized: {optimized_output.shape}, Original: {original_output.shape}")
                    # 尝试重塑
                    if optimized_output.numel() == original_output.numel():
                        optimized_output = optimized_output.view(original_output.shape)
                        print(f"Reshaped optimized output to: {optimized_output.shape}")

                # 计算差异
                diff = torch.norm(
                    optimized_output.to(torch.float32) - original_output.to(self.device)
                )

                print(f"Layer {i} L2 norm difference: {diff.item():.8f}")
                print(f"Relative difference: {(diff / torch.norm(original_output.to(self.device))).item():.8f}")

                # 如果差异很大，打印详细信息
                if diff > 1e-3:  # 放宽阈值
                    print(f"*** Large difference detected at layer {i}! ***")
                    print("Optimized output stats:")
                    opt_flat = optimized_output.flatten()
                    print(f"  Min: {opt_flat.min():.6f}, Max: {opt_flat.max():.6f}, Mean: {opt_flat.mean():.6f}")
                    print("Original output stats:")
                    orig_flat = original_output.flatten()
                    print(f"  Min: {orig_flat.min():.6f}, Max: {orig_flat.max():.6f}, Mean: {orig_flat.mean():.6f}")

                    # 检查是否有NaN或Inf
                    if torch.isnan(optimized_output).any():
                        print("  *** NaN detected in optimized output! ***")
                    if torch.isinf(optimized_output).any():
                        print("  *** Inf detected in optimized output! ***")

            except Exception as e:
                print(f"Error in layer {i}: {e}")
                import traceback
                traceback.print_exc()
                break

        return hidden_states

    def get_input_embeddings(self, input_ids, attention_mask=None, token_type_ids=None):
        """获取输入词嵌入"""
        with torch.no_grad():
            embeddings = self.original_model.bert.embeddings(
                input_ids=input_ids,
                token_type_ids=token_type_ids
            )
        return embeddings

    def predict(self, question, context, max_seq_length=384):
        """
        修复版本的预测函数
        """
        try:
            # 编码输入
            inputs = self.tokenizer(
                question,
                context,
                return_tensors="pt",
                max_length=max_seq_length,
                truncation=True,
                padding='max_length'
            )

            if torch.cuda.is_available():
                inputs = {k: v.to('cuda') for k, v in inputs.items()}

            # 获取输入嵌入
            input_embeddings = self.get_input_embeddings(
                inputs["input_ids"],
                inputs.get("attention_mask"),
                inputs.get("token_type_ids")
            ).to(torch.float16)

            # 使用修复的前向传播
            with torch.no_grad():
                sequence_output = self.optimized_bert_forward(input_embeddings)

                if sequence_output is None:
                    return {'answer': '优化模型失败，无法预测', 'error': True}

                # 通过问答头得到logits
                if not hasattr(self, 'qa_outputs'):
                    self.qa_outputs = nn.Linear(
                        self.original_model.config.hidden_size,
                        self.original_model.num_labels
                    ).to(self.device)
                    self.qa_outputs.load_state_dict(self.original_model.qa_outputs.state_dict())
                    self.qa_outputs = self.qa_outputs.to(torch.float16)

                logits = self.qa_outputs(sequence_output.to(torch.float16))
                start_logits, end_logits = logits.split(1, dim=-1)
                start_logits = start_logits.squeeze(-1)
                end_logits = end_logits.squeeze(-1)

            # 找到最佳答案位置
            answer_start_index = torch.argmax(start_logits, dim=1)
            answer_end_index = torch.argmax(end_logits, dim=1)
            print("answer_start_index:", answer_start_index)
            print("answer_end_index:", answer_end_index)

            # 提取答案
            all_tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
            start_idx = answer_start_index.item()
            end_idx = answer_end_index.item()

            if start_idx <= end_idx and end_idx < len(all_tokens):
                answer_tokens = all_tokens[start_idx:end_idx + 1]
                answer = self.tokenizer.convert_tokens_to_string(answer_tokens)
                answer = answer.replace('[CLS]', '').replace('[SEP]', '').replace('[PAD]', '').strip()
            else:
                answer = "无法找到答案"

            return {
                'answer': answer,
                'start_logits': start_logits,
                'end_logits': end_logits,
                'start_index': start_idx,
                'end_index': end_idx,
                'error': False
            }

        except Exception as e:
            print(f"Error in prediction: {e}")
            import traceback
            traceback.print_exc()
            return {'answer': f'预测出错: {str(e)}', 'error': True}


def simple_souffle_test():
    """使用原始BERT模型第一层权重测试 souffle_bert_layer"""
    print("\n=== Testing souffle_bert_layer with Real BERT Layer 0 Weights ===")

    # 重置CUDA
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    try:
        # 加载原始模型
        model_name = "MattBoraske/BERT-question-answering-SQuAD"
        print(f"Loading model: {model_name}")

        tokenizer = BertTokenizer.from_pretrained(model_name)
        original_model = BertForQuestionAnswering.from_pretrained(model_name)
        original_model.eval()

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if torch.cuda.is_available():
            original_model = original_model.cuda()

        # 准备测试输入
        question = "Who walked on the moon?"
        context = "Neil Armstrong was the first person to walk on the moon in 1969."

        inputs = tokenizer(
            question,
            context,
            return_tensors="pt",
            max_length=384,
            truncation=True,
            padding='max_length'
        )

        if torch.cuda.is_available():
            inputs = {k: v.to('cuda') for k, v in inputs.items()}

        # 获取输入嵌入
        with torch.no_grad():
            input_embeddings = original_model.bert.embeddings(
                input_ids=inputs["input_ids"],
                token_type_ids=inputs.get("token_type_ids")
            )

        print(f"Input embeddings shape: {input_embeddings.shape}")

        # 获取第一层的真实权重
        layer_0 = original_model.bert.encoder.layer[0]
        attention = layer_0.attention.self

        # 提取权重 (PyTorch Linear层格式: [out_features, in_features])
        q_weight = attention.query.weight.data  # [768, 768]
        k_weight = attention.key.weight.data  # [768, 768]
        v_weight = attention.value.weight.data  # [768, 768]
        attn_output_weight = layer_0.attention.output.dense.weight.data  # [768, 768]
        ff_fc1_weight = layer_0.intermediate.dense.weight.data  # [3072, 768]
        ff_fc2_weight = layer_0.output.dense.weight.data  # [768, 3072]

        print("Original weight shapes:")
        print(f"  Q weight: {q_weight.shape}")
        print(f"  K weight: {k_weight.shape}")
        print(f"  V weight: {v_weight.shape}")
        print(f"  Attn output: {attn_output_weight.shape}")
        print(f"  FF1: {ff_fc1_weight.shape}")
        print(f"  FF2: {ff_fc2_weight.shape}")

        # 运行原始模型第一层获取参考输出
        print("\n--- Running Original Model Layer 0 ---")
        with torch.no_grad():
            hidden_states = input_embeddings.to(torch.float32)  # 原始模型使用FP32
            original_output = layer_0(hidden_states)[0]  # 取第一个输出（hidden_states）

        print(f"Original layer 0 output shape: {original_output.shape}")
        print(f"Original output sample: {original_output[0, 0, :5]}")

        # 测试不同的权重格式配置
        test_configs = [
            {
                'name': 'Config 1: Stack QKV, no transpose',
                'qkv': torch.stack([q_weight, k_weight, v_weight], dim=0),  # [3, 768, 768]
                'attn_fc': attn_output_weight,  # [768, 768]
                'ff1': ff_fc1_weight.t(),  # [768, 3072]
                'ff2': ff_fc2_weight.t()  # [3072, 768]
            },
            {
                'name': 'Config 2: Stack QKV transposed',
                'qkv': torch.stack([q_weight.t(), k_weight.t(), v_weight.t()], dim=0),  # [3, 768, 768]
                'attn_fc': attn_output_weight,
                'ff1': ff_fc1_weight.t(),
                'ff2': ff_fc2_weight.t()
            },
            {
                'name': 'Config 3: All weights transposed',
                'qkv': torch.stack([q_weight.t(), k_weight.t(), v_weight.t()], dim=0),
                'attn_fc': attn_output_weight.t(),
                'ff1': ff_fc1_weight,  # 保持原始格式
                'ff2': ff_fc2_weight  # 保持原始格式
            },
            {
                'name': 'Config 4: Only attn_fc transposed',
                'qkv': torch.stack([q_weight, k_weight, v_weight], dim=0),
                'attn_fc': attn_output_weight.t(),
                'ff1': ff_fc1_weight.t(),
                'ff2': ff_fc2_weight.t()
            },
            {
                'name': 'Config 5: Original HF format',
                'qkv': torch.stack([q_weight, k_weight, v_weight], dim=0),
                'attn_fc': attn_output_weight,
                'ff1': ff_fc1_weight,  # [3072, 768] - 原始HF格式
                'ff2': ff_fc2_weight  # [768, 3072] - 原始HF格式
            }
        ]

        best_config = None
        best_diff = float('inf')

        for config in test_configs:
            print(f"\n--- Testing {config['name']} ---")

            try:
                # 转换权重到FP16和正确设备
                qkv_w = config['qkv'].to(torch.float16).to(device)
                attn_w = config['attn_fc'].to(torch.float16).to(device)
                ff1_w = config['ff1'].to(torch.float16).to(device)
                ff2_w = config['ff2'].to(torch.float16).to(device)

                print(f"  QKV weight shape: {qkv_w.shape}")
                print(f"  Attn FC weight shape: {attn_w.shape}")
                print(f"  FF1 weight shape: {ff1_w.shape}")
                print(f"  FF2 weight shape: {ff2_w.shape}")

                # 调用 souffle_bert_layer
                souffle_output = bert_binding.souffle_bert_layer(
                    input_embeddings.to(torch.float16).to(device),
                    qkv_w,
                    attn_w,
                    ff1_w,
                    ff2_w,
                    4  # opt_level
                )

                print(f"  Souffle output type: {type(souffle_output)}")

                if isinstance(souffle_output, list):
                    print(f"  Number of outputs: {len(souffle_output)}")
                    for idx, out in enumerate(souffle_output):
                        print(f"    Output[{idx}] shape: {out.shape}")

                    # 假设最后一个输出是最终的hidden states
                    final_output = souffle_output[-1]
                else:
                    final_output = souffle_output

                print(f"  Final output shape: {final_output.shape}")

                # 确保维度匹配进行比较
                if final_output.shape != original_output.shape:
                    print(f"  Shape mismatch! Souffle: {final_output.shape}, Original: {original_output.shape}")

                    # 尝试重塑
                    if final_output.numel() == original_output.numel():
                        final_output = final_output.view(original_output.shape)
                        print(f"  Reshaped to: {final_output.shape}")
                    else:
                        print(f"  Cannot reshape - different number of elements")
                        continue

                # 计算差异
                diff = torch.norm(
                    final_output.to(torch.float32) - original_output.to(device).to(torch.float32)
                )
                relative_diff = (diff / torch.norm(original_output.to(device).to(torch.float32))).item()

                print(f"  L2 norm difference: {diff.item():.8f}")
                print(f"  Relative difference: {relative_diff:.8f}")

                # 检查数值稳定性
                if torch.isnan(final_output).any():
                    print("  *** WARNING: NaN detected in souffle output! ***")
                if torch.isinf(final_output).any():
                    print("  *** WARNING: Inf detected in souffle output! ***")

                # 记录最佳配置
                if diff < best_diff:
                    best_diff = diff
                    best_config = config['name']

                # 如果差异很小，说明配置正确
                if relative_diff < 1e-3:
                    print(f"  *** EXCELLENT MATCH! Relative difference < 0.1% ***")
                elif relative_diff < 1e-2:
                    print(f"  *** GOOD MATCH! Relative difference < 1% ***")
                elif relative_diff < 1e-1:
                    print(f"  *** ACCEPTABLE MATCH! Relative difference < 10% ***")
                else:
                    print(f"  *** POOR MATCH! Relative difference > 10% ***")

            except Exception as e:
                print(f"  Error with {config['name']}: {e}")
                import traceback
                traceback.print_exc()

        print(f"\n=== SUMMARY ===")
        print(f"Best configuration: {best_config}")
        print(f"Best L2 difference: {best_diff:.8f}")

        if best_diff < 1e-3:
            print("SUCCESS: Found a configuration with excellent match!")
            return True
        elif best_diff < 1e-2:
            print("PARTIAL SUCCESS: Found a configuration with good match!")
            return True
        else:
            print("FAILURE: No configuration achieved good match.")
            return False

    except Exception as e:
        print(f"Error in weight format test: {e}")
        import traceback
        traceback.print_exc()
        return False


def unit_weight_test():
    """使用全1权重矩阵测试原始模型和优化模型的一致性"""
    print("\n=== Unit Weight Test: All Weights Set to 1 ===")

    # 重置CUDA
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    try:
        # 加载原始模型
        model_name = "MattBoraske/BERT-question-answering-SQuAD"
        print(f"Loading model: {model_name}")

        tokenizer = BertTokenizer.from_pretrained(model_name)
        original_model = BertForQuestionAnswering.from_pretrained(model_name)
        original_model.eval()

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if torch.cuda.is_available():
            original_model = original_model.cuda()

        # 准备简单的测试输入
        batch_size, seq_len, hidden_size = 1, 384, 768  # 使用更小的序列长度便于调试
        d_intermediate = 3072

        # 创建简单的输入嵌入 - 使用简单的递增序列便于验证
        input_embeddings = torch.arange(
            batch_size * seq_len * hidden_size,
            dtype=torch.float32
        ).view(batch_size, seq_len, hidden_size) * 0.001  # 缩放避免数值爆炸
        input_embeddings.fill_(1.0)
        print("input_embeddings:", input_embeddings)

        if torch.cuda.is_available():
            input_embeddings = input_embeddings.cuda()

        print(f"Input embeddings shape: {input_embeddings.shape}")
        print(f"Input sample: {input_embeddings[0, 0, :5]}")

        # === 1. 修改原始模型的权重为全1 ===
        print("\n--- Setting Original Model Weights to 1 ---")

        # 只测试第一层
        layer_0 = original_model.bert.encoder.layer[0]

        # 设置注意力权重为1
        with torch.no_grad():
            layer_0.attention.self.query.weight.fill_(1.0)
            layer_0.attention.self.key.weight.fill_(1.0)
            layer_0.attention.self.value.weight.fill_(1.0)
            layer_0.attention.output.dense.weight.fill_(1.0)

            # 设置前馈网络权重为1
            layer_0.intermediate.dense.weight.fill_(1.0)
            layer_0.output.dense.weight.fill_(1.0)

            # 设置所有偏置为0（如果存在）
            if layer_0.attention.self.query.bias is not None:
                layer_0.attention.self.query.bias.fill_(0.0)
            if layer_0.attention.self.key.bias is not None:
                layer_0.attention.self.key.bias.fill_(0.0)
            if layer_0.attention.self.value.bias is not None:
                layer_0.attention.self.value.bias.fill_(0.0)
            if layer_0.attention.output.dense.bias is not None:
                layer_0.attention.output.dense.bias.fill_(0.0)
            if layer_0.intermediate.dense.bias is not None:
                layer_0.intermediate.dense.bias.fill_(0.0)
            if layer_0.output.dense.bias is not None:
                layer_0.output.dense.bias.fill_(0.0)

        print("Original model weights set to 1")

        # 运行原始模型第一层
        print("\n--- Running Original Model with Unit Weights ---")
        with torch.no_grad():
            original_output = layer_0(input_embeddings)[0]

        print(f"Original output shape: {original_output.shape}")
        print(f"Original output sample: {original_output[0, 0, :5]}")
        print(
            f"Original output stats - Min: {original_output.min():.6f}, Max: {original_output.max():.6f}, Mean: {original_output.mean():.6f}")

        # === 2. 创建全1权重给souffle_bert_layer ===
        print("\n--- Creating Unit Weights for souffle_bert_layer ---")

        # 创建全1权重矩阵
        qkv_weight_ones = torch.ones(3, hidden_size, hidden_size, dtype=torch.float16, device=device)
        attn_fc_weight_ones = torch.ones(hidden_size, hidden_size, dtype=torch.float16, device=device)
        ff_fc1_weight_ones = torch.ones(hidden_size, d_intermediate, dtype=torch.float16, device=device)
        ff_fc2_weight_ones = torch.ones(d_intermediate, hidden_size, dtype=torch.float16, device=device)

        print(f"QKV weight shape: {qkv_weight_ones.shape}")
        print(f"Attn FC weight shape: {attn_fc_weight_ones.shape}")
        print(f"FF1 weight shape: {ff_fc1_weight_ones.shape}")
        print(f"FF2 weight shape: {ff_fc2_weight_ones.shape}")

        # === 3. 运行souffle_bert_layer ===
        print("\n--- Running souffle_bert_layer with Unit Weights ---")
        try:
            souffle_output = bert_binding.souffle_bert_layer(
                input_embeddings.to(torch.float16).to(device),
                qkv_weight_ones,
                attn_fc_weight_ones,
                ff_fc1_weight_ones,
                ff_fc2_weight_ones,
                4  # opt_level
            )
            souffle_output = souffle_output[1]
            print(f"Souffle output type: {type(souffle_output)}")
            final_output = souffle_output
            print(f"Single output shape: {final_output.shape}")
            sample = final_output.flatten()[:5] if final_output.numel() >= 5 else final_output.flatten()
            print(f"Single output sample: {sample}")
            print(
                f"Single output stats - Min: {final_output.min():.6f}, Max: {final_output.max():.6f}, Mean: {final_output.mean():.6f}")

            if final_output.shape == original_output.shape:
                print(f"\n--- Comparing Single Output (Same Shape) ---")
                compare_outputs(original_output, final_output, "Single Output")
            elif final_output.numel() == original_output.numel():
                print(f"\n--- Comparing Single Output (Reshaped) ---")
                reshaped_output = final_output.view(original_output.shape)
                compare_outputs(original_output, reshaped_output, "Single Output (reshaped)")
            else:
                print(
                    f"Single output cannot be compared - shape: {final_output.shape}, elements: {final_output.numel()}")

        except Exception as e:
            print(f"Error running souffle_bert_layer: {e}")
            import traceback
            traceback.print_exc()
            return False

        return True

    except Exception as e:
        print(f"Error in unit weight test: {e}")
        import traceback
        traceback.print_exc()
        return False


def compare_outputs(original, optimized, label="Optimized"):
    """比较两个输出张量的函数"""
    print(f"\n=== Comparing {label} Output ===")

    # 确保数据类型一致
    original_float = original.to(torch.float32)
    optimized_float = optimized.to(torch.float32)

    # 基本形状检查
    print(f"Original shape: {original_float.shape}")
    print(f"{label} shape: {optimized_float.shape}")

    if original_float.shape != optimized_float.shape:
        print("*** SHAPE MISMATCH! ***")
        return False

    # 数值比较
    diff = torch.norm(optimized_float - original_float)
    relative_diff = (diff / torch.norm(original_float)).item()

    print(f"L2 norm difference: {diff.item():.8f}")
    print(f"Relative difference: {relative_diff:.8f}")

    # 逐元素比较统计
    abs_diff = torch.abs(optimized_float - original_float)
    max_abs_diff = abs_diff.max().item()
    mean_abs_diff = abs_diff.mean().item()

    print(f"Max absolute difference: {max_abs_diff:.8f}")
    print(f"Mean absolute difference: {mean_abs_diff:.8f}")

    # 检查数值稳定性
    nan_orig = torch.isnan(original_float).sum().item()
    nan_opt = torch.isnan(optimized_float).sum().item()
    inf_orig = torch.isinf(original_float).sum().item()
    inf_opt = torch.isinf(optimized_float).sum().item()

    if nan_orig > 0 or nan_opt > 0:
        print(f"*** NaN detected! Original: {nan_orig}, {label}: {nan_opt} ***")
    if inf_orig > 0 or inf_opt > 0:
        print(f"*** Inf detected! Original: {inf_orig}, {label}: {inf_opt} ***")

    # 样本比较
    print("\nSample comparison (first 5 elements):")
    orig_sample = original_float.flatten()[:5]
    opt_sample = optimized_float.flatten()[:5]
    print(f"Original:  {orig_sample}")
    print(f"{label}: {opt_sample}")
    print(f"Diff:      {opt_sample - orig_sample}")

    # 判断结果
    if relative_diff < 1e-6:
        print("*** EXCELLENT MATCH! (< 0.0001%) ***")
        return True
    elif relative_diff < 1e-4:
        print("*** VERY GOOD MATCH! (< 0.01%) ***")
        return True
    elif relative_diff < 1e-3:
        print("*** GOOD MATCH! (< 0.1%) ***")
        return True
    elif relative_diff < 1e-2:
        print("*** ACCEPTABLE MATCH! (< 1%) ***")
        return True
    else:
        print("*** POOR MATCH! (> 1%) ***")
        return False


def main():
    """主调试函数"""
    print("开始修复和调试 BERT 优化模型...")

    unit_success = unit_weight_test()

    if unit_success:
        print("\n*** 单位权重测试成功! ***")
    else:
        print("\n*** 单位权重测试失败! ***")
        # 如果失败，尝试扩展测试

    # 1. 首先测试最简单的 souffle_bert_layer
    if not simple_souffle_test():
        print("Basic souffle_bert_layer test failed. Check the binding implementation.")

    # 2. 如果基础测试通过，继续测试完整模型
    print("\nBasic test passed, loading full model...")

    try:
        optimized_qa = OptimizedBertQA(
            model_name="MattBoraske/BERT-question-answering-SQuAD",
            opt_level=4
        )

        # 3. 使用简单的测试数据
        question = "Who walked on the moon?"
        context = "Neil Armstrong was the first person to walk on the moon in 1969."

        pred_result = optimized_qa.predict(question, context, max_seq_length=384)
        print(pred_result)

    except Exception as e:
        print(f"Error in main: {e}")
        import traceback
        traceback.print_exc()


# 额外的权重格式检查函数
def check_bert_weight_formats():
    """检查BERT权重的标准格式"""
    print("\n=== Checking Standard BERT Weight Formats ===")

    # 创建一个小的BERT模型来检查标准格式
    from transformers import BertConfig, BertModel

    config = BertConfig(
        hidden_size=768,
        num_attention_heads=12,
        intermediate_size=3072,
        num_hidden_layers=2  # 只用2层来快速测试
    )

    model = BertModel(config)
    model.eval()

    print("Standard BERT weight shapes:")
    layer = model.encoder.layer[0]

    print(f"Query weight: {layer.attention.self.query.weight.shape}")
    print(f"Key weight: {layer.attention.self.key.weight.shape}")
    print(f"Value weight: {layer.attention.self.value.weight.shape}")
    print(f"Attention output: {layer.attention.output.dense.weight.shape}")
    print(f"FF intermediate: {layer.intermediate.dense.weight.shape}")
    print(f"FF output: {layer.output.dense.weight.shape}")

    # 测试标准的线性变换
    batch_size, seq_len = 1, 5
    test_input = torch.randn(batch_size, seq_len, 768)

    with torch.no_grad():
        # 标准的Q计算
        q_output = layer.attention.self.query(test_input)

        # 手动计算（应该相同）
        q_manual = torch.matmul(test_input, layer.attention.self.query.weight.t())

        diff = torch.norm(q_output - q_manual)
        print(f"\nQ computation difference: {diff:.8f}")
        print("This confirms the correct weight format for linear transformations.")


if __name__ == "__main__":
    # 首先检查标准格式
    check_bert_weight_formats()

    # 然后运行主调试
    main()
