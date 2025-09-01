import torch
from torch import nn
import sys
import numpy as np
from transformers import BertForQuestionAnswering, BertTokenizer
import bert_binding


class OptimizedBertQA:
    def __init__(self, model_name="MattBoraske/BERT-question-answering-SQuAD", opt_level=4):
        """
        初始化优化的 BERT 问答模型

        Args:
            model_name: Hugging Face 模型名称
            opt_level: 优化级别 (通常为 1-4)
        """
        self.model_name = model_name
        self.opt_level = opt_level
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

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
        """从 Hugging Face 模型中提取 BERT 权重"""
        print("Extracting BERT weights...")

        self.bert_weights = {}
        bert_model = self.original_model.bert

        # 提取每一层的权重
        for i in range(12):  # BERT-base 有 12 层
            layer = bert_model.encoder.layer[i]

            # 注意力机制权重
            attention = layer.attention.self

            # QKV 权重合并 - 按照 bert_binding 期望的格式
            q_weight = attention.query.weight.data  # [768, 768]
            k_weight = attention.key.weight.data  # [768, 768]
            v_weight = attention.value.weight.data  # [768, 768]

            diff = torch.norm(q_weight.to(torch.float16).to(torch.float32) - q_weight)
            print("q_weight_diff:", diff)
            diff = torch.norm(k_weight.to(torch.float16).to(torch.float32) - k_weight)
            print("k_weight_diff:", diff)
            diff = torch.norm(v_weight.to(torch.float16).to(torch.float32) - v_weight)
            print("v_weight_diff:", diff)


            # 将 QKV 权重连接成 [3, 768, 768] 的形状
            qkv_weight = torch.stack([q_weight, k_weight, v_weight], dim=0)
            print("qkv_weight.shape:", qkv_weight.shape)

            # 注意力输出权重
            attn_output_weight = layer.attention.output.dense.weight.data
            print("attn_output_weight.shape:", attn_output_weight.shape)

            # Feed Forward 权重
            ff_fc1_weight = layer.intermediate.dense.weight.data.t()  # [768, 3072]
            ff_fc2_weight = layer.output.dense.weight.data.t()  # [3072, 768]
            print("ff_fc1_weight.shape:", ff_fc1_weight.shape)
            print("ff_fc2_weight.shape:", ff_fc2_weight.shape)


            self.bert_weights[f'layer_{i}'] = {
                'qkv_weight': qkv_weight.to(torch.float16).to(self.device),
                'attn_fc_weight': attn_output_weight.to(torch.float16).to(self.device),
                'ff_fc1_weight': ff_fc1_weight.to(torch.float16).to(self.device),
                'ff_fc2_weight': ff_fc2_weight.to(torch.float16).to(self.device)
            }

        # 提取问答头权重

        self.qa_outputs = nn.Linear(self.original_model.config.hidden_size, self.original_model.num_labels).to(self.device)
        # 2. 从官方模型加载权重和偏置
        self.qa_outputs.load_state_dict(self.original_model.qa_outputs.state_dict())
        self.qa_outputs.to(self.device).to(torch.float16)

        print("Weight extraction completed!")

    def run_original_forward(self, input_embeddings):
        """
        运行原始模型的 BERT 前向传播并返回每层的输出
        """
        original_outputs = []
        hidden_states = input_embeddings
        for i in range(self.num_layers):
            # 逐层调用原始模型的 forward 方法
            layer = self.original_model.bert.encoder.layer[i]

            # 禁用梯度计算
            with torch.no_grad():
                output = layer(hidden_states)

            # layer(hidden_states) 返回一个元组，第一个元素是张量
            hidden_states = output[0]
            original_outputs.append(hidden_states)

        return original_outputs

    def optimized_bert_forward(self, input_embeddings):
        """
        使用 bert_binding 进行优化的 BERT 前向传播

        Args:
            input_embeddings: 输入词嵌入 [batch_size, seq_length, hidden_size]

        Returns:
            最后一层的隐藏状态
        """
        # 确保输入格式正确
        if input_embeddings.dtype != torch.float16:
            input_embeddings = input_embeddings.to(torch.float16)
        if input_embeddings.device != torch.device(self.device):
            input_embeddings = input_embeddings.to(self.device)

        # 从原始模型获取每层的输出，用于对比
        original_outputs = self.run_original_forward(
            input_embeddings.to(torch.float32)  # 原始模型使用 FP32
        )

        # 调整输入形状以匹配 bert_binding 期望的格式
        # 从 [batch_size, seq_length, hidden_size] 调整为 [batch_size * seq_length, hidden_size]
        # batch_size, seq_length, hidden_size = input_embeddings.shape
        # hidden_states = input_embeddings.view(batch_size * seq_length, hidden_size)
        batch_size, seq_length, hidden_size = input_embeddings.shape
        print("input_embeddings.shape:", input_embeddings.shape)
        hidden_states = input_embeddings  # 保留原始三维形状

        # 逐层进行优化推理
        for i in range(self.num_layers):
            layer_weights = self.bert_weights[f'layer_{i}']

            # 使用 bert_binding 进行单层优化推理
            # layer_output 将是一个包含两个元素的列表：[attn_fc_output, feed_forward_fc2_output]
            layer_outputs = bert_binding.souffle_bert_layer(
                hidden_states,
                layer_weights['qkv_weight'],
                layer_weights['attn_fc_weight'],
                layer_weights['ff_fc1_weight'],
                layer_weights['ff_fc2_weight'],
                self.opt_level
            )

            # 检查返回值类型，并取第二个输出作为下一层的输入
            # souffle_bert_layer 的输出是 {attn_fc_output, feed_forward_fc2_output}
            # 在 BERT 中，下一层的输入是当前层前馈网络（feed_forward_fc2_output）的输出
            att = layer_outputs[0]
            print("att.shape:", att.shape)
            if isinstance(layer_outputs, list) and len(layer_outputs) == 2:
                # feed_forward_fc2_output 是列表的第二个元素 (索引为 1)
                hidden_states = layer_outputs[1]
                print("hidden_states.shape:", hidden_states.shape)
            else:
                # 如果不是预期的列表形式，这里可能需要报错或采取其他处理
                # 为了简化，这里假设它是一个单一张量（尽管根据描述不应该发生）
                print(f"Warning: Unexpected output type from souffle_bert_layer at layer {i}. "
                      f"Expected a list of two tensors, got {type(layer_outputs)}.")
                hidden_states = layer_outputs  # Fallback, likely incorrect logic if not a list

            hidden_states = hidden_states.unsqueeze(0)


            # --- 调试部分 ---
            # 获取原始模型在同一层的输出
            optimized_output = hidden_states
            original_output = original_outputs[i]

            # 计算 L2 范数差异
            diff = torch.norm(optimized_output.to(torch.float32) - original_output.to(self.device))

            print(f"--- Layer {i} Comparison ---")
            print(f"Optimized output shape: {optimized_output.shape}, dtype: {optimized_output.dtype}")
            print(f"Original output shape: {original_output.shape}, dtype: {original_output.dtype}")
            print(f"L2 norm difference: {diff.item():.6f}")

            # 如果差异过大，停止并打印详细信息
            if diff > 1e-4:
                print(f"*** Large difference detected at layer {i}. Potential bug found! ***")
                print("Optimized output sample:")
                print(hidden_states[0, 0, :10])
                print("Original output sample:")
                print(original_output[0, 0, :10])
                # 你可以在这里设置一个断点或抛出异常
                sys.exit()

        return hidden_states

    def get_input_embeddings(self, input_ids, attention_mask=None, token_type_ids=None):
        """获取输入词嵌入"""
        with torch.no_grad():
            # 使用原始模型的嵌入层
            embeddings = self.original_model.bert.embeddings(
                input_ids=input_ids,
                token_type_ids=token_type_ids
            )
        return embeddings

    def predict(self, question, context, max_seq_length=384):
        """
        使用优化模型进行问答预测

        Args:
            question: 问题文本
            context: 上下文文本
            max_seq_length: 最大序列长度

        Returns:
            预测的答案
        """
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

        # 使用优化的 BERT 进行前向传播
        with torch.no_grad():
            sequence_output = self.optimized_bert_forward(input_embeddings)

            # 通过问答头得到开始和结束位置的logits
            # logits = torch.matmul(sequence_output, self.qa_outputs_weight.t()) + self.qa_outputs_bias
            logits = self.qa_outputs(sequence_output)
            start_logits, end_logits = logits.split(1, dim=-1)
            start_logits = start_logits.squeeze(-1).view(self.batch_size, max_seq_length)
            end_logits = end_logits.squeeze(-1).view(self.batch_size, max_seq_length)

        # 找到最佳答案位置
        answer_start_index = torch.argmax(start_logits, dim=1)
        answer_end_index = torch.argmax(end_logits, dim=1)
        print("answer_index", answer_start_index, " ", answer_end_index)

        # 提取答案
        all_tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

        start_idx = answer_start_index.item()
        end_idx = answer_end_index.item()

        if start_idx <= end_idx and end_idx < len(all_tokens):
            answer_tokens = all_tokens[start_idx:end_idx + 1]
            answer = self.tokenizer.convert_tokens_to_string(answer_tokens)
            # 清理答案（移除特殊标记）
            answer = answer.replace('[CLS]', '').replace('[SEP]', '').replace('[PAD]', '').strip()
        else:
            answer = "无法找到答案"

        return {
            'answer': answer,
            'start_logits': start_logits,
            'end_logits': end_logits,
            'start_index': start_idx,
            'end_index': end_idx
        }

    def benchmark_comparison(self, question, context, num_runs=10):
        """
        比较原始模型和优化模型的性能

        Args:
            question: 测试问题
            context: 测试上下文
            num_runs: 运行次数用于计算平均时间
        """
        import time

        print("=" * 50)
        print("性能基准测试")
        print("=" * 50)

        # 预处理输入
        inputs = self.tokenizer(question, context, return_tensors="pt")
        if torch.cuda.is_available():
            inputs = {k: v.to('cuda') for k, v in inputs.items()}

        # 测试原始模型
        print("\n测试原始模型...")
        original_times = []
        with torch.no_grad():
            for i in range(num_runs):
                start_time = time.time()
                original_outputs = self.original_model(**inputs)

                # 模型输出包含答案开始和结束位置的预测
                answer_start_scores = original_outputs.start_logits
                answer_end_scores = original_outputs.end_logits

                # 找到分数最高的起始和结束位置
                answer_start_index = torch.argmax(answer_start_scores)
                answer_end_index = torch.argmax(answer_end_scores)
                print("answer_index", answer_start_index, " ", answer_end_index)

                # 获取所有标记（token）
                all_tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

                # 从标记中提取答案
                answer = self.tokenizer.convert_tokens_to_string(all_tokens[answer_start_index:answer_end_index + 1])

                print(f"问题: {question}")
                print(f"答案: {answer}")

                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                end_time = time.time()
                original_times.append(end_time - start_time)

        # 测试优化模型
        print("测试优化模型...")
        optimized_times = []
        for i in range(num_runs):
            start_time = time.time()
            _ = self.predict(question, context)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            end_time = time.time()
            optimized_times.append(end_time - start_time)

        # 计算统计信息
        orig_avg = np.mean(original_times) * 1000  # 转换为毫秒
        orig_std = np.std(original_times) * 1000
        opt_avg = np.mean(optimized_times) * 1000
        opt_std = np.std(optimized_times) * 1000

        speedup = orig_avg / opt_avg

        print(f"\n结果 (平均 {num_runs} 次运行):")
        print(f"原始模型:  {orig_avg:.2f} ± {orig_std:.2f} ms")
        print(f"优化模型:  {opt_avg:.2f} ± {opt_std:.2f} ms")
        print(f"加速比:    {speedup:.2f}x")

        return {
            'original_time': orig_avg,
            'optimized_time': opt_avg,
            'speedup': speedup
        }


def main():
    """主函数 - 演示优化模型的使用"""

    # 初始化优化模型
    print("初始化优化的 BERT 问答模型...")
    optimized_qa = OptimizedBertQA(
        model_name="MattBoraske/BERT-question-answering-SQuAD",
        opt_level=4
    )

    # 定义测试数据
    question = "Who was the first person to walk on the moon?"
    context = """ Apollo 11 was the fifth crewed mission of NASA's Apollo program, and it was the first mission to land humans on the Moon. On July 20, 1969, commander Neil Armstrong and lunar module pilot Buzz Aldrin became the first two humans to walk on the lunar surface. They landed the lunar module "Eagle" in a region on the moon called the Sea of Tranquility. After their moonwalk, the two spent about two and a half hours on the lunar surface, collecting rock samples and planting the American flag."""

    # 进行预测
    print("\n进行问答预测...")
    result = optimized_qa.predict(question, context)

    print(f"\n问题: {question}")
    print(f"答案: {result['answer']}")
    print(f"开始位置: {result['start_index']}")
    print(f"结束位置: {result['end_index']}")

    # 性能基准测试
    print("\n开始性能基准测试...")
    benchmark_result = optimized_qa.benchmark_comparison(question, context, num_runs=1)

    print("\n优化完成!")


if __name__ == "__main__":
    main()