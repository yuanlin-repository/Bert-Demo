import torch
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

            # QKV 权重合并
            q_weight = attention.query.weight.data
            k_weight = attention.key.weight.data
            v_weight = attention.value.weight.data
            qkv_weight = torch.stack([q_weight, k_weight, v_weight], dim=0)

            # 注意力输出权重
            attn_output_weight = layer.attention.output.dense.weight.data

            # Feed Forward 权重
            ff_fc1_weight = layer.intermediate.dense.weight.data.t()  # 转置以匹配期望格式
            ff_fc2_weight = layer.output.dense.weight.data.t()

            self.bert_weights[f'layer_{i}'] = {
                'qkv_weight': qkv_weight.to(torch.float16).to(self.device),
                'attn_fc_weight': attn_output_weight.to(torch.float16).to(self.device),
                'ff_fc1_weight': ff_fc1_weight.to(torch.float16).to(self.device),
                'ff_fc2_weight': ff_fc2_weight.to(torch.float16).to(self.device)
            }

        # 提取问答头权重
        self.qa_outputs_weight = self.original_model.qa_outputs.weight.data
        self.qa_outputs_bias = self.original_model.qa_outputs.bias.data

        print("Weight extraction completed!")

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

        hidden_states = input_embeddings

        # 逐层进行优化推理
        for i in range(self.num_layers):
            layer_weights = self.bert_weights[f'layer_{i}']

            # 使用 bert_binding 进行单层优化推理
            hidden_states = bert_binding.souffle_bert_layer(
                hidden_states,
                layer_weights['qkv_weight'],
                layer_weights['attn_fc_weight'],
                layer_weights['ff_fc1_weight'],
                layer_weights['ff_fc2_weight'],
                self.opt_level
            )

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
        )

        # 使用优化的 BERT 进行前向传播
        with torch.no_grad():
            sequence_output = self.optimized_bert_forward(input_embeddings)

            # 通过问答头得到开始和结束位置的logits
            logits = torch.matmul(sequence_output, self.qa_outputs_weight.t()) + self.qa_outputs_bias
            start_logits, end_logits = logits.split(1, dim=-1)
            start_logits = start_logits.squeeze(-1)
            end_logits = end_logits.squeeze(-1)

        # 找到最佳答案位置
        answer_start_index = torch.argmax(start_logits, dim=1)
        answer_end_index = torch.argmax(end_logits, dim=1)

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
    question = "What is Machine Learning?"
    context = """ Machine learning (ML) is the scientific study of algorithms and statistical models that computer systems use to progressively improve their performance 
                on a specific task. Machine learning algorithms build a mathematical model of sample data, known as "training data", in order to make predictions or 
                decisions without being explicitly programmed to perform the task. Machine learning algorithms are used in the applications of email filtering, detection 
                of network intruders, and computer vision, where it is infeasible to develop an algorithm of specific instructions for performing the task. Machine learning 
                is closely related to computational statistics, which focuses on making predictions using computers. The study of mathematical optimization delivers methods, 
                theory and application domains to the field of machine learning. Data mining is a field of study within machine learning, and focuses on exploratory 
                data analysis through unsupervised learning.In its application across business problems, machine learning is also referred to as predictive analytics. """

    # 进行预测
    print("\n进行问答预测...")
    result = optimized_qa.predict(question, context)

    print(f"\n问题: {question}")
    print(f"答案: {result['answer']}")
    print(f"开始位置: {result['start_index']}")
    print(f"结束位置: {result['end_index']}")

    # 性能基准测试
    print("\n开始性能基准测试...")
    benchmark_result = optimized_qa.benchmark_comparison(question, context, num_runs=5)

    print("\n优化完成!")


if __name__ == "__main__":
    main()