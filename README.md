# Bert-Demo

## Reproduce steps
1. ssh yuanlin@34.44.107.72
2. docker exec -it ca023b78bfb0 /bin/bash
3. cd /workspace/souffle-models/python/models/bert
4. python bert_qa_optimization.py

**Sample output**
```shell
root@ca023b78bfb0:/workspace/souffle-models/python/models/bert# python bert_qa_optimization.py

=== Checking Standard BERT Weight Formats ===
Standard BERT weight shapes:
Query weight: torch.Size([768, 768])
Key weight: torch.Size([768, 768])
Value weight: torch.Size([768, 768])
Attention output: torch.Size([768, 768])
FF intermediate: torch.Size([3072, 768])
FF output: torch.Size([768, 3072])

Q computation difference: 0.00000000
This confirms the correct weight format for linear transformations.
开始修复和调试 BERT 优化模型...

=== Unit Weight Test: All Weights Set to 1 ===
Loading model: MattBoraske/BERT-question-answering-SQuAD
/workspace/anaconda3/lib/python3.9/site-packages/huggingface_hub/file_download.py:945: FutureWarning: `resume_download` is deprecated and will be removed in version 1.0.0. Downloads always resume when possible. If you want to force a new download, use `force_download=True`.
  warnings.warn(
input_embeddings: tensor([[[1., 1., 1.,  ..., 1., 1., 1.],
         [1., 1., 1.,  ..., 1., 1., 1.],
         [1., 1., 1.,  ..., 1., 1., 1.],
         ...,
         [1., 1., 1.,  ..., 1., 1., 1.],
         [1., 1., 1.,  ..., 1., 1., 1.],
         [1., 1., 1.,  ..., 1., 1., 1.]]])
Input embeddings shape: torch.Size([1, 384, 768])
Input sample: tensor([1., 1., 1., 1., 1.], device='cuda:0')

--- Setting Original Model Weights to 1 ---
Original model weights set to 1

--- Running Original Model with Unit Weights ---
Original output shape: torch.Size([1, 384, 768])
Original output sample: tensor([ 0.5606, -0.0302, -0.5636, -0.7326,  0.7882], device='cuda:0')
Original output stats - Min: -5.727265, Max: 1.323480, Mean: -0.012385

--- Creating Unit Weights for souffle_bert_layer ---
QKV weight shape: torch.Size([3, 768, 768])
Attn FC weight shape: torch.Size([768, 768])
FF1 weight shape: torch.Size([768, 3072])
FF2 weight shape: torch.Size([3072, 768])

--- Running souffle_bert_layer with Unit Weights ---
fused_attn
fused_feedforward
Souffle output type: <class 'torch.Tensor'>
Single output shape: torch.Size([384, 768])
Single output sample: tensor([nan, nan, nan, nan, nan], device='cuda:0', dtype=torch.float16)
Single output stats - Min: nan, Max: nan, Mean: nan

--- Comparing Single Output (Reshaped) ---

=== Comparing Single Output (reshaped) Output ===
Original shape: torch.Size([1, 384, 768])
Single Output (reshaped) shape: torch.Size([1, 384, 768])
L2 norm difference: nan
Relative difference: nan
Max absolute difference: nan
Mean absolute difference: nan
*** NaN detected! Original: 0, Single Output (reshaped): 294912 ***

Sample comparison (first 5 elements):
Original:  tensor([ 0.5606, -0.0302, -0.5636, -0.7326,  0.7882], device='cuda:0')
Single Output (reshaped): tensor([nan, nan, nan, nan, nan], device='cuda:0')
Diff:      tensor([nan, nan, nan, nan, nan], device='cuda:0')
*** POOR MATCH! (> 1%) ***

*** 单位权重测试成功! ***

=== Testing souffle_bert_layer with Real BERT Layer 0 Weights ===
Loading model: MattBoraske/BERT-question-answering-SQuAD
Input embeddings shape: torch.Size([1, 384, 768])
Original weight shapes:
  Q weight: torch.Size([768, 768])
  K weight: torch.Size([768, 768])
  V weight: torch.Size([768, 768])
  Attn output: torch.Size([768, 768])
  FF1: torch.Size([3072, 768])
  FF2: torch.Size([768, 3072])

--- Running Original Model Layer 0 ---
Original layer 0 output shape: torch.Size([1, 384, 768])
Original output sample: tensor([ 0.2156, -0.0810, -0.0950, -0.2347,  0.0978], device='cuda:0')

--- Testing Config 1: Stack QKV, no transpose ---
  QKV weight shape: torch.Size([3, 768, 768])
  Attn FC weight shape: torch.Size([768, 768])
  FF1 weight shape: torch.Size([768, 3072])
  FF2 weight shape: torch.Size([3072, 768])
fused_attn
fused_feedforward
  Souffle output type: <class 'list'>
  Number of outputs: 2
    Output[0] shape: torch.Size([384, 768])
    Output[1] shape: torch.Size([384, 768])
  Final output shape: torch.Size([384, 768])
  Shape mismatch! Souffle: torch.Size([384, 768]), Original: torch.Size([1, 384, 768])
  Reshaped to: torch.Size([1, 384, 768])
  L2 norm difference: 545.95764160
  Relative difference: 1.82201290
  *** POOR MATCH! Relative difference > 10% ***

--- Testing Config 2: Stack QKV transposed ---
  QKV weight shape: torch.Size([3, 768, 768])
  Attn FC weight shape: torch.Size([768, 768])
  FF1 weight shape: torch.Size([768, 3072])
  FF2 weight shape: torch.Size([3072, 768])
fused_attn
fused_feedforward
  Souffle output type: <class 'list'>
  Number of outputs: 2
    Output[0] shape: torch.Size([384, 768])
    Output[1] shape: torch.Size([384, 768])
  Final output shape: torch.Size([384, 768])
  Shape mismatch! Souffle: torch.Size([384, 768]), Original: torch.Size([1, 384, 768])
  Reshaped to: torch.Size([1, 384, 768])
  L2 norm difference: 547.16674805
  Relative difference: 1.82604802
  *** POOR MATCH! Relative difference > 10% ***

--- Testing Config 3: All weights transposed ---
  QKV weight shape: torch.Size([3, 768, 768])
  Attn FC weight shape: torch.Size([768, 768])
  FF1 weight shape: torch.Size([3072, 768])
  FF2 weight shape: torch.Size([768, 3072])
fused_attn
fused_feedforward
  Souffle output type: <class 'list'>
  Number of outputs: 2
    Output[0] shape: torch.Size([384, 768])
    Output[1] shape: torch.Size([384, 768])
  Final output shape: torch.Size([384, 768])
  Shape mismatch! Souffle: torch.Size([384, 768]), Original: torch.Size([1, 384, 768])
  Reshaped to: torch.Size([1, 384, 768])
  L2 norm difference: 547.16674805
  Relative difference: 1.82604802
  *** POOR MATCH! Relative difference > 10% ***

--- Testing Config 4: Only attn_fc transposed ---
  QKV weight shape: torch.Size([3, 768, 768])
  Attn FC weight shape: torch.Size([768, 768])
  FF1 weight shape: torch.Size([768, 3072])
  FF2 weight shape: torch.Size([3072, 768])
fused_attn
fused_feedforward
  Souffle output type: <class 'list'>
  Number of outputs: 2
    Output[0] shape: torch.Size([384, 768])
    Output[1] shape: torch.Size([384, 768])
  Final output shape: torch.Size([384, 768])
  Shape mismatch! Souffle: torch.Size([384, 768]), Original: torch.Size([1, 384, 768])
  Reshaped to: torch.Size([1, 384, 768])
  L2 norm difference: 545.95764160
  Relative difference: 1.82201290
  *** POOR MATCH! Relative difference > 10% ***

--- Testing Config 5: Original HF format ---
  QKV weight shape: torch.Size([3, 768, 768])
  Attn FC weight shape: torch.Size([768, 768])
  FF1 weight shape: torch.Size([3072, 768])
  FF2 weight shape: torch.Size([768, 3072])
fused_attn
fused_feedforward
  Souffle output type: <class 'list'>
  Number of outputs: 2
    Output[0] shape: torch.Size([384, 768])
    Output[1] shape: torch.Size([384, 768])
  Final output shape: torch.Size([384, 768])
  Shape mismatch! Souffle: torch.Size([384, 768]), Original: torch.Size([1, 384, 768])
  Reshaped to: torch.Size([1, 384, 768])
  L2 norm difference: 545.95764160
  Relative difference: 1.82201290
  *** POOR MATCH! Relative difference > 10% ***

=== SUMMARY ===
Best configuration: Config 1: Stack QKV, no transpose
Best L2 difference: 545.95764160
FAILURE: No configuration achieved good match.
Basic souffle_bert_layer test failed. Check the binding implementation.

Basic test passed, loading full model...
Loading model: MattBoraske/BERT-question-answering-SQuAD
Extracting BERT weights (Fixed Version)...
Layer 0 weight shapes:
  QKV: torch.Size([3, 768, 768])
  Attn output: torch.Size([768, 768])
  FF1: torch.Size([3072, 768]) -> torch.Size([768, 3072])
  FF2: torch.Size([768, 3072]) -> torch.Size([3072, 768])
Layer 1 weight shapes:
  QKV: torch.Size([3, 768, 768])
  Attn output: torch.Size([768, 768])
  FF1: torch.Size([3072, 768]) -> torch.Size([768, 3072])
  FF2: torch.Size([768, 3072]) -> torch.Size([3072, 768])
Layer 2 weight shapes:
  QKV: torch.Size([3, 768, 768])
  Attn output: torch.Size([768, 768])
  FF1: torch.Size([3072, 768]) -> torch.Size([768, 3072])
  FF2: torch.Size([768, 3072]) -> torch.Size([3072, 768])
Layer 3 weight shapes:
  QKV: torch.Size([3, 768, 768])
  Attn output: torch.Size([768, 768])
  FF1: torch.Size([3072, 768]) -> torch.Size([768, 3072])
  FF2: torch.Size([768, 3072]) -> torch.Size([3072, 768])
Layer 4 weight shapes:
  QKV: torch.Size([3, 768, 768])
  Attn output: torch.Size([768, 768])
  FF1: torch.Size([3072, 768]) -> torch.Size([768, 3072])
  FF2: torch.Size([768, 3072]) -> torch.Size([3072, 768])
Layer 5 weight shapes:
  QKV: torch.Size([3, 768, 768])
  Attn output: torch.Size([768, 768])
  FF1: torch.Size([3072, 768]) -> torch.Size([768, 3072])
  FF2: torch.Size([768, 3072]) -> torch.Size([3072, 768])
Layer 6 weight shapes:
  QKV: torch.Size([3, 768, 768])
  Attn output: torch.Size([768, 768])
  FF1: torch.Size([3072, 768]) -> torch.Size([768, 3072])
  FF2: torch.Size([768, 3072]) -> torch.Size([3072, 768])
Layer 7 weight shapes:
  QKV: torch.Size([3, 768, 768])
  Attn output: torch.Size([768, 768])
  FF1: torch.Size([3072, 768]) -> torch.Size([768, 3072])
  FF2: torch.Size([768, 3072]) -> torch.Size([3072, 768])
Layer 8 weight shapes:
  QKV: torch.Size([3, 768, 768])
  Attn output: torch.Size([768, 768])
  FF1: torch.Size([3072, 768]) -> torch.Size([768, 3072])
  FF2: torch.Size([768, 3072]) -> torch.Size([3072, 768])
Layer 9 weight shapes:
  QKV: torch.Size([3, 768, 768])
  Attn output: torch.Size([768, 768])
  FF1: torch.Size([3072, 768]) -> torch.Size([768, 3072])
  FF2: torch.Size([768, 3072]) -> torch.Size([3072, 768])
Layer 10 weight shapes:
  QKV: torch.Size([3, 768, 768])
  Attn output: torch.Size([768, 768])
  FF1: torch.Size([3072, 768]) -> torch.Size([768, 3072])
  FF2: torch.Size([768, 3072]) -> torch.Size([3072, 768])
Layer 11 weight shapes:
  QKV: torch.Size([3, 768, 768])
  Attn output: torch.Size([768, 768])
  FF1: torch.Size([3072, 768]) -> torch.Size([768, 3072])
  FF2: torch.Size([768, 3072]) -> torch.Size([3072, 768])
Weight extraction completed!
output[0].shape torch.Size([1, 384, 768])
hidden_states[0].shape torch.Size([384, 768])
output[0].shape torch.Size([1, 384, 768])
hidden_states[0].shape torch.Size([384, 768])
output[0].shape torch.Size([1, 384, 768])
hidden_states[0].shape torch.Size([384, 768])
output[0].shape torch.Size([1, 384, 768])
hidden_states[0].shape torch.Size([384, 768])
output[0].shape torch.Size([1, 384, 768])
hidden_states[0].shape torch.Size([384, 768])
output[0].shape torch.Size([1, 384, 768])
hidden_states[0].shape torch.Size([384, 768])
output[0].shape torch.Size([1, 384, 768])
hidden_states[0].shape torch.Size([384, 768])
output[0].shape torch.Size([1, 384, 768])
hidden_states[0].shape torch.Size([384, 768])
output[0].shape torch.Size([1, 384, 768])
hidden_states[0].shape torch.Size([384, 768])
output[0].shape torch.Size([1, 384, 768])
hidden_states[0].shape torch.Size([384, 768])
output[0].shape torch.Size([1, 384, 768])
hidden_states[0].shape torch.Size([384, 768])
output[0].shape torch.Size([1, 384, 768])
hidden_states[0].shape torch.Size([384, 768])
Input embeddings shape: torch.Size([1, 384, 768])

=== Processing Layer 0 ===
Layer 0 weight shapes:
  qkv_weight: torch.Size([3, 768, 768])
  attn_fc_weight: torch.Size([768, 768])
  ff_fc1_weight: torch.Size([768, 3072])
  ff_fc2_weight: torch.Size([3072, 768])
fused_attn
fused_feedforward
Layer 0 outputs type: <class 'list'>
Number of outputs: 2
  Output[0] shape: torch.Size([384, 768])
  Output[0] sample: tensor([-0.3762, -1.2764, -0.0109], device='cuda:0', dtype=torch.float16)
  Output[1] shape: torch.Size([384, 768])
  Output[1] sample: tensor([-0.6343, -2.0449, -1.3330], device='cuda:0', dtype=torch.float16)
Next layer input shape: torch.Size([1, 384, 768])
Layer 0 L2 norm difference: 545.95721436
Relative difference: 1.82201326
*** Large difference detected at layer 0! ***
Optimized output stats:
  Min: -12.335938, Max: 5.839844, Mean: -0.000004
Original output stats:
  Min: -9.939704, Max: 2.695145, Mean: -0.022126

=== Processing Layer 1 ===
Layer 1 weight shapes:
  qkv_weight: torch.Size([3, 768, 768])
  attn_fc_weight: torch.Size([768, 768])
  ff_fc1_weight: torch.Size([768, 3072])
  ff_fc2_weight: torch.Size([3072, 768])
fused_attn
fused_feedforward
Layer 1 outputs type: <class 'list'>
Number of outputs: 2
  Output[0] shape: torch.Size([384, 768])
  Output[0] sample: tensor([-0.1086, -1.9424, -1.4600], device='cuda:0', dtype=torch.float16)
  Output[1] shape: torch.Size([384, 768])
  Output[1] sample: tensor([ 1.3848, -1.1582,  0.2180], device='cuda:0', dtype=torch.float16)
Next layer input shape: torch.Size([1, 384, 768])
Layer 1 L2 norm difference: 632.73785400
Relative difference: 1.54907346
*** Large difference detected at layer 1! ***
Optimized output stats:
  Min: -6.738281, Max: 4.566406, Mean: -0.000008
Original output stats:
  Min: -10.813071, Max: 3.426248, Mean: -0.024693
answer_start_index: tensor([207], device='cuda:0')
answer_end_index: tensor([18], device='cuda:0')
{'answer': '无法找到答案', 'start_logits': tensor([[-1.2715e+00, -7.7539e-01, -1.0488e+00, -1.0107e+00, -1.0244e+00,
         -4.0454e-01, -1.6924e+00, -6.4502e-01, -9.6631e-01, -1.0928e+00,
         -6.8604e-01, -5.3564e-01, -3.0151e-01,  3.6450e-01, -2.2388e-01,
         -5.7568e-01, -6.6504e-01, -8.3057e-01,  3.4607e-02, -1.0052e-01,
          6.9238e-01, -3.5352e-01,  1.4392e-01,  6.6064e-01,  1.0703e+00,
          7.8955e-01,  5.0342e-01,  5.3516e-01,  4.9927e-01,  6.9922e-01,
          3.6182e-01,  1.3672e-01,  1.8433e-01,  2.4133e-01,  4.0112e-01,
          5.5762e-01,  1.1169e-01,  4.1992e-01,  5.8203e-01,  5.6250e-01,
          3.0908e-01, -6.8848e-02, -4.8950e-02,  1.5967e-01,  1.2817e-01,
          1.0876e-01, -2.6782e-01, -3.8794e-01,  3.5205e-01,  4.0015e-01,
          2.3572e-01,  3.1372e-01,  1.2610e-01,  6.3818e-01,  6.9727e-01,
          2.6929e-01,  9.7778e-02, -1.9446e-01, -1.2042e-01,  3.2153e-01,
         -3.4521e-01, -6.5674e-01, -2.5366e-01,  2.8038e-03,  2.9199e-01,
         -1.3025e-01, -1.3281e-01,  2.0312e-01,  2.4634e-01,  2.6611e-02,
         -3.4277e-01, -7.5391e-01, -8.1726e-02, -1.4270e-01, -4.5532e-01,
         -3.8184e-01, -6.3867e-01, -5.0293e-02,  1.5955e-01,  3.3600e-02,
          1.0004e-01,  3.8239e-02,  5.1331e-02,  2.7539e-01,  1.3806e-01,
         -4.8145e-01, -1.6431e-01, -3.9246e-02, -4.4250e-02, -1.1841e-01,
         -3.0615e-01, -3.1982e-01,  1.5747e-01,  5.1123e-01,  2.3938e-01,
          4.8248e-02, -5.1605e-02,  2.9907e-01,  1.2238e-01, -4.8615e-02,
         -1.6553e-01, -1.4771e-01,  1.0065e-01,  2.3877e-01,  7.7148e-02,
          1.3354e-01,  3.5913e-01,  4.6729e-01,  4.4507e-01,  6.7334e-01,
          1.4465e-01,  3.0273e-01,  5.0781e-01,  4.7925e-01,  1.6382e-01,
         -1.6980e-01, -4.0698e-01,  5.4016e-02,  5.7098e-02, -1.8713e-01,
         -2.6953e-01,  8.7509e-03,  1.1554e-01,  1.9275e-01, -6.7871e-02,
         -8.2031e-02, -2.3560e-01, -3.0469e-01,  5.9204e-02, -2.3694e-01,
         -8.9746e-01, -4.0332e-01,  1.8372e-01,  2.4573e-01,  5.4443e-01,
          9.4482e-02,  3.3539e-02,  8.0615e-01,  3.9673e-01,  5.0684e-01,
          5.4291e-02,  9.0393e-02,  2.4731e-01,  3.1494e-01,  1.1389e-01,
         -1.8967e-02, -1.1322e-01, -3.9014e-01, -5.4785e-01, -3.0347e-01,
         -3.1934e-01, -2.4939e-01, -4.0356e-01, -2.4341e-01, -1.1246e-02,
          1.3696e-01, -2.5684e-01, -2.5854e-01, -2.8467e-01,  2.6779e-02,
         -1.3409e-03, -1.5332e-01, -4.7021e-01, -1.9873e-01,  9.6130e-02,
         -9.1187e-02,  7.5378e-02,  2.5955e-02,  1.7249e-01,  4.5312e-01,
          5.4736e-01,  1.5674e-01,  2.2253e-01,  4.3018e-01,  3.0933e-01,
          3.1567e-01,  9.5337e-02,  6.6113e-01,  4.3884e-02,  1.4026e-01,
          4.0649e-01, -7.9224e-02,  1.9824e-01,  7.8369e-01,  7.9248e-01,
          5.2441e-01,  3.7451e-01,  8.2178e-01,  5.5322e-01,  9.8145e-01,
          8.2227e-01,  5.1025e-01,  6.7041e-01,  9.8047e-01,  6.7529e-01,
          3.2861e-01,  1.0273e+00,  6.3379e-01,  4.1943e-01,  5.3955e-01,
          4.9390e-01,  9.2725e-01,  3.8428e-01,  5.3662e-01,  1.0010e+00,
          1.1201e+00,  8.5645e-01,  1.1719e+00,  9.8975e-01,  1.1709e+00,
          1.0723e+00,  7.4707e-01,  2.9492e-01,  5.9375e-01,  2.8223e-01,
          5.0342e-01,  6.7383e-02,  2.7197e-01,  6.5674e-01,  5.3125e-01,
          2.2913e-01,  3.1128e-01,  4.5142e-01,  6.4404e-01,  2.4329e-01,
         -6.6223e-02, -8.5510e-02,  1.6125e-01,  3.4155e-01, -3.0078e-01,
         -1.0712e-01,  3.5474e-01, -3.3984e-01,  2.0300e-01,  3.2056e-01,
          3.2532e-02,  3.6084e-01,  4.2920e-01,  3.4814e-01,  7.4170e-01,
          3.5718e-01,  1.8164e-01,  5.1611e-01,  7.9712e-02,  3.3887e-01,
          7.0801e-01,  8.7585e-02,  2.6538e-01,  7.5586e-01,  5.8447e-01,
          6.3184e-01,  1.1592e+00,  1.0358e-01,  2.8467e-01,  2.5708e-01,
          2.1472e-01,  5.4248e-01,  2.7783e-01, -1.9751e-01,  1.4563e-01,
         -1.8555e-01,  3.8843e-01,  2.0215e-01,  6.6467e-02,  4.6582e-01,
          4.8682e-01,  5.3955e-01,  6.7969e-01,  3.4888e-01,  3.2593e-01,
          4.1284e-01, -2.0801e-01,  6.1475e-01, -8.4717e-02, -5.5481e-02,
         -1.3452e-01, -2.3059e-01,  5.0830e-01,  2.7368e-01,  3.8257e-01,
          8.4839e-02,  5.4639e-01,  9.3213e-01,  1.1328e+00,  5.5029e-01,
          2.8052e-01,  7.2461e-01,  6.8262e-01,  7.0605e-01,  3.1543e-01,
          4.4702e-01, -3.7323e-02, -1.3135e-01, -4.2908e-02,  2.6685e-01,
         -1.2769e-01,  8.0750e-02, -2.0508e-01,  7.9163e-02,  4.9243e-01,
          2.7393e-01,  6.9043e-01,  4.2236e-01,  4.7729e-01,  6.1865e-01,
          7.4902e-01,  7.9541e-01,  5.8203e-01,  6.6064e-01,  3.0225e-01,
          2.7573e-02,  3.1885e-01,  1.6833e-01,  2.0581e-01,  2.3047e-01,
          4.2847e-01,  3.0933e-01,  3.7500e-01,  6.0498e-01,  3.8623e-01,
          8.7988e-01,  6.9336e-01,  4.3530e-01,  2.3938e-01,  2.3840e-01,
          2.2778e-01,  2.3328e-01, -6.1798e-02, -4.0308e-01, -3.1421e-01,
         -2.5391e-01,  4.9500e-02, -3.3081e-01, -4.7510e-01,  1.1841e-01,
          1.5930e-01,  6.4062e-01,  3.8721e-01,  6.7480e-01,  1.4612e-01,
          6.5918e-01,  8.2959e-01,  8.4326e-01,  4.6655e-01, -2.9488e-03,
          2.0459e-01,  1.6333e-01,  8.5449e-02,  2.6440e-01,  2.1744e-02,
         -3.8892e-01,  2.5659e-01,  7.3608e-02,  9.2957e-02,  1.1945e-01,
         -1.0638e-01,  7.9041e-02,  5.0146e-01, -1.9031e-01,  4.7314e-01,
          2.4573e-01,  1.7871e-01,  2.8833e-01,  1.9287e-01,  1.2537e-01,
          6.8848e-01,  5.3955e-01,  1.3696e-01,  4.7437e-01,  1.4246e-01,
         -1.9302e-02,  1.8115e-01,  5.9448e-02,  3.0493e-01,  2.6904e-01,
         -4.4019e-01,  1.1871e-01,  3.9453e-01,  1.8921e-01,  1.9458e-01,
          1.3257e-01,  1.7371e-01,  6.6797e-01, -2.5562e-01]], device='cuda:0',
       dtype=torch.float16), 'end_logits': tensor([[-0.1119, -0.7354,  0.0226,  0.0279,  0.5117,  1.1328, -0.6543, -0.1688,
          0.8408, -0.2347, -0.3274,  0.3533, -0.1934,  0.8335, -0.6079, -0.6582,
         -0.1333,  0.5684,  1.3457, -0.6230,  0.8809, -0.3577,  0.1646,  0.9233,
          0.8696,  0.9761,  0.7373,  0.5991,  0.8862,  0.8164,  0.5996,  0.7798,
          0.4495,  0.6187,  0.7573,  0.6919,  0.6685,  0.6709,  0.5391,  0.8154,
          0.4314,  0.3691,  0.4971,  0.3523,  0.4988,  0.4192,  0.2382,  0.4846,
          0.6431,  0.7119,  0.9302,  0.7290,  0.7427,  0.9888,  0.7549,  0.7178,
          0.5825,  0.4368,  0.8730,  0.5786,  0.2788,  0.2303,  0.1749,  0.4700,
          0.7749,  0.4702,  0.5674,  0.5776,  0.4910,  0.3682, -0.2256, -0.1094,
          0.1923,  0.0120,  0.0302,  0.0255, -0.2795,  0.2781,  0.2433,  0.3259,
          0.2949,  0.2244,  0.1443,  0.5400,  0.1831,  0.1317, -0.0916,  0.0509,
          0.4932,  0.1335, -0.1635,  0.0399, -0.1130,  0.2351,  0.3247, -0.1306,
          0.1436,  0.1827, -0.0911, -0.2277, -0.1276,  0.0289,  0.1506,  0.1742,
          0.2240,  0.1687,  0.2350,  0.3286,  0.4238,  0.2588,  0.2500,  0.2495,
          0.1139,  0.4082,  0.1825, -0.0091,  0.2676,  0.1060,  0.1345,  0.3862,
         -0.0253,  0.0849,  0.4241,  0.2443,  0.0820,  0.2169,  0.0152,  0.2140,
          0.3560,  0.4221,  0.6367,  0.4854,  0.4573,  0.8872,  0.9126,  0.0630,
          0.6577,  0.7134,  0.3479,  0.8232,  0.5112,  0.6851,  0.7368,  0.6294,
          0.5854,  0.7266,  0.7017,  0.3364,  0.2656,  0.2915,  0.1718,  0.5420,
          0.3523,  0.4336,  0.4614,  0.7695,  0.0842,  0.3059,  0.5479,  0.3772,
          0.0208,  0.1046,  0.1582,  0.3152,  0.0648,  0.0345,  0.0228,  0.3096,
          0.0657,  0.5317,  0.2219,  0.3252,  0.2563,  0.6055,  0.3882,  0.1871,
          0.0023,  0.5249,  0.1819,  0.4187,  0.1360,  0.1276,  0.1554,  0.5176,
          0.6753,  0.1931,  0.5322,  0.4731,  0.5200,  0.4783,  0.2300,  0.3428,
          0.3806,  0.6758,  0.2690,  0.4153,  0.6138,  0.4309,  0.2128,  0.5420,
          0.5376,  0.4580,  0.5811,  0.6885,  0.6997,  0.4841,  0.5776,  0.6636,
          0.4480,  0.7896,  0.5972,  0.7090,  0.4814,  0.4255,  0.4175,  0.7505,
          0.4294,  0.5054,  0.3076,  0.5566,  0.5552,  0.4287,  0.7559,  0.5073,
          0.3215,  0.1539,  0.1808,  0.1835,  0.0998, -0.3257, -0.2805,  0.0851,
         -0.2371,  0.3987,  0.3135,  0.0897,  0.1630,  0.1111,  0.0785,  0.3855,
         -0.1245, -0.0202,  0.3079, -0.1896, -0.0666, -0.0718, -0.3193, -0.0380,
          0.3425, -0.0666,  0.3584,  0.7544, -0.2217,  0.1851,  0.1539,  0.1931,
         -0.0336,  0.2539, -0.0618, -0.0795,  0.1693,  0.1482,  0.5327,  0.2603,
          0.4790,  0.4321,  0.3508,  0.7319,  0.3621,  0.5049,  0.3250,  0.2223,
          0.1064, -0.0687, -0.0494, -0.5547, -0.3662,  0.1366,  0.2373,  0.2664,
         -0.1011,  0.4165,  0.6294,  0.5654,  0.2981,  0.3301,  0.4075,  0.4731,
          0.3831,  0.4141,  0.2668, -0.1909,  0.1930,  0.1562, -0.0164, -0.0115,
          0.1589, -0.2046,  0.1542,  0.0505,  0.3450,  0.3857,  0.3796,  0.4009,
          0.3167,  0.4983,  0.1794,  0.4128, -0.2369, -0.1620, -0.0409,  0.3499,
          0.0309,  0.0967,  0.0230,  0.0383, -0.0212,  0.3379,  0.1068,  0.1860,
          0.6216,  0.3047,  0.4553,  0.4763,  0.3557,  0.1667,  0.5269, -0.0027,
          0.2927,  0.4231,  0.2993,  0.2092,  0.0323,  0.2358,  0.1210,  0.4822,
          0.2148,  0.2634,  0.4536,  0.0748,  0.8066,  0.1973,  0.1080, -0.0645,
         -0.0367,  0.4575,  0.4128,  0.0331, -0.0377, -0.0383,  0.1489,  0.0183,
          0.3650,  0.4165,  0.0444,  0.5078,  0.1126,  0.0750, -0.2494,  0.2561,
         -0.0035,  0.1415,  0.1787,  0.1553,  0.1371,  0.3242,  0.2986,  0.3464,
          0.2141,  0.0046,  0.1239,  0.3071, -0.0747,  0.3147, -0.1295, -0.1451,
          0.0142,  0.0441,  0.1431,  0.0193,  0.2168,  0.3979,  0.5010, -0.1570]],
       device='cuda:0', dtype=torch.float16), 'start_index': 207, 'end_index': 18, 'error': False}
```
