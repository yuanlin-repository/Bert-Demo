# Bert-Demo

## Reproduce steps
1. ssh yuanlin@{ip}
2. docker exec -it ca023b78bfb0 /bin/bash
3. cd /workspace/souffle-models/python/models/bert
4. python bert_qa_optimization.py

**Sample output**
```shell
=== Unit Weight Test: All Weights Set to 1 ===
Loading model: MattBoraske/BERT-question-answering-SQuAD
/workspace/anaconda3/lib/python3.9/site-packages/huggingface_hub/file_download.py:945: FutureWarning: `resume_download` is deprecated and will be removed in version 1.0.0. Downloads always resume when possible. If you want to force a new download, use `force_download=True`.
  warnings.warn(
input_embeddings:
 tensor([[[1., 1., 1.,  ..., 1., 1., 1.],
         [1., 1., 1.,  ..., 1., 1., 1.],
         [1., 1., 1.,  ..., 1., 1., 1.],
         ...,
         [1., 1., 1.,  ..., 1., 1., 1.],
         [1., 1., 1.,  ..., 1., 1., 1.],
         [1., 1., 1.,  ..., 1., 1., 1.]]])

--- Setting Original Model Weights to 1 ---

--- Running Original Model with Unit Weights ---
Original output:
 tensor([[[ 0.5606, -0.0302, -0.5636,  ...,  0.3001, -0.1634, -0.2333],
         [ 0.5606, -0.0302, -0.5636,  ...,  0.3001, -0.1634, -0.2333],
         [ 0.5606, -0.0302, -0.5636,  ...,  0.3001, -0.1634, -0.2333],
         ...,
         [ 0.5606, -0.0302, -0.5636,  ...,  0.3001, -0.1634, -0.2333],
         [ 0.5606, -0.0302, -0.5636,  ...,  0.3001, -0.1634, -0.2333],
         [ 0.5606, -0.0302, -0.5636,  ...,  0.3001, -0.1634, -0.2333]]],
       device='cuda:0')

--- Creating Unit Weights for souffle_bert_layer ---

--- Running souffle_bert_layer with Unit Weights ---
fused_attn
fused_feedforward
souffle_bert_layer output:
 tensor([[nan, nan, nan,  ..., nan, nan, nan],
        [nan, nan, nan,  ..., nan, nan, nan],
        [nan, nan, nan,  ..., nan, nan, nan],
        ...,
        [nan, nan, nan,  ..., nan, nan, nan],
        [nan, nan, nan,  ..., nan, nan, nan],
        [nan, nan, nan,  ..., nan, nan, nan]], device='cuda:0')

--- Compare original_output and souffle_output ---
delta matrix:
 tensor([[[nan, nan, nan,  ..., nan, nan, nan],
         [nan, nan, nan,  ..., nan, nan, nan],
         [nan, nan, nan,  ..., nan, nan, nan],
         ...,
         [nan, nan, nan,  ..., nan, nan, nan],
         [nan, nan, nan,  ..., nan, nan, nan],
         [nan, nan, nan,  ..., nan, nan, nan]]], device='cuda:0')

*** 单位权重测试成功! ***

=== Testing souffle_bert_layer with Real BERT Layer 0 Weights ===
Loading model: MattBoraske/BERT-question-answering-SQuAD
input embeddings delta matrix(input_embeddings - input_embeddings.to(torch.float16).to(torch.float16)):
 tensor([[[-8.8215e-06, -6.4582e-05, -4.8667e-05,  ...,  3.9305e-08,
          -7.0333e-06, -5.3063e-05],
         [ 5.4255e-05,  4.1515e-05,  5.7787e-05,  ..., -3.9637e-05,
           5.3346e-05, -1.6838e-05],
         [-1.4728e-04, -1.5867e-04,  2.5094e-05,  ..., -2.4164e-04,
          -1.1519e-04, -2.1106e-04],
         ...,
         [-7.3344e-05,  2.0713e-04, -4.2766e-06,  ...,  1.3161e-04,
           2.1905e-06, -1.0431e-05],
         [ 1.9625e-05, -1.3530e-04, -4.5747e-05,  ...,  8.0228e-05,
           1.1399e-05,  1.0967e-04],
         [ 2.4885e-06,  1.9622e-04,  1.0151e-05,  ..., -1.4818e-04,
          -3.8594e-06,  1.7077e-05]]], device='cuda:0')

--- Running Original Model Layer 0 ---

--- Testing Config 1: Stack QKV, no transpose ---
fused_attn
fused_feedforward
  delta matrix:
 tensor([[[-0.8498, -1.9639, -1.2380,  ...,  0.6257,  0.2714,  0.2088],
         [-0.1458, -1.1572,  1.2654,  ...,  0.2389, -0.3984, -0.1095],
         [-0.6162,  0.1609,  0.9333,  ..., -0.6576, -0.6399, -0.1122],
         ...,
         [ 0.1895, -0.0020,  0.8376,  ..., -0.2132, -0.6871, -1.3387],
         [-0.2252,  0.0948,  0.8687,  ...,  0.5513, -0.2536, -1.1607],
         [-0.0807, -0.6074,  0.8607,  ...,  0.1184, -0.6046, -0.7333]]],
       device='cuda:0')
  L2 norm difference: 545.95764160
  *** POOR MATCH! Relative difference > 10% ***

--- Testing Config 2: Stack QKV transposed ---
fused_attn
fused_feedforward
  delta matrix:
 tensor([[[ 0.1487, -1.6016, -1.3757,  ...,  0.6965,  0.9699,  0.0625],
         [ 1.3020, -0.2136,  0.4316,  ...,  0.3038, -0.1748, -0.0949],
         [ 0.1539,  0.4471,  0.3088,  ..., -0.7445,  0.5047, -0.0106],
         ...,
         [ 0.6119,  0.7915,  0.2374,  ...,  0.4303, -0.0855, -0.9764],
         [ 0.3345,  0.9193,  0.2732,  ...,  1.3103,  0.1480, -0.7462],
         [ 0.5017,  0.1570,  0.3499,  ...,  0.9262, -0.0523, -0.5038]]],
       device='cuda:0')
  L2 norm difference: 547.16674805
  *** POOR MATCH! Relative difference > 10% ***

--- Testing Config 3: All weights transposed ---
fused_attn
fused_feedforward
  delta matrix:
 tensor([[[ 0.1487, -1.6016, -1.3757,  ...,  0.6965,  0.9699,  0.0625],
         [ 1.3020, -0.2136,  0.4316,  ...,  0.3038, -0.1748, -0.0949],
         [ 0.1539,  0.4471,  0.3088,  ..., -0.7445,  0.5047, -0.0106],
         ...,
         [ 0.6119,  0.7915,  0.2374,  ...,  0.4303, -0.0855, -0.9764],
         [ 0.3345,  0.9193,  0.2732,  ...,  1.3103,  0.1480, -0.7462],
         [ 0.5017,  0.1570,  0.3499,  ...,  0.9262, -0.0523, -0.5038]]],
       device='cuda:0')
  L2 norm difference: 547.16674805
  *** POOR MATCH! Relative difference > 10% ***

--- Testing Config 4: Only attn_fc transposed ---
fused_attn
fused_feedforward
  delta matrix:
 tensor([[[-0.8498, -1.9639, -1.2380,  ...,  0.6257,  0.2714,  0.2088],
         [-0.1458, -1.1572,  1.2654,  ...,  0.2389, -0.3984, -0.1095],
         [-0.6162,  0.1609,  0.9333,  ..., -0.6576, -0.6399, -0.1122],
         ...,
         [ 0.1895, -0.0020,  0.8376,  ..., -0.2132, -0.6871, -1.3387],
         [-0.2252,  0.0948,  0.8687,  ...,  0.5513, -0.2536, -1.1607],
         [-0.0807, -0.6074,  0.8607,  ...,  0.1184, -0.6046, -0.7333]]],
       device='cuda:0')
  L2 norm difference: 545.95764160
  *** POOR MATCH! Relative difference > 10% ***

--- Testing Config 5: Original HF format ---
fused_attn
fused_feedforward
  delta matrix:
 tensor([[[-0.8498, -1.9639, -1.2380,  ...,  0.6257,  0.2714,  0.2088],
         [-0.1458, -1.1572,  1.2654,  ...,  0.2389, -0.3984, -0.1095],
         [-0.6162,  0.1609,  0.9333,  ..., -0.6576, -0.6399, -0.1122],
         ...,
         [ 0.1895, -0.0020,  0.8376,  ..., -0.2132, -0.6871, -1.3387],
         [-0.2252,  0.0948,  0.8687,  ...,  0.5513, -0.2536, -1.1607],
         [-0.0807, -0.6074,  0.8607,  ...,  0.1184, -0.6046, -0.7333]]],
       device='cuda:0')
  L2 norm difference: 545.95764160
  *** POOR MATCH! Relative difference > 10% ***
```
