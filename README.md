# Bert-Demo

## Reproduce steps
1. enter container `souffle-tvm-0.8:latest`
2. cd /workspace/souffle-models/python/models/bert
3. copy, paste test_multi_souffle_layer.py
4. python test_multi_souffle_layer.py

## Sample output


### 每层输出(使用 torch.testing.assert_close 对比)
```shell
---------------------layer-0-count-of-nan---------------------

output_qkv: 0
query_key_output: 0
attn_value_output: 0
attn_fc_output: 0
feed_forward_fc1_output: 0
feed_forward_fc2_output: 0
✅ tensor: tensors are close within tolerance.
✅ tensor: tensors are close within tolerance.
✅ tensor: tensors are close within tolerance.
✅ tensor: tensors are close within tolerance.
❌ tensor: tensors differ beyond tolerance.
Tensor-likes are not close!

Mismatched elements: 1 / 1179648 (0.0%)
Greatest absolute difference: 0.015625 at index (355, 2180) (up to 0.01 allowed)
Greatest relative difference: 0.032928466796875 at index (355, 2180) (up to 0.01 allowed)
✅ tensor: tensors are close within tolerance.
Layer0 output match original model success!!!
fused_attn
fused_feedforward
---------------------layer- 1 ---------------------

---------------------layer- 1 -count-of-nan---------------------

output_qkv: 0
query_key_output 0
attn_value_output 0
attn_fc_output 0
feed_forward_fc1_output 0
feed_forward_fc2_output 0
❌ tensor: tensors differ beyond tolerance.
Tensor-likes are not close!

Mismatched elements: 15 / 884736 (0.0%)
Greatest absolute difference: 0.01611328125 at index (1, 8, 231, 45) (up to 0.01 allowed)
Greatest relative difference: 55.9375 at index (1, 2, 46, 34) (up to 0.01 allowed)
✅ tensor: tensors are close within tolerance.
✅ tensor: tensors are close within tolerance.
❌ tensor: tensors differ beyond tolerance.
Tensor-likes are not close!

Mismatched elements: 2 / 294912 (0.0%)
Greatest absolute difference: 0.0135498046875 at index (178, 225) (up to 0.01 allowed)
Greatest relative difference: 0.11199951171875 at index (206, 225) (up to 0.01 allowed)
❌ tensor: tensors differ beyond tolerance.
Tensor-likes are not close!

Mismatched elements: 1 / 1179648 (0.0%)
Greatest absolute difference: 0.011474609375 at index (341, 204) (up to 0.01 allowed)
Greatest relative difference: 0.09857177734375 at index (341, 204) (up to 0.01 allowed)
✅ tensor: tensors are close within tolerance.
fused_attn
fused_feedforward
---------------------layer- 2 ---------------------

---------------------layer- 2 -count-of-nan---------------------

output_qkv: 0
query_key_output 0
attn_value_output 0
attn_fc_output 0
feed_forward_fc1_output 0
feed_forward_fc2_output 0
❌ tensor: tensors differ beyond tolerance.
Tensor-likes are not close!

Mismatched elements: 10 / 884736 (0.0%)
Greatest absolute difference: 0.0133056640625 at index (0, 0, 190, 20) (up to 0.01 allowed)
Greatest relative difference: 5.125 at index (1, 1, 169, 12) (up to 0.01 allowed)
✅ tensor: tensors are close within tolerance.
✅ tensor: tensors are close within tolerance.
✅ tensor: tensors are close within tolerance.
❌ tensor: tensors differ beyond tolerance.
Tensor-likes are not close!

Mismatched elements: 6 / 1179648 (0.0%)
Greatest absolute difference: 0.029296875 at index (103, 1072) (up to 0.01 allowed)
Greatest relative difference: 0.1138916015625 at index (94, 1072) (up to 0.01 allowed)
❌ tensor: tensors differ beyond tolerance.
Tensor-likes are not close!

Mismatched elements: 3 / 294912 (0.0%)
Greatest absolute difference: 0.07421875 at index (103, 381) (up to 0.02 allowed)
Greatest relative difference: 0.0252532958984375 at index (103, 381) (up to 0.01 allowed)
fused_attn
fused_feedforward
---------------------layer- 3 ---------------------

---------------------layer- 3 -count-of-nan---------------------

output_qkv: 0
query_key_output 0
attn_value_output 0
attn_fc_output 0
feed_forward_fc1_output 0
feed_forward_fc2_output 0
❌ tensor: tensors differ beyond tolerance.
Tensor-likes are not close!

Mismatched elements: 151 / 884736 (0.0%)
Greatest absolute difference: 0.046875 at index (1, 8, 102, 9) (up to 0.01 allowed)
Greatest relative difference: 21.234375 at index (1, 4, 65, 47) (up to 0.01 allowed)
✅ tensor: tensors are close within tolerance.
✅ tensor: tensors are close within tolerance.
❌ tensor: tensors differ beyond tolerance.
Tensor-likes are not close!

Mismatched elements: 11 / 294912 (0.0%)
Greatest absolute difference: 0.064453125 at index (103, 381) (up to 0.01 allowed)
Greatest relative difference: 0.36474609375 at index (319, 249) (up to 0.01 allowed)
❌ tensor: tensors differ beyond tolerance.
Tensor-likes are not close!

Mismatched elements: 16 / 1179648 (0.0%)
Greatest absolute difference: 0.0390625 at index (99, 1693) (up to 0.01 allowed)
Greatest relative difference: 0.424560546875 at index (103, 2602) (up to 0.01 allowed)
❌ tensor: tensors differ beyond tolerance.
Tensor-likes are not close!

Mismatched elements: 3 / 294912 (0.0%)
Greatest absolute difference: 0.142578125 at index (103, 381) (up to 0.02 allowed)
Greatest relative difference: 0.050567626953125 at index (103, 381) (up to 0.01 allowed)
fused_attn
fused_feedforward
---------------------layer- 4 ---------------------

---------------------layer- 4 -count-of-nan---------------------

output_qkv: 0
query_key_output 223
attn_value_output 11776
attn_fc_output 108288
feed_forward_fc1_output 433152
feed_forward_fc2_output 108288
❌ tensor: tensors differ beyond tolerance.
Tensor-likes are not close!

Mismatched elements: 1108 / 884736 (0.1%)
Greatest absolute difference: 0.09375 at index (1, 8, 99, 48) (up to 0.01 allowed)
Greatest relative difference: 193.875 at index (1, 4, 99, 39) (up to 0.01 allowed)
❌ tensor: tensors differ beyond tolerance.
Tensor-likes are not close!

Mismatched elements: 537 / 1769472 (0.0%)
Greatest absolute difference: nan at index (0, 0, 121) (up to 0.01 allowed)
Greatest relative difference: nan at index (0, 0, 121) (up to 0.01 allowed)
❌ tensor: tensors differ beyond tolerance.
Tensor-likes are not close!

Mismatched elements: 11778 / 294912 (4.0%)
Greatest absolute difference: nan at index (0, 0) (up to 0.01 allowed)
Greatest relative difference: nan at index (0, 0) (up to 0.01 allowed)
❌ tensor: tensors differ beyond tolerance.
Tensor-likes are not close!

Mismatched elements: 108295 / 294912 (36.7%)
Greatest absolute difference: nan at index (0, 0) (up to 0.01 allowed)
Greatest relative difference: nan at index (0, 0) (up to 0.01 allowed)
❌ tensor: tensors differ beyond tolerance.
Tensor-likes are not close!

Mismatched elements: 433213 / 1179648 (36.7%)
Greatest absolute difference: nan at index (0, 0) (up to 0.01 allowed)
Greatest relative difference: nan at index (0, 0) (up to 0.01 allowed)
❌ tensor: tensors differ beyond tolerance.
Tensor-likes are not close!

Mismatched elements: 108334 / 294912 (36.7%)
Greatest absolute difference: nan at index (0, 0) (up to 0.02 allowed)
Greatest relative difference: nan at index (0, 0) (up to 0.01 allowed)
fused_attn
fused_feedforward
---------------------layer- 5 ---------------------

---------------------layer- 5 -count-of-nan---------------------

output_qkv: 324864
query_key_output 1769472
attn_value_output 294912
attn_fc_output 294912
feed_forward_fc1_output 1179648
feed_forward_fc2_output 294912
❌ tensor: tensors differ beyond tolerance.
Tensor-likes are not close!

Mismatched elements: 328212 / 884736 (37.1%)
Greatest absolute difference: nan at index (0, 0, 0, 0) (up to 0.01 allowed)
Greatest relative difference: nan at index (0, 0, 0, 0) (up to 0.01 allowed)
❌ tensor: tensors differ beyond tolerance.
Tensor-likes are not close!

Mismatched elements: 1769472 / 1769472 (100.0%)
Greatest absolute difference: nan at index (0, 0, 0) (up to 0.01 allowed)
Greatest relative difference: nan at index (0, 0, 0) (up to 0.01 allowed)
❌ tensor: tensors differ beyond tolerance.
Tensor-likes are not close!

Mismatched elements: 294912 / 294912 (100.0%)
Greatest absolute difference: nan at index (0, 0) (up to 0.01 allowed)
Greatest relative difference: nan at index (0, 0) (up to 0.01 allowed)
❌ tensor: tensors differ beyond tolerance.
Tensor-likes are not close!

Mismatched elements: 294912 / 294912 (100.0%)
Greatest absolute difference: nan at index (0, 0) (up to 0.01 allowed)
Greatest relative difference: nan at index (0, 0) (up to 0.01 allowed)
❌ tensor: tensors differ beyond tolerance.
Tensor-likes are not close!

Mismatched elements: 1179648 / 1179648 (100.0%)
Greatest absolute difference: nan at index (0, 0) (up to 0.01 allowed)
Greatest relative difference: nan at index (0, 0) (up to 0.01 allowed)
❌ tensor: tensors differ beyond tolerance.
Tensor-likes are not close!

Mismatched elements: 294912 / 294912 (100.0%)
Greatest absolute difference: nan at index (0, 0) (up to 0.02 allowed)
Greatest relative difference: nan at index (0, 0) (up to 0.01 allowed)
fused_attn
fused_feedforward
---------------------layer- 6 ---------------------

---------------------layer- 6 -count-of-nan---------------------

output_qkv: 884736
query_key_output 1769472
attn_value_output 294912
attn_fc_output 294912
feed_forward_fc1_output 1179648
feed_forward_fc2_output 294912
❌ tensor: tensors differ beyond tolerance.
Tensor-likes are not close!

Mismatched elements: 884736 / 884736 (100.0%)
Greatest absolute difference: nan at index (0, 0, 0, 0) (up to 0.01 allowed)
Greatest relative difference: nan at index (0, 0, 0, 0) (up to 0.01 allowed)
❌ tensor: tensors differ beyond tolerance.
Tensor-likes are not close!

Mismatched elements: 1769472 / 1769472 (100.0%)
Greatest absolute difference: nan at index (0, 0, 0) (up to 0.01 allowed)
Greatest relative difference: nan at index (0, 0, 0) (up to 0.01 allowed)
❌ tensor: tensors differ beyond tolerance.
Tensor-likes are not close!

Mismatched elements: 294912 / 294912 (100.0%)
Greatest absolute difference: nan at index (0, 0) (up to 0.01 allowed)
Greatest relative difference: nan at index (0, 0) (up to 0.01 allowed)
❌ tensor: tensors differ beyond tolerance.
Tensor-likes are not close!

Mismatched elements: 294912 / 294912 (100.0%)
Greatest absolute difference: nan at index (0, 0) (up to 0.01 allowed)
Greatest relative difference: nan at index (0, 0) (up to 0.01 allowed)
❌ tensor: tensors differ beyond tolerance.
Tensor-likes are not close!

Mismatched elements: 1179648 / 1179648 (100.0%)
Greatest absolute difference: nan at index (0, 0) (up to 0.01 allowed)
Greatest relative difference: nan at index (0, 0) (up to 0.01 allowed)
❌ tensor: tensors differ beyond tolerance.
Tensor-likes are not close!

Mismatched elements: 294912 / 294912 (100.0%)
Greatest absolute difference: nan at index (0, 0) (up to 0.02 allowed)
Greatest relative difference: nan at index (0, 0) (up to 0.01 allowed)
fused_attn
fused_feedforward
---------------------layer- 7 ---------------------

---------------------layer- 7 -count-of-nan---------------------

output_qkv: 884736
query_key_output 1769472
attn_value_output 294912
attn_fc_output 294912
feed_forward_fc1_output 1179648
feed_forward_fc2_output 294912
❌ tensor: tensors differ beyond tolerance.
Tensor-likes are not close!

Mismatched elements: 884736 / 884736 (100.0%)
Greatest absolute difference: nan at index (0, 0, 0, 0) (up to 0.01 allowed)
Greatest relative difference: nan at index (0, 0, 0, 0) (up to 0.01 allowed)
❌ tensor: tensors differ beyond tolerance.
Tensor-likes are not close!

Mismatched elements: 1769472 / 1769472 (100.0%)
Greatest absolute difference: nan at index (0, 0, 0) (up to 0.01 allowed)
Greatest relative difference: nan at index (0, 0, 0) (up to 0.01 allowed)
❌ tensor: tensors differ beyond tolerance.
Tensor-likes are not close!

Mismatched elements: 294912 / 294912 (100.0%)
Greatest absolute difference: nan at index (0, 0) (up to 0.01 allowed)
Greatest relative difference: nan at index (0, 0) (up to 0.01 allowed)
❌ tensor: tensors differ beyond tolerance.
Tensor-likes are not close!

Mismatched elements: 294912 / 294912 (100.0%)
Greatest absolute difference: nan at index (0, 0) (up to 0.01 allowed)
Greatest relative difference: nan at index (0, 0) (up to 0.01 allowed)
❌ tensor: tensors differ beyond tolerance.
Tensor-likes are not close!

Mismatched elements: 1179648 / 1179648 (100.0%)
Greatest absolute difference: nan at index (0, 0) (up to 0.01 allowed)
Greatest relative difference: nan at index (0, 0) (up to 0.01 allowed)
❌ tensor: tensors differ beyond tolerance.
Tensor-likes are not close!

Mismatched elements: 294912 / 294912 (100.0%)
Greatest absolute difference: nan at index (0, 0) (up to 0.02 allowed)
Greatest relative difference: nan at index (0, 0) (up to 0.01 allowed)
fused_attn
fused_feedforward
---------------------layer- 8 ---------------------

---------------------layer- 8 -count-of-nan---------------------

output_qkv: 884736
query_key_output 1769472
attn_value_output 294912
attn_fc_output 294912
feed_forward_fc1_output 1179648
feed_forward_fc2_output 294912
❌ tensor: tensors differ beyond tolerance.
Tensor-likes are not close!

Mismatched elements: 884736 / 884736 (100.0%)
Greatest absolute difference: nan at index (0, 0, 0, 0) (up to 0.01 allowed)
Greatest relative difference: nan at index (0, 0, 0, 0) (up to 0.01 allowed)
❌ tensor: tensors differ beyond tolerance.
Tensor-likes are not close!

Mismatched elements: 1769472 / 1769472 (100.0%)
Greatest absolute difference: nan at index (0, 0, 0) (up to 0.01 allowed)
Greatest relative difference: nan at index (0, 0, 0) (up to 0.01 allowed)
❌ tensor: tensors differ beyond tolerance.
Tensor-likes are not close!

Mismatched elements: 294912 / 294912 (100.0%)
Greatest absolute difference: nan at index (0, 0) (up to 0.01 allowed)
Greatest relative difference: nan at index (0, 0) (up to 0.01 allowed)
❌ tensor: tensors differ beyond tolerance.
Tensor-likes are not close!

Mismatched elements: 294912 / 294912 (100.0%)
Greatest absolute difference: nan at index (0, 0) (up to 0.01 allowed)
Greatest relative difference: nan at index (0, 0) (up to 0.01 allowed)
❌ tensor: tensors differ beyond tolerance.
Tensor-likes are not close!

Mismatched elements: 1179648 / 1179648 (100.0%)
Greatest absolute difference: nan at index (0, 0) (up to 0.01 allowed)
Greatest relative difference: nan at index (0, 0) (up to 0.01 allowed)
❌ tensor: tensors differ beyond tolerance.
Tensor-likes are not close!

Mismatched elements: 294912 / 294912 (100.0%)
Greatest absolute difference: nan at index (0, 0) (up to 0.02 allowed)
Greatest relative difference: nan at index (0, 0) (up to 0.01 allowed)
fused_attn
fused_feedforward
---------------------layer- 9 ---------------------

---------------------layer- 9 -count-of-nan---------------------

output_qkv: 884736
query_key_output 1769472
attn_value_output 294912
attn_fc_output 294912
feed_forward_fc1_output 1179648
feed_forward_fc2_output 294912
❌ tensor: tensors differ beyond tolerance.
Tensor-likes are not close!

Mismatched elements: 884736 / 884736 (100.0%)
Greatest absolute difference: nan at index (0, 0, 0, 0) (up to 0.01 allowed)
Greatest relative difference: nan at index (0, 0, 0, 0) (up to 0.01 allowed)
❌ tensor: tensors differ beyond tolerance.
Tensor-likes are not close!

Mismatched elements: 1769472 / 1769472 (100.0%)
Greatest absolute difference: nan at index (0, 0, 0) (up to 0.01 allowed)
Greatest relative difference: nan at index (0, 0, 0) (up to 0.01 allowed)
❌ tensor: tensors differ beyond tolerance.
Tensor-likes are not close!

Mismatched elements: 294912 / 294912 (100.0%)
Greatest absolute difference: nan at index (0, 0) (up to 0.01 allowed)
Greatest relative difference: nan at index (0, 0) (up to 0.01 allowed)
❌ tensor: tensors differ beyond tolerance.
Tensor-likes are not close!

Mismatched elements: 294912 / 294912 (100.0%)
Greatest absolute difference: nan at index (0, 0) (up to 0.01 allowed)
Greatest relative difference: nan at index (0, 0) (up to 0.01 allowed)
❌ tensor: tensors differ beyond tolerance.
Tensor-likes are not close!

Mismatched elements: 1179648 / 1179648 (100.0%)
Greatest absolute difference: nan at index (0, 0) (up to 0.01 allowed)
Greatest relative difference: nan at index (0, 0) (up to 0.01 allowed)
❌ tensor: tensors differ beyond tolerance.
Tensor-likes are not close!

Mismatched elements: 294912 / 294912 (100.0%)
Greatest absolute difference: nan at index (0, 0) (up to 0.02 allowed)
Greatest relative difference: nan at index (0, 0) (up to 0.01 allowed)
fused_attn
fused_feedforward
---------------------layer- 10 ---------------------

---------------------layer- 10 -count-of-nan---------------------

output_qkv: 884736
query_key_output 1769472
attn_value_output 294912
attn_fc_output 294912
feed_forward_fc1_output 1179648
feed_forward_fc2_output 294912
❌ tensor: tensors differ beyond tolerance.
Tensor-likes are not close!

Mismatched elements: 884736 / 884736 (100.0%)
Greatest absolute difference: nan at index (0, 0, 0, 0) (up to 0.01 allowed)
Greatest relative difference: nan at index (0, 0, 0, 0) (up to 0.01 allowed)
❌ tensor: tensors differ beyond tolerance.
Tensor-likes are not close!

Mismatched elements: 1769472 / 1769472 (100.0%)
Greatest absolute difference: nan at index (0, 0, 0) (up to 0.01 allowed)
Greatest relative difference: nan at index (0, 0, 0) (up to 0.01 allowed)
❌ tensor: tensors differ beyond tolerance.
Tensor-likes are not close!

Mismatched elements: 294912 / 294912 (100.0%)
Greatest absolute difference: nan at index (0, 0) (up to 0.01 allowed)
Greatest relative difference: nan at index (0, 0) (up to 0.01 allowed)
❌ tensor: tensors differ beyond tolerance.
Tensor-likes are not close!

Mismatched elements: 294912 / 294912 (100.0%)
Greatest absolute difference: nan at index (0, 0) (up to 0.01 allowed)
Greatest relative difference: nan at index (0, 0) (up to 0.01 allowed)
❌ tensor: tensors differ beyond tolerance.
Tensor-likes are not close!

Mismatched elements: 1179648 / 1179648 (100.0%)
Greatest absolute difference: nan at index (0, 0) (up to 0.01 allowed)
Greatest relative difference: nan at index (0, 0) (up to 0.01 allowed)
❌ tensor: tensors differ beyond tolerance.
Tensor-likes are not close!

Mismatched elements: 294912 / 294912 (100.0%)
Greatest absolute difference: nan at index (0, 0) (up to 0.02 allowed)
Greatest relative difference: nan at index (0, 0) (up to 0.01 allowed)
fused_attn
fused_feedforward
---------------------layer- 11 ---------------------

---------------------layer- 11 -count-of-nan---------------------

output_qkv: 884736
query_key_output 1769472
attn_value_output 294912
attn_fc_output 294912
feed_forward_fc1_output 1179648
feed_forward_fc2_output 294912
❌ tensor: tensors differ beyond tolerance.
Tensor-likes are not close!

Mismatched elements: 884736 / 884736 (100.0%)
Greatest absolute difference: nan at index (0, 0, 0, 0) (up to 0.01 allowed)
Greatest relative difference: nan at index (0, 0, 0, 0) (up to 0.01 allowed)
❌ tensor: tensors differ beyond tolerance.
Tensor-likes are not close!

Mismatched elements: 1769472 / 1769472 (100.0%)
Greatest absolute difference: nan at index (0, 0, 0) (up to 0.01 allowed)
Greatest relative difference: nan at index (0, 0, 0) (up to 0.01 allowed)
❌ tensor: tensors differ beyond tolerance.
Tensor-likes are not close!

Mismatched elements: 294912 / 294912 (100.0%)
Greatest absolute difference: nan at index (0, 0) (up to 0.01 allowed)
Greatest relative difference: nan at index (0, 0) (up to 0.01 allowed)
❌ tensor: tensors differ beyond tolerance.
Tensor-likes are not close!

Mismatched elements: 294912 / 294912 (100.0%)
Greatest absolute difference: nan at index (0, 0) (up to 0.01 allowed)
Greatest relative difference: nan at index (0, 0) (up to 0.01 allowed)
❌ tensor: tensors differ beyond tolerance.
Tensor-likes are not close!

Mismatched elements: 1179648 / 1179648 (100.0%)
Greatest absolute difference: nan at index (0, 0) (up to 0.01 allowed)
Greatest relative difference: nan at index (0, 0) (up to 0.01 allowed)
❌ tensor: tensors differ beyond tolerance.
Tensor-likes are not close!

Mismatched elements: 294912 / 294912 (100.0%)
Greatest absolute difference: nan at index (0, 0) (up to 0.02 allowed)
Greatest relative difference: nan at index (0, 0) (up to 0.01 allowed)
```
