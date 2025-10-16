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
✅ output_qkv: tensors are close within tolerance.
✅ query_key_output: tensors are close within tolerance.
✅ attn_value_output: tensors are close within tolerance.
✅ attn_fc_output: tensors are close within tolerance.
❌ feed_forward_fc1_output: tensors differ beyond tolerance.
Tensor-likes are not close!

Mismatched elements: 1 / 1179648 (0.0%)
Greatest absolute difference: 0.015625 at index (355, 2180) (up to 0.01 allowed)
Greatest relative difference: 0.032928466796875 at index (355, 2180) (up to 0.01 allowed)
✅ feed_forward_fc2_output: tensors are close within tolerance.
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
❌ output_qkv: tensors differ beyond tolerance.
Tensor-likes are not close!

Mismatched elements: 15 / 884736 (0.0%)
Greatest absolute difference: 0.01611328125 at index (1, 8, 231, 45) (up to 0.01 allowed)
Greatest relative difference: 55.9375 at index (1, 2, 46, 34) (up to 0.01 allowed)
✅ query_key_output: tensors are close within tolerance.
✅ attn_value_output: tensors are close within tolerance.
❌ attn_fc_output: tensors differ beyond tolerance.
Tensor-likes are not close!

Mismatched elements: 2 / 294912 (0.0%)
Greatest absolute difference: 0.0135498046875 at index (178, 225) (up to 0.01 allowed)
Greatest relative difference: 0.11199951171875 at index (206, 225) (up to 0.01 allowed)
✅ feed_forward_fc1_output: tensors are close within tolerance.
✅ feed_forward_fc2_output: tensors are close within tolerance.
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
❌ output_qkv: tensors differ beyond tolerance.
Tensor-likes are not close!

Mismatched elements: 10 / 884736 (0.0%)
Greatest absolute difference: 0.0133056640625 at index (0, 0, 190, 20) (up to 0.01 allowed)
Greatest relative difference: 5.125 at index (1, 1, 169, 12) (up to 0.01 allowed)
✅ query_key_output: tensors are close within tolerance.
✅ attn_value_output: tensors are close within tolerance.
✅ attn_fc_output: tensors are close within tolerance.
❌ feed_forward_fc1_output: tensors differ beyond tolerance.
Tensor-likes are not close!

Mismatched elements: 7 / 1179648 (0.0%)
Greatest absolute difference: 0.0322265625 at index (103, 1072) (up to 0.01 allowed)
Greatest relative difference: 0.0784912109375 at index (94, 1072) (up to 0.01 allowed)
❌ feed_forward_fc2_output: tensors differ beyond tolerance.
Tensor-likes are not close!

Mismatched elements: 3 / 294912 (0.0%)
Greatest absolute difference: 0.08203125 at index (103, 381) (up to 0.02 allowed)
Greatest relative difference: 0.0311279296875 at index (78, 381) (up to 0.01 allowed)
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
❌ output_qkv: tensors differ beyond tolerance.
Tensor-likes are not close!

Mismatched elements: 148 / 884736 (0.0%)
Greatest absolute difference: 0.046875 at index (1, 8, 102, 9) (up to 0.01 allowed)
Greatest relative difference: 63.5625 at index (0, 4, 103, 52) (up to 0.01 allowed)
✅ query_key_output: tensors are close within tolerance.
✅ attn_value_output: tensors are close within tolerance.
❌ attn_fc_output: tensors differ beyond tolerance.
Tensor-likes are not close!

Mismatched elements: 9 / 294912 (0.0%)
Greatest absolute difference: 0.0703125 at index (103, 381) (up to 0.01 allowed)
Greatest relative difference: 0.9697265625 at index (187, 521) (up to 0.01 allowed)
❌ feed_forward_fc1_output: tensors differ beyond tolerance.
Tensor-likes are not close!

Mismatched elements: 11 / 1179648 (0.0%)
Greatest absolute difference: 0.0220947265625 at index (108, 1693) (up to 0.01 allowed)
Greatest relative difference: 1.2958984375 at index (146, 2831) (up to 0.01 allowed)
❌ feed_forward_fc2_output: tensors differ beyond tolerance.
Tensor-likes are not close!

Mismatched elements: 2 / 294912 (0.0%)
Greatest absolute difference: 0.140625 at index (108, 381) (up to 0.02 allowed)
Greatest relative difference: 0.036712646484375 at index (103, 381) (up to 0.01 allowed)
fused_attn
fused_feedforward
---------------------layer- 4 ---------------------

---------------------layer- 4 -count-of-nan---------------------

output_qkv: 0
query_key_output 223
attn_value_output 11776
attn_fc_output 109056
feed_forward_fc1_output 436224
feed_forward_fc2_output 109056
❌ output_qkv: tensors differ beyond tolerance.
Tensor-likes are not close!

Mismatched elements: 861 / 884736 (0.1%)
Greatest absolute difference: 0.078125 at index (1, 8, 99, 48) (up to 0.01 allowed)
Greatest relative difference: 9.859375 at index (1, 3, 119, 3) (up to 0.01 allowed)
❌ query_key_output: tensors differ beyond tolerance.
Tensor-likes are not close!

Mismatched elements: 536 / 1769472 (0.0%)
Greatest absolute difference: nan at index (0, 0, 121) (up to 0.01 allowed)
Greatest relative difference: nan at index (0, 0, 121) (up to 0.01 allowed)
❌ attn_value_output: tensors differ beyond tolerance.
Tensor-likes are not close!

Mismatched elements: 11776 / 294912 (4.0%)
Greatest absolute difference: nan at index (0, 0) (up to 0.01 allowed)
Greatest relative difference: nan at index (0, 0) (up to 0.01 allowed)
❌ attn_fc_output: tensors differ beyond tolerance.
Tensor-likes are not close!

Mismatched elements: 109060 / 294912 (37.0%)
Greatest absolute difference: nan at index (0, 0) (up to 0.01 allowed)
Greatest relative difference: nan at index (0, 0) (up to 0.01 allowed)
❌ feed_forward_fc1_output: tensors differ beyond tolerance.
Tensor-likes are not close!

Mismatched elements: 436309 / 1179648 (37.0%)
Greatest absolute difference: nan at index (0, 0) (up to 0.01 allowed)
Greatest relative difference: nan at index (0, 0) (up to 0.01 allowed)
❌ feed_forward_fc2_output: tensors differ beyond tolerance.
Tensor-likes are not close!

Mismatched elements: 109100 / 294912 (37.0%)
Greatest absolute difference: nan at index (0, 0) (up to 0.02 allowed)
Greatest relative difference: nan at index (0, 0) (up to 0.01 allowed)
fused_attn
fused_feedforward
---------------------layer- 5 ---------------------

---------------------layer- 5 -count-of-nan---------------------

output_qkv: 327168
query_key_output 1769472
attn_value_output 294912
attn_fc_output 294912
feed_forward_fc1_output 1179648
feed_forward_fc2_output 294912
❌ output_qkv: tensors differ beyond tolerance.
Tensor-likes are not close!

Mismatched elements: 330757 / 884736 (37.4%)
Greatest absolute difference: nan at index (0, 0, 0, 0) (up to 0.01 allowed)
Greatest relative difference: nan at index (0, 0, 0, 0) (up to 0.01 allowed)
❌ query_key_output: tensors differ beyond tolerance.
Tensor-likes are not close!

Mismatched elements: 1769472 / 1769472 (100.0%)
Greatest absolute difference: nan at index (0, 0, 0) (up to 0.01 allowed)
Greatest relative difference: nan at index (0, 0, 0) (up to 0.01 allowed)
❌ attn_value_output: tensors differ beyond tolerance.
Tensor-likes are not close!

Mismatched elements: 294912 / 294912 (100.0%)
Greatest absolute difference: nan at index (0, 0) (up to 0.01 allowed)
Greatest relative difference: nan at index (0, 0) (up to 0.01 allowed)
❌ attn_fc_output: tensors differ beyond tolerance.
Tensor-likes are not close!

Mismatched elements: 294912 / 294912 (100.0%)
Greatest absolute difference: nan at index (0, 0) (up to 0.01 allowed)
Greatest relative difference: nan at index (0, 0) (up to 0.01 allowed)
❌ feed_forward_fc1_output: tensors differ beyond tolerance.
Tensor-likes are not close!

Mismatched elements: 1179648 / 1179648 (100.0%)
Greatest absolute difference: nan at index (0, 0) (up to 0.01 allowed)
Greatest relative difference: nan at index (0, 0) (up to 0.01 allowed)
❌ feed_forward_fc2_output: tensors differ beyond tolerance.
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
❌ output_qkv: tensors differ beyond tolerance.
Tensor-likes are not close!

Mismatched elements: 884736 / 884736 (100.0%)
Greatest absolute difference: nan at index (0, 0, 0, 0) (up to 0.01 allowed)
Greatest relative difference: nan at index (0, 0, 0, 0) (up to 0.01 allowed)
❌ query_key_output: tensors differ beyond tolerance.
Tensor-likes are not close!

Mismatched elements: 1769472 / 1769472 (100.0%)
Greatest absolute difference: nan at index (0, 0, 0) (up to 0.01 allowed)
Greatest relative difference: nan at index (0, 0, 0) (up to 0.01 allowed)
❌ attn_value_output: tensors differ beyond tolerance.
Tensor-likes are not close!

Mismatched elements: 294912 / 294912 (100.0%)
Greatest absolute difference: nan at index (0, 0) (up to 0.01 allowed)
Greatest relative difference: nan at index (0, 0) (up to 0.01 allowed)
❌ attn_fc_output: tensors differ beyond tolerance.
Tensor-likes are not close!

Mismatched elements: 294912 / 294912 (100.0%)
Greatest absolute difference: nan at index (0, 0) (up to 0.01 allowed)
Greatest relative difference: nan at index (0, 0) (up to 0.01 allowed)
❌ feed_forward_fc1_output: tensors differ beyond tolerance.
Tensor-likes are not close!

Mismatched elements: 1179648 / 1179648 (100.0%)
Greatest absolute difference: nan at index (0, 0) (up to 0.01 allowed)
Greatest relative difference: nan at index (0, 0) (up to 0.01 allowed)
❌ feed_forward_fc2_output: tensors differ beyond tolerance.
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
❌ output_qkv: tensors differ beyond tolerance.
Tensor-likes are not close!

Mismatched elements: 884736 / 884736 (100.0%)
Greatest absolute difference: nan at index (0, 0, 0, 0) (up to 0.01 allowed)
Greatest relative difference: nan at index (0, 0, 0, 0) (up to 0.01 allowed)
❌ query_key_output: tensors differ beyond tolerance.
Tensor-likes are not close!

Mismatched elements: 1769472 / 1769472 (100.0%)
Greatest absolute difference: nan at index (0, 0, 0) (up to 0.01 allowed)
Greatest relative difference: nan at index (0, 0, 0) (up to 0.01 allowed)
❌ attn_value_output: tensors differ beyond tolerance.
Tensor-likes are not close!

Mismatched elements: 294912 / 294912 (100.0%)
Greatest absolute difference: nan at index (0, 0) (up to 0.01 allowed)
Greatest relative difference: nan at index (0, 0) (up to 0.01 allowed)
❌ attn_fc_output: tensors differ beyond tolerance.
Tensor-likes are not close!

Mismatched elements: 294912 / 294912 (100.0%)
Greatest absolute difference: nan at index (0, 0) (up to 0.01 allowed)
Greatest relative difference: nan at index (0, 0) (up to 0.01 allowed)
❌ feed_forward_fc1_output: tensors differ beyond tolerance.
Tensor-likes are not close!

Mismatched elements: 1179648 / 1179648 (100.0%)
Greatest absolute difference: nan at index (0, 0) (up to 0.01 allowed)
Greatest relative difference: nan at index (0, 0) (up to 0.01 allowed)
❌ feed_forward_fc2_output: tensors differ beyond tolerance.
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
❌ output_qkv: tensors differ beyond tolerance.
Tensor-likes are not close!

Mismatched elements: 884736 / 884736 (100.0%)
Greatest absolute difference: nan at index (0, 0, 0, 0) (up to 0.01 allowed)
Greatest relative difference: nan at index (0, 0, 0, 0) (up to 0.01 allowed)
❌ query_key_output: tensors differ beyond tolerance.
Tensor-likes are not close!

Mismatched elements: 1769472 / 1769472 (100.0%)
Greatest absolute difference: nan at index (0, 0, 0) (up to 0.01 allowed)
Greatest relative difference: nan at index (0, 0, 0) (up to 0.01 allowed)
❌ attn_value_output: tensors differ beyond tolerance.
Tensor-likes are not close!

Mismatched elements: 294912 / 294912 (100.0%)
Greatest absolute difference: nan at index (0, 0) (up to 0.01 allowed)
Greatest relative difference: nan at index (0, 0) (up to 0.01 allowed)
❌ attn_fc_output: tensors differ beyond tolerance.
Tensor-likes are not close!

Mismatched elements: 294912 / 294912 (100.0%)
Greatest absolute difference: nan at index (0, 0) (up to 0.01 allowed)
Greatest relative difference: nan at index (0, 0) (up to 0.01 allowed)
❌ feed_forward_fc1_output: tensors differ beyond tolerance.
Tensor-likes are not close!

Mismatched elements: 1179648 / 1179648 (100.0%)
Greatest absolute difference: nan at index (0, 0) (up to 0.01 allowed)
Greatest relative difference: nan at index (0, 0) (up to 0.01 allowed)
❌ feed_forward_fc2_output: tensors differ beyond tolerance.
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
❌ output_qkv: tensors differ beyond tolerance.
Tensor-likes are not close!

Mismatched elements: 884736 / 884736 (100.0%)
Greatest absolute difference: nan at index (0, 0, 0, 0) (up to 0.01 allowed)
Greatest relative difference: nan at index (0, 0, 0, 0) (up to 0.01 allowed)
❌ query_key_output: tensors differ beyond tolerance.
Tensor-likes are not close!

Mismatched elements: 1769472 / 1769472 (100.0%)
Greatest absolute difference: nan at index (0, 0, 0) (up to 0.01 allowed)
Greatest relative difference: nan at index (0, 0, 0) (up to 0.01 allowed)
❌ attn_value_output: tensors differ beyond tolerance.
Tensor-likes are not close!

Mismatched elements: 294912 / 294912 (100.0%)
Greatest absolute difference: nan at index (0, 0) (up to 0.01 allowed)
Greatest relative difference: nan at index (0, 0) (up to 0.01 allowed)
❌ attn_fc_output: tensors differ beyond tolerance.
Tensor-likes are not close!

Mismatched elements: 294912 / 294912 (100.0%)
Greatest absolute difference: nan at index (0, 0) (up to 0.01 allowed)
Greatest relative difference: nan at index (0, 0) (up to 0.01 allowed)
❌ feed_forward_fc1_output: tensors differ beyond tolerance.
Tensor-likes are not close!

Mismatched elements: 1179648 / 1179648 (100.0%)
Greatest absolute difference: nan at index (0, 0) (up to 0.01 allowed)
Greatest relative difference: nan at index (0, 0) (up to 0.01 allowed)
❌ feed_forward_fc2_output: tensors differ beyond tolerance.
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
❌ output_qkv: tensors differ beyond tolerance.
Tensor-likes are not close!

Mismatched elements: 884736 / 884736 (100.0%)
Greatest absolute difference: nan at index (0, 0, 0, 0) (up to 0.01 allowed)
Greatest relative difference: nan at index (0, 0, 0, 0) (up to 0.01 allowed)
❌ query_key_output: tensors differ beyond tolerance.
Tensor-likes are not close!

Mismatched elements: 1769472 / 1769472 (100.0%)
Greatest absolute difference: nan at index (0, 0, 0) (up to 0.01 allowed)
Greatest relative difference: nan at index (0, 0, 0) (up to 0.01 allowed)
❌ attn_value_output: tensors differ beyond tolerance.
Tensor-likes are not close!

Mismatched elements: 294912 / 294912 (100.0%)
Greatest absolute difference: nan at index (0, 0) (up to 0.01 allowed)
Greatest relative difference: nan at index (0, 0) (up to 0.01 allowed)
❌ attn_fc_output: tensors differ beyond tolerance.
Tensor-likes are not close!

Mismatched elements: 294912 / 294912 (100.0%)
Greatest absolute difference: nan at index (0, 0) (up to 0.01 allowed)
Greatest relative difference: nan at index (0, 0) (up to 0.01 allowed)
❌ feed_forward_fc1_output: tensors differ beyond tolerance.
Tensor-likes are not close!

Mismatched elements: 1179648 / 1179648 (100.0%)
Greatest absolute difference: nan at index (0, 0) (up to 0.01 allowed)
Greatest relative difference: nan at index (0, 0) (up to 0.01 allowed)
❌ feed_forward_fc2_output: tensors differ beyond tolerance.
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
❌ output_qkv: tensors differ beyond tolerance.
Tensor-likes are not close!

Mismatched elements: 884736 / 884736 (100.0%)
Greatest absolute difference: nan at index (0, 0, 0, 0) (up to 0.01 allowed)
Greatest relative difference: nan at index (0, 0, 0, 0) (up to 0.01 allowed)
❌ query_key_output: tensors differ beyond tolerance.
Tensor-likes are not close!

Mismatched elements: 1769472 / 1769472 (100.0%)
Greatest absolute difference: nan at index (0, 0, 0) (up to 0.01 allowed)
Greatest relative difference: nan at index (0, 0, 0) (up to 0.01 allowed)
❌ attn_value_output: tensors differ beyond tolerance.
Tensor-likes are not close!

Mismatched elements: 294912 / 294912 (100.0%)
Greatest absolute difference: nan at index (0, 0) (up to 0.01 allowed)
Greatest relative difference: nan at index (0, 0) (up to 0.01 allowed)
❌ attn_fc_output: tensors differ beyond tolerance.
Tensor-likes are not close!

Mismatched elements: 294912 / 294912 (100.0%)
Greatest absolute difference: nan at index (0, 0) (up to 0.01 allowed)
Greatest relative difference: nan at index (0, 0) (up to 0.01 allowed)
❌ feed_forward_fc1_output: tensors differ beyond tolerance.
Tensor-likes are not close!

Mismatched elements: 1179648 / 1179648 (100.0%)
Greatest absolute difference: nan at index (0, 0) (up to 0.01 allowed)
Greatest relative difference: nan at index (0, 0) (up to 0.01 allowed)
❌ feed_forward_fc2_output: tensors differ beyond tolerance.
Tensor-likes are not close!

Mismatched elements: 294912 / 294912 (100.0%)
Greatest absolute difference: nan at index (0, 0) (up to 0.02 allowed)
Greatest relative difference: nan at index (0, 0) (up to 0.01 allowed)
```
