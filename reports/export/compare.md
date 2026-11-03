| host | model | device | size MB | infer ms | total ms | FPS | mAP@0.5 | mAP@0.5:0.95 | ΔmAP@0.5 | match vs .pt |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| kaggle-t4 | finetune.pt | 0 | 18.3 | 8.1 | 10.5 | 95 | 0.6819 | 0.4924 | +0.0% | ref |
| kaggle-t4 | finetune_fp16.engine | 0 | 94.8 | 2.5 | 5.1 | 195 | 0.6794 | 0.4834 | -0.4% | 98.8% |
| kaggle-t4 | finetune_int8.engine | 0 | 48.0 | 2.2 | 4.8 | 209 | 0.6684 | 0.4702 | -2.0% | 95.8% |
| kaggle-t4 | finetune.onnx | cpu | 36.2 | 224.6 | 228.4 | 4 | 0.6795 | 0.4836 | -0.4% | 98.9% |
| kaggle-t4 | finetune_int8.onnx | cpu | 12.0 | 117.6 | 121.8 | 8 | 0.6838 | 0.4783 | +0.3% | 94.2% |
| mac-m5 | finetune.pt | mps | 18.3 | 2.3 | 8.3 | 121 | 0.6819 | 0.4924 | +0.0% | ref |
| mac-m5 | finetune.pt | cpu | 18.3 | 28.2 | 28.9 | 34 | 0.6819 | 0.4924 | +0.0% | ref |
| mac-m5 | finetune.onnx | cpu | 36.2 | 31.8 | 32.9 | 30 | 0.6795 | 0.4836 | -0.4% | 98.9% |
| mac-m5 | finetune_int8.onnx | cpu | 12.0 | 25.1 | 26.2 | 38 | 0.6803 | 0.4838 | -0.2% | 96.6% |
