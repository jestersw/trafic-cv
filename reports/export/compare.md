| host | model | device | size MB | infer ms | total ms | FPS | mAP@0.5 | mAP@0.5:0.95 | ΔmAP@0.5 | match vs .pt |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| mac-m5 | finetune.pt | mps | 18.3 | 2.7 | 8.4 | 119 | 0.6819 | 0.4924 | +0.0% | ref |
| mac-m5 | finetune.pt | cpu | 18.3 | 33.3 | 34.2 | 29 | 0.6819 | 0.4924 | +0.0% | ref |
| mac-m5 | finetune.onnx | cpu | 36.2 | 31.6 | 33.5 | 30 | 0.6795 | 0.4836 | -0.4% | 98.9% |
| mac-m5 | finetune_int8.onnx | cpu | 12.0 | 32.6 | 34.6 | 29 | 0.6803 | 0.4838 | -0.2% | 96.6% |
