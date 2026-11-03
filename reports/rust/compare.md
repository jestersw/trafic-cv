| pipeline | FPS | frame ms p50 | frame ms p95 | detector ms p50 | tracker ms p50 | peak RSS MB | binary MB | line in/out |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| python (ultralytics + onnxruntime) | 21.8 | 43.5 | 46.5 | 38.3 | 0.34 | 687 | - | 23/16 |
| rust (ort + opencv) | 81.7 | 11.0 | 11.7 | 8.4 | 0.01 | 325 | 18.9 | 23/16 |

detections on 300 frames: 99.6% of python boxes matched (IoU>=0.5, same class), mean IoU 0.9825, max coord diff 4.71 px; line counts identical: True
