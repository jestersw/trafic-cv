### Split summary (split is by video)

| split | sequences | frames | exported_frames | tracks | boxes |
| --- | --- | --- | --- | --- | --- |
| train | 51 | 70,751 | 14,153 | 4,841 | 104,998 |
| val | 9 | 13,040 | 1,305 | 1,111 | 6,984 |
| test | 40 | 56,340 | 5,640 | 2,337 | 63,757 |

### Boxes per class

| class | train | val | test | total | share |
| --- | --- | --- | --- | --- | --- |
| bus | 5,699 | 472 | 6,670 | 12,841 | 7.3% |
| car | 89,050 | 5,610 | 51,835 | 146,495 | 83.4% |
| others | 584 | 78 | 1,659 | 2,321 | 1.3% |
| van | 9,665 | 824 | 3,593 | 14,082 | 8.0% |

### Object size per class

| class | small (<32²) | medium | large (>96²) |
| --- | --- | --- | --- |
| bus | 50 | 2,512 | 10,279 |
| car | 15,217 | 102,894 | 28,384 |
| others | 15 | 969 | 1,337 |
| van | 1,111 | 8,491 | 4,480 |

### Sequences per weather condition

| weather | train | val | test |
| --- | --- | --- | --- |
| cloudy | 16 | 3 | 11 |
| night | 14 | 2 | 12 |
| rainy | 8 | 2 | 9 |
| sunny | 13 | 2 | 8 |
