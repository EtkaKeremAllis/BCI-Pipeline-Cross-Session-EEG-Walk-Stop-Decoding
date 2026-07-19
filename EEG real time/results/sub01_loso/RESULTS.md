# sub-01 ses-01..ses-08 Strict LOSO

Held-out session feature selection, normalization ve model fit islemlerine katilmadi.

Konfigurasyon: 9 kanal, window=0.2s, step=0.05s, context=1.0s, n_features=24, shrinkage=0.2.

| Held-out | N train | N test | Accuracy | Balanced acc. | STOP recall | WALK recall |
|---|---:|---:|---:|---:|---:|---:|
| ses-01 | 8225 | 4892 | 76.39% | 79.76% | 67.46% | 92.06% |
| ses-02 | 8225 | 4892 | 79.44% | 80.70% | 76.09% | 85.30% |
| ses-03 | 8225 | 4892 | 83.20% | 85.04% | 78.31% | 91.78% |
| ses-04 | 8225 | 4892 | 84.32% | 84.94% | 82.67% | 87.22% |
| ses-05 | 8225 | 4892 | 84.48% | 82.40% | 90.02% | 74.77% |
| ses-06 | 8225 | 4892 | 81.38% | 83.25% | 76.41% | 90.09% |
| ses-07 | 8225 | 4892 | 79.31% | 77.47% | 84.21% | 70.72% |
| ses-08 | 8225 | 4892 | 80.11% | 80.63% | 78.72% | 82.55% |

## Session-macro

- accuracy: 81.08% (session SD 2.81 puan)
- balanced_accuracy: 81.77% (session SD 2.63 puan)
- stop_recall: 79.24% (session SD 6.66 puan)
- walk_recall: 84.31% (session SD 7.91 puan)

Toplam sure: 181.96 saniye.
