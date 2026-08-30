# Overall Confusion Matrix Report

## Default Threshold (0.5)

|                 | Predicted Not Toxic | Predicted Toxic |
|-----------------|--------------------:|--------------:|
| True Not Toxic | 166,067 | 7,066 |
| True Toxic     | 11,114 | 29,701 |

- Accuracy: 0.9150
- Precision: 0.8078
- Recall: 0.7277
- F1 Score: 0.7657

## Optimized Thresholds

|                 | Predicted Not Toxic | Predicted Toxic |
|-----------------|--------------------:|--------------:|
| True Not Toxic | 161,131 | 12,002 |
| True Toxic     | 8,237 | 32,578 |

- Accuracy: 0.9054
- Precision: 0.7308
- Recall: 0.7982
- F1 Score: 0.7630

## Improvement from Optimized Thresholds

- Accuracy: -0.0096
- Precision: -0.0770
- Recall: 0.0705
- F1 Score: -0.0027
