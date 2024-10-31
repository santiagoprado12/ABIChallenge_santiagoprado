
# Validation Report

## Confusion Matrix

![Confusion Matrix](./confusion_matrix_20241030_234512.png)

|                |   Predicted Dead |   Predicted Survive |
|:---------------|-----------------:|--------------------:|
| Actual Dead    |              514 |                  35 |
| Actual Survive |               89 |                 253 |

## Classification Report

```bash
              precision    recall  f1-score   support

           0       0.85      0.94      0.89       549
           1       0.88      0.74      0.80       342

    accuracy                           0.86       891
   macro avg       0.87      0.84      0.85       891
weighted avg       0.86      0.86      0.86       891

```
    