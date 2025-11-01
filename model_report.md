# 模型評估報告

## SVM
最佳參數: {'C': 10, 'kernel': 'rbf'}
交叉驗證 F1 分數: 0.9237

### 評估指標
- ham: Precision: 0.983, Recall: 0.998, F1: 0.990
- spam: Precision: 0.985, Recall: 0.886, F1: 0.933
- Accuracy: 0.983
### 混淆矩陣
[[963   2]
 [ 17 132]]

## 邏輯回歸
最佳參數: {'C': 10, 'penalty': 'l2', 'solver': 'lbfgs'}
交叉驗證 F1 分數: 0.9203

### 評估指標
- ham: Precision: 0.982, Recall: 0.998, F1: 0.990
- spam: Precision: 0.985, Recall: 0.879, F1: 0.929
- Accuracy: 0.982
### 混淆矩陣
[[963   2]
 [ 18 131]]
