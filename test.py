import numpy as np
from collections import Counter

def adaptive_weighted_knn(X_train, y_train, X_test, k):
    y_pred = []
    for test_point in X_test:
        distances = []
        for train_point in X_train:
            distance = np.sqrt(np.sum((test_point - train_point) ** 2))  # 計算歐氏距離
            distances.append(distance)
        distances = np.array(distances)
        sorted_indices = np.argsort(distances)
        k_nearest_distances = distances[sorted_indices[:k]]
        k_nearest_labels = y_train[sorted_indices[:k]]

        # 計算每個鄰居的權重
        weights = 1 / (k_nearest_distances + 1e-10)  # 避免除以零

        # 計算加權的類別
        weighted_labels = []
        for label, weight in zip(k_nearest_labels, weights):
            weighted_labels.extend([label] * int(weight))  # 將權重分配給相應的類別

        # 如果加權的類別列表不為空，則進行預測
        if weighted_labels:
            # 使用Counter計算每個類別的加權數量
            weighted_counter = Counter(weighted_labels)

            # 選擇加權最多的類別作為預測
            predicted_label = weighted_counter.most_common(1)[0][0]
            y_pred.append(predicted_label)
        else:
            # 如果加權的類別列表為空，則將預測標籤設置為 None 或其他值
            y_pred.append(None)

    return np.array(y_pred)

# 示例數據
X_train = np.array([[1, 2],
                    [2, 3],
                    [3, 4],
                    [4, 5]])
y_train = np.array([0, 0, 1, 1])
X_test = np.array([[2, 4]])

# 調用自適應加權KNN函數
k = 3
y_pred = adaptive_weighted_knn(X_train, y_train, X_test, k)
print("Predicted Label:", y_pred)
