import json
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.metrics import classification_report

# 1. 加载数据
with open("annotations.json", "r") as f:
    data = json.load(f)

# 2. 数据预处理
X = []
y = []

for item in data:
    keypoints = item["keypoints"][0]
    X.append(np.array(keypoints).flatten())
    y.append(item["label"])

X = np.array(X)
# 标签编码
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# 3. 准备训练和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42
)
print("X_train shape:", X_train)
print("y_train shape:", y_train)
print("X_test shape:", X_test)
print("y_test shape:", y_test)

# 4. 构建模型
num_classes = len(label_encoder.classes_)
input_shape = X_train.shape[1]

model = Sequential()
model.add(Dense(128, activation="relu", input_shape=(input_shape,)))
model.add(Dropout(0.5))
model.add(Dense(64, activation="relu"))
model.add(Dense(num_classes, activation="softmax"))

model.compile(
    optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
)

# 5. 训练模型
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.1)

# 6. 评估模型
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
print("\n测试集准确率:", test_acc)

# 7. 预测和报告
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
print("预测的类别：", y_pred_classes)
print("真实的类别：", y_test)
y_test_labels = label_encoder.inverse_transform(y_test)
y_pred_labels = label_encoder.inverse_transform(y_pred_classes)

print(classification_report(y_test_labels, y_pred_labels))
