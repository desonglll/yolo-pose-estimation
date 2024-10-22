import json
import numpy as np
from keras._tf_keras.keras.models import load_model
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from keras._tf_keras.keras.models import Sequential
from keras._tf_keras.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import pickle


def train_model(dataset: str, saved_model):
    print("start training model")
    with open(dataset, "r") as f:
        data = json.load(f)

    X = []
    y = []

    for item in data:
        keypoints = item["keypoints"][0]
        X.append(np.array(keypoints).flatten())
        y.append(item["label"])
    X = np.array(X)

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    print(y_encoded)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42
    )
    print("X_train shape:", X_train.shape)
    print("y_train shape:", y_train.shape)
    print("X_test shape:", X_test.shape)
    print("y_test shape:", y_test.shape)

    num_classes = len(label_encoder.classes_)
    input_shape = X_train.shape[1]

    X_train = X_train.reshape(-1, 17, 2, 1)
    X_test = X_test.reshape(-1, 17, 2, 1)

    num_classes = len(label_encoder.classes_)
    input_shape = (17, 2, 1)  # 高度、宽度、通道数

    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 2), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 1)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, kernel_size=(3, 1), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 1)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    # 编译模型
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.1)
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
    print("\n测试集准确率:", test_acc)

    # y_pred = model.predict(X_test)
    # y_pred_classes = np.argmax(y_pred, axis=1)
    # print("预测的类别：", y_pred_classes)
    # print("真实的类别：", y_test)
    # y_test_labels = label_encoder.inverse_transform(y_test)
    # y_pred_labels = label_encoder.inverse_transform(y_pred_classes)
    #
    # print(classification_report(y_test_labels, y_pred_labels))

    model.save(saved_model)
    print(f"model saved to {saved_model}")
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    print(y_encoded)

    # 保存 LabelEncoder
    with open("../label_encoder.pkl", "wb") as le_file:
        pickle.dump(label_encoder, le_file)

    print("finished training model")


def predict(model_name, data_to_predict, result_path):
    # 加载标签编码器
    with open("../label_encoder.pkl", "rb") as le_file:
        label_encoder = pickle.load(le_file)

    # 加载训练好的模型
    model = load_model(model_name)

    # 定义一个函数来预处理关键点数据
    def preprocess_keypoints(keypoints):
        """
        预处理关键点数据，使其符合模型输入要求。

        参数:
            keypoints (list): 关键点列表，每个关键点为 [x, y]。

        返回:
            numpy.ndarray: 预处理后的关键点数据，形状为 (17, 2, 1)。
        """
        keypoints_array = np.array(keypoints).flatten()  # 将关键点展平成一维数组
        keypoints_array = keypoints_array.reshape(-1, 17, 2, 1)  # 重新调整形状以匹配模型输入
        return keypoints_array

    # 加载 annotations.json 数据
    with open(data_to_predict, "r") as f:
        annotations = json.load(f)

    # 提取关键点并进行预处理
    X_new = []
    image_paths = []  # 用于存储图像路径（可选）
    for item in annotations:
        keypoints = item["keypoints"][0]  # 假设每个条目只有一个关键点集
        X_new.append(preprocess_keypoints(keypoints))
        image_paths.append(item["image_path"])

    X_new = np.vstack(X_new)  # 合并所有样本，形状为 (样本数, 17, 2, 1)

    # 进行预测
    y_pred = model.predict(X_new)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_pred_labels = label_encoder.inverse_transform(y_pred_classes)

    # 打印预测结果
    for img_path, pred_label in zip(image_paths, y_pred_labels):
        print(f"Image: {img_path} --> Predicted Label: {pred_label}")

    # 可选：将预测结果保存到一个新的 JSON 文件中
    predictions = []
    for img_path, pred_label in zip(image_paths, y_pred_labels):
        predictions.append({
            "image_path": img_path,
            "predicted_label": pred_label
        })

    with open(result_path, "w") as f:
        json.dump(predictions, f, indent=4)

    print(f"预测完成，结果已保存到 {result_path}")
