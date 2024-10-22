import cv2
from ultralytics import YOLO

annotations = []


def process(image_paths, generated_data):
    print("start process")
    for image_path in image_paths:
        process_single(image_path)
    save_data(generated_data)
    print("finished preprocess")


def process_single(image_path):
    image = cv2.imread(image_path)
    model = YOLO("yolo11n-pose")
    results = model(source=image, show=True, conf=0.3, save=True)
    # 遍历每一帧的结果
    for idx, result in enumerate(results):
        print(f"处理第 {idx + 1} 帧")

        # 检查是否检测到关键点
        if result.keypoints is not None:
            keypoints = result.keypoints[0]

            # 包含 x, y 坐标和置信度
            keypoints_array = keypoints.numpy()
            print(f"第 1 个人的xyn关键点：")
            # print(keypoints_array)

            # 在这里可以对关键点数据进行处理，例如动作识别
            # print("======================================")
            print("======================================")
            print(keypoints_array.xyn)
            cv2.imshow("Annotate", image)
            cv2.waitKey(1)
            label = input(f"为图像 {image_path} 输入动作标签：")
            annotations.append(
                {
                    "image_path": image_path,
                    "keypoints": keypoints_array.xyn.tolist(),
                    "label": label,
                }
            )

        else:
            print("未检测到关键点")
    cv2.destroyAllWindows()


def save_data(saved_file):
    import json

    with open(saved_file, "w") as f:
        json.dump(annotations, f)
