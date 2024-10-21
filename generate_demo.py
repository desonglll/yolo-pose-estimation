import random
import json

actions = ["jump", "run", "walk", "stand"]


def generate_keypoints_for_action(action):
    keypoints = []
    for i in range(17):  # 17个关键点
        if action == "jump":
            # 跳跃动作的关键点生成逻辑
            x = random.uniform(0.4, 0.6)  # 中心位置
            y = (
                random.uniform(0.2, 0.5) if i < 11 else random.uniform(0.5, 1.0)
            )  # 上半身和下半身
        elif action == "run":
            # 跑步动作的关键点生成逻辑
            x = random.uniform(0.3, 0.7)
            y = random.uniform(0.3, 1.0)
        elif action == "walk":
            # 行走动作的关键点生成逻辑
            x = random.uniform(0.35, 0.65)
            y = random.uniform(0.3, 1.0)
        elif action == "stand":
            # 站立动作的关键点生成逻辑
            x = random.uniform(0.45, 0.55)
            y = random.uniform(0.3, 1.0)
        else:
            # 其他动作
            x = random.uniform(0.0, 1.0)
            y = random.uniform(0.0, 1.0)
        keypoints.append([x, y])
    return [keypoints]  # 外层列表表示批次


def main():
    data = []
    num_samples_per_action = 10000  # 每个动作生成100个样本

    for action in actions:
        for i in range(num_samples_per_action):
            keypoints = generate_keypoints_for_action(action)
            sample = {
                "image_path": f"./{action}_{i+1}.png",
                "keypoints": keypoints,
                "label": action,
            }
            data.append(sample)

    # 将数据保存为 JSON 文件
    with open("generated_data.json", "w") as f:
        json.dump(data, f, indent=4)


if __name__ == "__main__":
    main()
