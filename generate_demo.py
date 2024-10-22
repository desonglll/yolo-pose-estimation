import random
import json

# Define the list of actions
actions = ["jump", "run", "walk", "stand"]


def generate_keypoints_for_action(action):
    keypoints = []
    for i in range(17):  # 17 keypoints
        if action == "jump":
            if i == 4:
                # Keypoint 4 is always [0.0, 0.0] as per the provided data
                x, y = 0.0, 0.0
            elif i < 11:
                # Upper body keypoints
                # x range based on provided data: approximately [0.07, 0.74]
                x = random.uniform(0.07, 0.75)
                # y range based on provided data: approximately [0.2, 0.86]
                y = random.uniform(0.2, 0.86)
            else:
                # Lower body keypoints
                # x range remains similar to upper body
                x = random.uniform(0.07, 0.75)
                # y range remains similar to upper body
                y = random.uniform(0.2, 0.86)
        elif action == "run":
            # Running action keypoints generation logic
            x = random.uniform(0.3, 0.7)
            y = random.uniform(0.3, 1.0)
        elif action == "walk":
            # Walking action keypoints generation logic
            x = random.uniform(0.35, 0.65)
            y = random.uniform(0.3, 1.0)
        elif action == "stand":
            # Standing action keypoints generation logic
            x = random.uniform(0.45, 0.55)
            y = random.uniform(0.3, 1.0)
        else:
            # Other actions: general keypoint generation
            x = random.uniform(0.0, 1.0)
            y = random.uniform(0.0, 1.0)

        keypoints.append([x, y])
    return [keypoints]  # Outer list represents the batch


def main():
    data = []
    num_samples_per_action = 10000  # Number of samples per action

    for action in actions:
        for i in range(num_samples_per_action):
            keypoints = generate_keypoints_for_action(action)
            sample = {
                "image_path": f"./{action}_{i + 1}.png",
                "keypoints": keypoints,
                "label": action,
            }
            data.append(sample)

    # Save the generated data to a JSON file
    with open("generated_data.json", "w") as f:
        json.dump(data, f, indent=4)


if __name__ == "__main__":
    main()
