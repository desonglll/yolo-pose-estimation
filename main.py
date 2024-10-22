from ype import model, preprocess


def main():
    data_path = "annotations.json"
    dataset_path = "generated_data.json"
    result_path = "predictions.json"
    model_name = "pose.keras"

    image_paths = ["./g1.png", "./g2.png"]
    preprocess.process(image_paths, data_path)

    model.train_model(dataset_path, model_name)
    model.predict(model_name, data_path, result_path)
    pass


if __name__ == "__main__":
    main()
    pass
