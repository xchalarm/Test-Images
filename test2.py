from keras.models import load_model
import numpy as np
import pandas as pd
import cv2

FOLDER_PATH = "dataset/Test Images/"
CSV_PATH = "dataset/test.csv"
IMAGE_SIZE_PAIR = (224, 224)

pair_model = load_model("best_model.keras")


def load_image(image_path, size):
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, size)
    img = img / 255.0
    return np.expand_dims(img, axis=0)


def pair_prediction(image1, image2, model):
    predict = model.predict([image1, image2])
    return 2 if predict > 0.5 else 1


dataframe = pd.read_csv(CSV_PATH, delimiter=",", header=0)
predictions = []

for _, row in dataframe.iterrows():
    img1_path = f"{FOLDER_PATH}/{row['Image 1']}"
    img2_path = f"{FOLDER_PATH}/{row['Image 2']}"

    image_1 = load_image(img1_path, IMAGE_SIZE_PAIR)
    image_2 = load_image(img2_path, IMAGE_SIZE_PAIR)

    winner = pair_prediction(image_1, image_2, pair_model)
    predictions.append(winner)


dataframe["Winner"] = predictions
dataframe.to_csv("data_with_predictions.csv", index=False)

headers = ["Image 1", "Image 2", "Predicted Winner"]

print("\nPredictions saved successfully!")
