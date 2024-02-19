import uvicorn
import tensorflow as tf
from fastapi import FastAPI, File, UploadFile
from io import BytesIO
from PIL import Image
import numpy as np
import cv2
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, Flatten, Dense
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator

app = FastAPI()

# Load colorization models and points
prototxt_path = 'C:/Users/mange/Desktop/B& W image/model/colorization_deploy_v2.prototxt'
caffemodel_path = 'C:/Users/mange/Desktop/B& W image/model/colorization_release_v2.caffemodel'
pts_path = 'C:/Users/mange/Desktop/B& W image/model/pts_in_hull.npy'

net = cv2.dnn.readNetFromCaffe(prototxt_path, caffemodel_path)
pts = np.load(pts_path)

class8 = net.getLayerId("class8_ab")
conv8 = net.getLayerId("conv8_313_rh")
pts = pts.transpose().reshape(2, 313, 1, 1)

net.getLayer(class8).blobs = [pts.astype("float32")]
net.getLayer(conv8).blobs = [np.full([1, 313, 1, 1], 2.606, dtype='float32')]

@app.get('/')
def index():
    return {'Deployment': 'Prediction of Images Between the dog and cat'}

@app.post("/predict")
async def predict1(file: UploadFile = File(...)):
    image = await file.read()
    image = Image.open(BytesIO(image))
    image = image.convert('RGB')
    # Resize the image to match the target size used during training
    image = image.resize((100, 100))

    # Convert image to grayscale
    gray_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    gray_image = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2RGB)

    # Colorization
    scaled = gray_image.astype("float32") / 255.0
    lab = cv2.cvtColor(scaled, cv2.COLOR_RGB2LAB)

    resized = cv2.resize(lab, (224, 224))
    L = cv2.split(resized)[0]
    L -= 50

    net.setInput(cv2.dnn.blobFromImage(L))
    ab = net.forward()[0, :, :, :].transpose((1, 2, 0))

    ab = cv2.resize(ab, (gray_image.shape[1], gray_image.shape[0]))

    L = cv2.split(lab)[0]
    colorized = np.concatenate((L[:, :, np.newaxis], ab), axis=2)

    colorized = cv2.cvtColor(colorized, cv2.COLOR_LAB2RGB)
    colorized = np.clip(colorized, 0, 1)
    colorized = (255 * colorized).astype("uint8")

    # Concatenate original grayscale image and colorized image side by side
    combined_image = np.hstack((gray_image, colorized))

    # Convert the combined image to PIL Image
    combined_image = Image.fromarray(combined_image)

    # Save the combined image or perform any other operations as needed
    combined_image.save('output_image.jpg')

    return {"Prediction": "Colorization completed. Check 'output_image.jpg'"}

if __name__ == "__main__":
    uvicorn.run(app, host='127.0.0.1', port=5000)
