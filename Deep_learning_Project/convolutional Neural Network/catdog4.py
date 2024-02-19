# working

import uvicorn
import tensorflow as tf
from fastapi import FastAPI, File, UploadFile
from io import BytesIO
from PIL import Image
import numpy as np
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, Flatten, Dense
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator

app = FastAPI()

train_data = ImageDataGenerator(rescale=1./255).flow_from_directory(
    'C:/Users/mange/Desktop/image/train', target_size=(100, 100), class_mode='binary'
)

test_data = ImageDataGenerator(rescale=1./255).flow_from_directory(
    'C:/Users/mange/Desktop/image/test', target_size=(100, 100), class_mode='binary'
)

classifier = Sequential()
classifier.add(Convolution2D(32, 3, 3, input_shape=(100, 100, 3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))
classifier.add(Flatten())
classifier.add(Dense(128, activation='relu'))
classifier.add(Dense(1, activation='sigmoid'))

classifier.compile(optimizer=Adam(), loss="mean_squared_error", metrics=['accuracy'])

classifier.fit_generator(train_data, epochs=5, validation_data=test_data)

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


    predict_modified = np.array(image)
    predict_modified = predict_modified / 255
    predict_modified = np.expand_dims(predict_modified, axis=0)

    final = classifier.predict(predict_modified)
    prediction = final[0][0]

    output = 'Cat' if prediction < 0.5 else 'Dog'

    return {"Prediction": output}

if __name__ == "__main__":
    uvicorn.run(app, host='127.0.0.1', port=5000)
