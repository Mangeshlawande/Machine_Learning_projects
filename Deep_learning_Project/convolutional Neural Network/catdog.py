import uvicorn
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import models
from tensorflow.keras.models import load_model
import warnings
warnings.simplefilter('ignore')
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import Flatten
from sklearn.model_selection import train_test_split
from fastapi import FastAPI, File, UploadFile
from io import BytesIO
from PIL import Image
import numpy as np
import pandas as pd

app = FastAPI()

from keras.preprocessing.image import ImageDataGenerator
train = ImageDataGenerator(rescale = 1./255)
test = ImageDataGenerator(rescale = 1./255)
train_data = train.flow_from_directory('C:/Users/mange/Desktop/image/train', target_size = (100, 100), class_mode = 'binary')

test_data = test.flow_from_directory('C:/Users/mange/Desktop/image/test', target_size = (100, 100), class_mode = 'binary')
    
classifier=Sequential()
classifier.add(Convolution2D(32,3,3,input_shape=(100,100,3),activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2,2)))
classifier.add(Flatten())
classifier.add(Dense(128,activation='relu'))
classifier.add(Dense(1,activation='sigmoid'))

classifier.compile(optimizer=keras.optimizers.Adam(), loss="mean_squared_error",
                metrics=['accuracy'])

classifier.fit_generator(train_data,epochs=5,validation_data=test_data)



    # print(final)

@app.get('/')
def index():
    return {'Deployment': 'Prediction of Images Between the dog and cat'}


@app.post("/predict")
async def predict1(
    file: UploadFile = File(...)):

    

    image = await file.read()
    image = Image.open(BytesIO(image))
    image = image.convert('RGB')

     # Resize the image to match the input size expected by your model
    image = image.resize((224, 224))

    predict_modified = np.array(image)

    predict_modified = predict_modified / 255
    predict_modified = np.expand_dims(predict_modified, axis = 0)
    final = classifier.predict(predict_modified)
    prediction = final[0]
    if(prediction < 0.5):
        output = 'Cat'
    if(prediction > 0.5):
        output = 'Dog'

    return {"Prediction": output}


if __name__ == "__main__":
    uvicorn.run(app, host='127.0.0.1', port=5000)