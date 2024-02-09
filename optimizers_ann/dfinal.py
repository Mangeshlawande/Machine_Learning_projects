# working properly


import uvicorn
import tensorflow as tf
from tensorflow import keras 
from tensorflow.python.keras import models
from tensorflow.python.keras import Sequential
from sklearn.model_selection import train_test_split
from fastapi import FastAPI , File, UploadFile
from io import BytesIO
from PIL import Image
import numpy as np
import pandas as pd

app = FastAPI()

@app.get('/')
def index():
    return {'Deployment': 'Prediction of Images Between 1 to 9'}

@app.post("/predict")
async def predict1(
    file: UploadFile = File(...)):
    train = pd.read_csv("/Users/mange/Desktop/mnist/mnist_train.csv")
    test = pd.read_csv('/Users/mange/Desktop/mnist/mnist_test.csv')

    y_train = train.iloc[:,0]
    X_train = train.iloc[:,1:]
    y_test = test.iloc[:,0]
    X_test = test.iloc[:,1:]

    X_train = X_train/ 255
    X_test = X_test /255
    X_train_flattened = X_train.values.reshape(len(X_train), 28*28)
    X_test_flattened = X_test.values.reshape(len(X_test), 28*28)


    model = keras.Sequential([
        keras.layers.Dense(300, input_dim=(784),activation = 'relu'),
        keras.layers.Dense(10, activation='softmax')
    ])


    model.compile(optimizer='sgd',loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    y_train = y_train.astype(str).astype(int)
    print(y_train)
    model.fit(X_train_flattened, y_train, epochs=5)

    image = await file.read()
    image = Image.open(BytesIO(image))
    image = image.convert("L")

    # Resize the image to (28, 28)
    image = image.resize((28, 28))

    import numpy as np
    pic = np.array(image)

    pic = pic/255
     # This will flatten the array while maintaining its size
    
    pic_flattened = pic.reshape(28*28)
    pic_flattened = np.expand_dims(pic_flattened, axis=0)
    predicted = model.predict(pic_flattened)
    prediction = np.argmax(predicted[0])
    if(prediction==0):
        output='Zero'
    if(prediction==1):
        output='One'
    if(prediction==2):
        output='Two'
    if(prediction==3):
        output='Three'
    if(prediction==4):
        output='four'
    if(prediction==5):
        output='Five'
    if(prediction==6):
        output='Six'
    if(prediction==7):
        output='Seven'
    if(prediction==8):
        output='Eight'
    if(prediction==9):
        output='Nine'

    return{"Prediction": output}


if __name__=="__main__":
    uvicorn.run(app, host='127.0.0.1', port = 5000)
