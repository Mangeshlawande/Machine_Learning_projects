import uvicorn
import tensorflow as tf
from fastapi import FastAPI, File, UploadFile
from io import BytesIO
from PIL import Image
import numpy as np

app = FastAPI()

# Define your CNN model
model = tf.keras.Sequential([
    # Add your convolutional layers, pooling layers, and dense layers here
    # Example:
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')  # Assuming binary classification (cat vs dog)
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Optional: Train the model with your dataset (replace with your actual training code)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

@app.get('/')
def index():
    return {'Deployment': 'Prediction of Images Between the dog and cat'}

@app.post("/predict")
async def predict1(file: UploadFile = File(...)):
    image = await file.read()
    image = Image.open(BytesIO(image))
    image = image.convert('RGB')

    # Resize the image to match the input size expected by your model
    image = image.resize((224, 224))
    
    pic = np.array(image)
    pic = pic / 255
    pic = np.expand_dims(pic, axis=0)

    # Make a prediction using the model
    predicted = classifier.predict(pic)
    prediction = predicted[0][0]

    # Set a threshold for binary classification
    output = 'Cat' if prediction < 0.5 else 'Dog'

    return {"Prediction": output}

if __name__ == "__main__":
    uvicorn.run(app, host='127.0.0.1', port=5000)
