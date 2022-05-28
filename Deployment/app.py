import numpy as np
import pandas as pd
#import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from pathlib import Path
from keras.models import load_model
import gradio as gr

model1 = load_model('BrailleNet.h5')

target_names = ['a', 'b', 'c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']
def predict_image(img):
    input=img.reshape(-1,28,28,3)
    prediction=model1.predict(input)[0]
    return {target_names[i]: float(prediction[i]) for i in range(26)}

image = gr.inputs.Image(shape=(28,28))
label = gr.outputs.Label(num_top_classes=26)

gr.Interface(fn=predict_image, inputs=image, outputs=label,interpretation='default').launch(debug='True')