import cv2
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from PIL import Image 
import PIL.ImageOps
import os, time, ssl

X = np.load('image.npz')['arr_0']
Y= pd.read_csv("laels_csv")("labels")

classes=["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]

nclasses=len(classes)

X_train, X_test, Y_train, Y_test = train_test_split = (X, Y, random_state=9, train_size=3500, test_size=500)

xTrainScaled=X_train/255.0
xTestScaled=X_test/255.0

classifier= LogisticRegression(solver="saga", multi_class= "multinomial").fit(xTrainScaled, Y_train)

def get_prediction(image):
    im_pil = Image.open(image)

    image_bw = im_pil.convert('L')
    image_bw_resized = image_bw.resize((28,28), Image.ANTIALIAS)

    image_bw_resized_inverted = PIL.ImageOps.invert(image_bw_resized)
    pixel_filter = 20
    
    min_pixel = np.percentile(image_bw_resized_inverted, pixel_filter)
    
    image_bw_resized_inverted_scaled = np.clip(image_bw_resized_inverted-min_pixel, 0, 255)
    max_pixel = np.max(image_bw_resized_inverted)
    image_bw_resized_inverted_scaled = np.asarray(image_bw_resized_inverted_scaled)/max_pixel

    test_sample = np.array(image_bw_resized_inverted_scaled).reshape(1,660)
    test_pred = clf.predict(test_sample)
    
    return test_pred[0]
















