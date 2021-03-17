import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from PIL import Image, ImageEnhance
import pickle
import random
import time
from sklearn.metrics import classification_report, plot_confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn import svm

    #Mark the starting time of the code
start = time.time()

    #Get the directory of the database
directory = r'C:\Users\sara6\OneDrive\Documents\Files\Uni\Intelligent Engineering Algorithms\Project\Databases'

categories = ['Old', 'Young']
data = []

    #Loop through all the images of old and young people in Databases Folder
for category in categories:
    path = os.path.join(directory, category)
    label = categories.index(category)

    for img in os.listdir(path):
        imgpath = os.path.join(path, img)
        passport = Image.open(imgpath)
    #Sharpen the image being read
        enhancer = ImageEnhance.Sharpness(passport)
        img_sh = enhancer.enhance(5)
        img_sh.save('sharpened.png')
        im = cv2.imread('sharpened.png')
    #Pass the image through the Canny edge detector and resize it to be 128x128
        edges = cv2.Canny(im, 128, 128)
    #Converts the image into a 2D numpy array and then flattens it a 1D array
        image = np.array(edges).flatten()
    #Append the array with it's respective label to data
        data.append([image, label])

    #Write the data into a pickle file before training
pick_in = open('data2.pickle', 'wb')
pickle.dump(data, pick_in)
pick_in.close()

   #Read the pickle file containing the labeled data
pick_in = open('data2.pickle', 'rb')
   #Load the pickle file into data variable
data = pickle.load(pick_in)
pick_in.close()

   #Shuffle the data
random.shuffle(data)
features = []
labels = []

   #Split the elements in data into features and labels
for feature, label in data:
    features.append(feature)
    labels.append(label)

   #Split the data into train (70%) and test data (30%)
xtrain, xtest, ytrain, ytest = train_test_split(features, labels, test_size=0.3)

   #Define a parameter grid for the SVM model
param_grid = {'C': [0.1, 1, 10, 100, 1000],
              'gamma': [0.01, 0.001, 0.0001],
              'kernel': ['rbf', 'poly', 'linear', 'sigmoid']}

   #Define the SVM model
svc = svm.SVC(probability=True)
   #Chooses the best parameters from param_grid for the SVM model
model = GridSearchCV(svc, param_grid, cv=3)
   #Trains the model on the specified training data
model.fit(xtrain, ytrain)
   #Prints the best parameters that the model chose for the given data
print(model.best_params_)

   #Saves the model in 'model.sav' folder
pick = open('model.sav', 'wb')
pickle.dump(model, pick)
pick.close()

   #Prints the execution time of the training phase
print('Execution time: ', time.time()-start)

   #Opens and reads the model
pick = open('model.sav', 'rb')
model = pickle.load(pick)
pick.close()

  #Testing phase: predict and store the predictions of the testing data in model_predictions
model_predictions = model.predict(xtest)
  #Print out a classification report for the model that includes: precision, accuracy, f-value, and recall
print(classification_report(ytest, model_predictions))

  #Plot the confusion matrix that includes the correctly and wrongly classified data
plot_confusion_matrix(model, xtest, ytest,values_format='d', display_labels=["old", "young"])
plt.show()

  #Another method used to calculate the accuarcy
accuracy = np.mean(model_predictions==ytest)

categories = ['Old', 'Young'];
print('Accuracy: ', accuracy);
print('Prediction is: ', categories[model_predictions[0]]);

   #Plot an edge-detected image from the training set
person = xtest[0].reshape(128,128);
# plt.subplot(121),plt.imshow(passport,cmap = 'gray')
plt.subplot(122),plt.imshow(person,cmap = 'gray')
plt.show()

