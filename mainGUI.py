import platform
from PyQt5.QtCore import (QCoreApplication, QPropertyAnimation, QDate, QDateTime, QMetaObject, QObject, QPoint, QRect, QSize, QTime, QUrl, Qt, QEvent)
from PyQt5.QtWidgets import *
from PyQt5.QtGui import QPixmap, QColor, QDropEvent
from ProjectIGUI import Ui_MainWindow
from ui_functions import *
from PyQt5 import QtGui
from ProjectIGUI import *
import sys, os
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap
from PyQt5 import QtCore, QtGui, QtWidgets
from mainSecondWindow import MainSecondWindow
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image, ImageEnhance
import pickle
from PIL import Image as im


class MainWindow(QMainWindow):
   # GUI Main Window constructor
        def __init__(self):
            QMainWindow.__init__(self)
            self.ui = Ui_MainWindow()
            self.ui.setupUi(self)

   #Create an instance of the second window
            self.secondwindow = MainSecondWindow()

            # MOVE WINDOW
            def moveWindow(event):
                # Restore before move
                if UIFunctions.returnStatus(self) == 1:
                    UIFunctions.maximize_restore(self)

                # IF LEFT CLICK MOVE WINDOW
                if event.buttons() == Qt.LeftButton:
                    self.move(self.pos() + event.globalPos() - self.dragPos)
                    self.dragPos = event.globalPos()
                    event.accept()

            # SET TITLE BAR
            self.ui.Title_Bar.mouseMoveEvent = moveWindow

            #Connect the "Predict" button to the testNewImage function
            self.ui.btn_predict.clicked.connect(self.testNewImage)

            ## ==> SET UI DEFINITIONS
            UIFunctions.uiDefinitions(self)

            # SHOW ==> MAIN WINDOW
            self.show()

            # APP EVENTS
        def mousePressEvent(self, event):
            self.dragPos = event.globalPos()

   #Function interfacing a new input to the GUI with the trained model 'model.sav'
        def testNewImage(self):

   #Get the path of the new function from the GUI widget QLineEdit
            path = self.ui.lineEdit.text()


            face_cascade = cv2.CascadeClassifier(
                'haarcascade_frontalface_default.xml')  # the trained face detection model found in OpenCV

            img = cv2.imread(path)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # convert image to grayscale

            faces = face_cascade.detectMultiScale(gray, 1.3, 4)  # existing function to detect the faces
            print("Number of Faces Found = {0}".format(len(faces)))

            # Open the trained model
            pick = open('model.sav', 'rb')
            model = pickle.load(pick)
            pick.close()

            if len(faces) != 0:  # if the detector was able find faces in the image
                for i, face in enumerate(faces):  # iterate over these faces to classify each
                    faces = [face]
                    print('Face Number: ', i + 1)

                    for (x, y, w, h) in faces:  # x,y,w and h make up the 4 courners of the face's bounding rectangle
                        img = cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)  # draw the bounding rectangle on the image
                        roi_gray = gray[y:y + h, x:x + w]
                        roi_color = img[y:y + h, x:x + w]
                        cropimg = img[y:y + h, x:x + w]  # crops the image to have the face only
                        cropimg = cv2.resize(cropimg, (
                        128, 128))  # resize the array of the image to have the same size used for training
                        cropimg = Image.fromarray(cropimg)  # creates an image from the array object
                        cropimg = cropimg.resize(
                            (128, 128))  # resize the shape of the image to the image size previously used
                        cropimg.save('orgimg.jpg')
                        image = cv2.imread('orgimg.jpg')
                        org_resized = cv2.resize(image, (381, 311))
                        cv2.imwrite('orgimg.jpg', org_resized)
                        plt.imshow(cropimg, cmap='gray')
                        # plt.show()

                        enhancer = ImageEnhance.Sharpness(cropimg)
                        img_sh = enhancer.enhance(5)
                        sharpened = np.asarray(img_sh)
                        edges = cv2.Canny(sharpened, 128, 128)
                        image = np.array(edges).flatten()  # 1D array of image == 16 384
                        image1 = image.reshape(1, 16384)
                        # image = image.resize(16384)
                        # image1= cv2.resize(image,(128,128))
                        prediction = model.predict(image1)

                        person = image.reshape(128, 128)  # making the 1D array 2D

                        newimg = im.fromarray(person)
                        newimg.save('edgeimg.jpg')
                        img = cv2.imread('edgeimg.jpg')
                        resized = cv2.resize(img, (381, 311))
                        cv2.imwrite('edgeimg.jpg', resized)

                        plt.imshow(person, cmap='gray')
                        # plt.show()

            categories = ['Old', 'Young']
            q = model.predict_proba(image1)


   #Print out the prediction onto the second window
            self.secondwindow.ui.label.setText((str)(categories[prediction[0]]))
   # Print out the prediction accuracy onto the second window
            self.secondwindow.ui.label_2.setText((str)(q[0][0] * 100))
   # Print out the edge-detected image onto the second window
            self.secondwindow.ui.label_6.setPixmap(QPixmap('edgeimg.jpg'))
   # Print out the original image with the face detected onto the second window
            self.secondwindow.ui.label_5.setPixmap(QPixmap('orgimg.jpg'))
   #Show the second window
            self.secondwindow.showWindow()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    sys.exit(app.exec_())