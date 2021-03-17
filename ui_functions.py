#GUI FILE
from mainGUI import QColor, QMainWindow, Qt, QSizeGrip, QGraphicsDropShadowEffect
from PyQt5 import QtCore
from mainGUI import *
## ==> GLOBALS

GLOBAL_STATE = 0

class UIFunctions(QMainWindow):

    ## ==> MAXIMIZE RESTORE FUNCTION
    def maximize_restore(self):
        global GLOBAL_STATE
        status = GLOBAL_STATE

        # IF NOT MAXIMIZED
        if status == 0:
            self.showMaximized()

            # SET GLOBAL TO 1
            GLOBAL_STATE = 1

            # IF MAXIMIZED REMOVE MARGINS AND BORDER RADIUS
            self.ui.verticalLayout.setContentsMargins(0, 0, 0, 0)
            self.ui.frame.setGeometry(500, 200, 911, 641)
            self.ui.drop_shadow_frame.setStyleSheet("background-color: qlineargradient(spread:pad, x1:0, y1:0.949, x2:1, y2:0, stop:0.0995025 rgba(82, 120, 145, 255), stop:0.985075 rgba(61, 71, 91, 255)); border-radius: 20px;")
            self.ui.btn_maximize.setToolTip("Restore")
        else:
            GLOBAL_STATE = 0
            self.showNormal()
            self.resize(self.width()+1, self.height()+1)
            self.ui.frame.setGeometry(40, 30, 911, 641)
            self.ui.verticalLayout.setContentsMargins(10, 10, 10, 10)
            self.ui.drop_shadow_frame.setStyleSheet("background-color: qlineargradient(spread:pad, x1:0, y1:0.949, x2:1, y2:0, stop:0.0995025 rgba(82, 120, 145, 255), stop:0.985075 rgba(61, 71, 91, 255)); border-radius: 20px;")
            self.ui.btn_maximize.setToolTip("Maximize")

    ## ==> UI DEFINITIONS
    def uiDefinitions(self):

        # REMOVE TITLE BAR
        self.setWindowFlag(QtCore.Qt.FramelessWindowHint)
        self.setAttribute(QtCore.Qt.WA_TranslucentBackground)

        # SET DROPSHADOW WINDOW
        self.shadow = QGraphicsDropShadowEffect(self)
        self.shadow.setBlurRadius(20)
        self.shadow.setXOffset(0)
        self.shadow.setYOffset(0)
        self.shadow.setColor(QColor(0, 0, 0, 100))

        # APPLY DROPSHADOW TO FRAME
        self.ui.drop_shadow_frame.setGraphicsEffect(self.shadow)

        # MAXIMIZE / RESTORE
        self.ui.btn_maximize.clicked.connect(lambda: UIFunctions.maximize_restore(self))

         # MINIMIZE
        self.ui.btn_minimize.clicked.connect(lambda: self.showMinimized())

        # CLOSE
        self.ui.btn_close.clicked.connect(lambda: self.close())





    ## RETURN STATUS IF WINDOWS IS MAXIMIZE OR RESTAURED
    def returnStatus(self):
        return GLOBAL_STATE