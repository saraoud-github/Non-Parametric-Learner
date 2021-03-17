import sys
from PyQt5.QtCore import (QCoreApplication, QPropertyAnimation, QDate, QDateTime, QMetaObject, QObject, QPoint, QRect, QSize, QTime, QUrl, Qt, QEvent)
from PyQt5.QtGui import (QBrush, QColor, QConicalGradient, QCursor, QFont, QFontDatabase, QIcon, QKeySequence, QLinearGradient, QPalette, QPainter, QPixmap, QRadialGradient)
from PyQt5.QtWidgets import *
from ui_functionsSecondWindow import *
from SecondWindow import Ui_SecondWindow

class MainSecondWindow(QMainWindow):

  #Second window constructor
    def __init__(self):
        QMainWindow.__init__(self)
        self.ui = Ui_SecondWindow()
        self.ui.setupUi(self)
        # print("init second")

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

        ## ==> SET UI DEFINITIONS
        UIFunctions.uiDefinitions(self)

        # SHOW ==> MAIN WINDOW
    def showWindow(self):
        self.show()

    # APP EVENTS
    def mousePressEvent(self, event):
        self.dragPos = event.globalPos()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainSecondWindow()
    sys.exit(app.exec_())