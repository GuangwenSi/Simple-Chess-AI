from PyQt5.QtWidgets import QWidget, QToolTip, QPushButton, QApplication, QLabel, QColorDialog
from PyQt5.QtWidgets import QMessageBox, QDesktopWidget, QSlider, QLineEdit, QSpinBox
from PyQt5.QtWidgets import QVBoxLayout, QWidget, QLCDNumber, QHBoxLayout, QFormLayout
from PyQt5.QtWidgets import QMainWindow, QFileDialog, QErrorMessage, QInputDialog
from PyQt5.QtGui import QFont, QPixmap, QImage, QIcon
from PyQt5.QtCore import QCoreApplication, Qt, QSize, pyqtSignal
from boardLabel import BoardLabel
import sys
import os
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

class PanelWindow(QMainWindow):
    """Panel class for interaction"""
    def __init__(self):
        super(PanelWindow, self).__init__()
        self._addActions()
        self.initUI()
        self.show()

    def _addActions(self):
        """add actions for Menu"""
        mainToolBar = self.addToolBar('Main tool bar')

        startMenu = self.menuBar().addMenu("Start(&S)")

        newGameAction = startMenu.addAction(QIcon("images/media-playback-start.png"), "New Game(&N)")
        mainToolBar.addAction(newGameAction)
        newGameAction.triggered.connect(self._newGameTriggered)
        # TODO: START NEW GAME

        exitAction = startMenu.addAction(QIcon("images/exit.png"), "Exit(&E)")
        mainToolBar.addAction(exitAction)
        exitAction.triggered.connect(QCoreApplication.instance().quit)

    def _newGameTriggered(self):
        """start a new game"""
        self.chessBoard.startNewGame()
        self.statusBar().showMessage("Starting a new chess game")

    def initUI(self):
        """initialize UI settings"""
        self.setWindowTitle("LAC CHESS AI V1.0")
        self.setWindowIcon(QIcon("chess.ico"))
        self.statusBar().showMessage("WELCOME TO PLAY CHESS!")

        self.mainLayout=QHBoxLayout()

        self.chessBoard=BoardLabel(parent=self)

        self.mainLayout.addWidget(self.chessBoard)

        self.mainCentralWidget=QWidget()
        self.setCentralWidget(self.mainCentralWidget)
        self.centralWidget().setLayout(self.mainLayout)

    def closeEvent(self, event):
        """deal with close window event"""
        reply = QMessageBox.question(self, 'Message', 'Are you sure to quit?', QMessageBox.Yes | QMessageBox.No,
                                     QMessageBox.No)
        if reply == QMessageBox.Yes:
            event.accept()
        else:
            event.ignore()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = PanelWindow()
    # canvas=CanvasSizeWidget()
    sys.exit(app.exec_())