import sys
import numpy as np
from PyQt5.QtWidgets import (QWidget, QToolTip, QPushButton, QApplication, QLabel, QColorDialog)
from PyQt5.QtWidgets import QMessageBox, QDesktopWidget, QSlider, QLineEdit, QSpinBox
from PyQt5.QtWidgets import QVBoxLayout, QWidget, QLCDNumber, QHBoxLayout, QFormLayout
from PyQt5.QtWidgets import QMainWindow, QErrorMessage
from PyQt5.QtGui import QFont, QPixmap, QImage, QIcon, QPainter
from PyQt5.QtCore import QCoreApplication, Qt, QSize, pyqtSignal, QThread
import matplotlib.pyplot as plt
import cv2 as cv
from chessJudger import ChessJudger
from chessTypes import *
from ai import ChessAI
import time

WHITE=0x0
BLACK=0x40

class AICalculateThread(QThread):
    _signal = pyqtSignal(str)
 
    def __init__(self,board):
        super(AICalculateThread, self).__init__()
        self.board=board

    def __del__(self):
        self.wait()
 
    def run(self):
        self.opponent=ChessAI(nSteps=4)
        x1,y1,x2,y2=self.opponent.findNextMove(self.board)
        retStr=str.format("%d,%d,%d,%d" % (x1,y1,x2,y2))
        self._signal.emit(retStr)

class BoardLabel(QLabel):
    """My label class, with mouse event overwritten"""
    def __init__(self, parent=None,w=480,h=480):
        super(BoardLabel, self).__init__(parent)
        self.parent = parent
        self.recordPosition = False
        self.chessSelected = False
        self.playing = False
        self.setFixedSize(w,h)
        self.opponent=ChessAI(nSteps=4)
        self.selectedPosition = [-1,-1]
        self.boardImage=np.zeros((480,480,3),dtype=np.uint8)
        self.loadChesses()
        self.initBlankBoard()
        self.sync=True
        self.displayBoard()

    def startNewGame(self):
        self.initBoard()
        self.displayBoard()
        self.playing=True
        self.recordPosition=True

    def loadChesses(self):
        """load all chess images"""
        self.chessColor=dict([(0,"white"),(64,"black")])
        self.chessName=dict([(2,"Castle"),(4,"Horse"),(8,"Knight"),(16,"Queen"),(32,"King"),(1,"Soldier")])
        self.chessImageDict=dict()
        for i in self.chessColor.keys():
            for j in self.chessName.keys():
                chessPicName="images/allChesses/"+self.chessColor[i]+self.chessName[j]+".jpg"
                chessPic=cv.imread(chessPicName)
                self.chessImageDict[i+j]=chessPic
        print("ALL TYPES OF CHESS IMAGE LOADED")

    def initBlankBoard(self):
        self.chessBoard=np.zeros((8,8),dtype=np.int32)

    def initBoard(self):
        """init a board with chess positions"""
        self.chessBoard=np.array([[66,68,72,80,96,72,68,66],
                                  [65,65,65,65,65,65,65,65],
                                  [0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ],
                                  [0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ],
                                  [0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ],
                                  [0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ],
                                  [1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ],
                                  [2 ,4 ,8 ,16,32,8 ,4 ,2 ]],dtype=np.int32)
        print("BOARD INITED WITH ORIGINAL STATE")

    def judgeGame(self,color):
        """judge whether game has ended for color side"""
        if color==WHITE:
            if BLACK+KING not in self.chessBoard:
                self.parent.statusBar().showMessage("YOU WIN!")
                self.recordPosition=False
            else:
                self.recordPosition=True
        elif color==BLACK:
            if KING not in self.chessBoard:
                self.parent.statusBar().showMessage("YOU LOSE!")
                self.recordPosition=False
            else:
                self.recordPosition=True

    def redrawBoard(self):
        """redraw board according to present situation"""
        self.boardImage=np.zeros((480,480,3),dtype=np.uint8)
        for i in range(8):
            for j in range(8):
                self._displayGrid(i,j)
        self.sync=True

    def displayBoard(self):
        """display the board"""
        self.redrawBoard()
        QImg = QImage(self.boardImage.data, 480, 480, 480 * 3, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(QImg)
        self.setPixmap(pixmap)

    def _displayGrid(self,i,j):
        """display board at given position"""
        if self.chessBoard[i][j]==0: # simple grid
	        # colors=[0x00b35c00,0x003de1ad]
            # print("Simple grid at (%d,%d)" % (i,j))
            colors=[[0x00,0x5c,0xb3],[0xad,0xe1,0x3d]]
            for k in range(3):
                self.boardImage[60*i:60*(i+1),60*j:60*(j+1),k]=colors[(i+j)%2][2-k]
        else:
            self.boardImage[60*i:60*(i+1),60*j:60*(j+1)]=self.chessImageDict[self.chessBoard[i][j]]          

    def showPossibleMoves(self,validMoves):
        """show possible movements in red color"""
        positionsX,positionsY=np.where(validMoves==1)
        nPositions=len(positionsX)
        for i in range(nPositions):
            x=positionsX[i]; y=positionsY[i]
            colors=[0xff,0x00,0x00]
            for k in range(3):
                self.boardImage[60*x:60*(x+1),60*y:60*(y+1),k]=colors[k]
        # plt.imshow(self.boardImage); plt.show()
        QImg = QImage(self.boardImage.data, 480, 480, 480 * 3, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(QImg)
        self.setPixmap(pixmap)    
        self.sync=False # NEED TO SYNCHRONIZE WHEN MOVEMENTS ARE MADE    

    def aiCallback(self, msg):
        print("Total time: [%.3f] seconds" % (time.time()-self.start))
        splitMsg=msg.split(',')
        intMsg=[int(msg) for msg in splitMsg]
        x1,y1,x2,y2=intMsg
        judger=ChessJudger()
        self.chessBoard=judger.movedSituation(self.chessBoard,[x1,y1],[x2,y2])
        self.displayBoard()
        self.parent.statusBar().showMessage("")
        self.judgeGame(BLACK)

    def mouseReleaseEvent(self, event):
        """mouse released"""
        x = event.pos().x()
        y = event.pos().y()
        # print("Mouse pressed at", x, y)
        x,y=y,x
        # assert(x>=0 and x<450 and y>=0 and y<450)
        if self.recordPosition==True:
            positionX=int(x/60)
            positionY=int(y/60)
            chessValue=self.chessBoard[positionX][positionY]
            # print("Releasing mouse and selecting (%d,%d) which is %d" % (positionX,positionY,chessValue))
            if self.chessSelected==False:
                if chessValue==0 or chessValue>=64: # None or black
                    return
                # FIRST SELECT
                # print(positionX,positionY,chessValue)
                self.parent.statusBar().showMessage(self.chessName[chessValue]+" selected")
                self.chessSelected=True
                self.selectedPosition=[positionX,positionY]
                judger=ChessJudger()
                validMoves=judger.validMoves(self.chessBoard,self.selectedPosition)
                self.validMoves=validMoves
                self.showPossibleMoves(validMoves)
            else: # need to check whether this is a valid move                 
                x,y=self.selectedPosition
                lastChessValue=self.chessBoard[x][y]
                if self.validMoves[positionX][positionY]==0:
                    if chessValue>0 and chessValue<64 and (x!=positionX or y!=positionY): # selected another chess
                        self.parent.statusBar().showMessage(self.chessName[chessValue]+" selected")
                        self.selectedPosition=[positionX,positionY]  
                        judger=ChessJudger()
                        validMoves=judger.validMoves(self.chessBoard,self.selectedPosition)
                        self.validMoves=validMoves
                        if self.sync==False:
                            self.redrawBoard()
                        self.showPossibleMoves(validMoves) 
                    else:
                        self.parent.statusBar().showMessage("Cancelled.")
                        self.selectedPosition=[-1,-1]
                        self.chessSelected=False
                        self.displayBoard()
                else:
                    judger=ChessJudger()
                    self.chessBoard=judger.movedSituation(self.chessBoard,self.selectedPosition,[positionX,positionY])
                    self.selectedPosition=[-1,-1]
                    self.chessSelected=False
                    self.displayBoard()
                    self.judgeGame(WHITE)
                    self.parent.statusBar().showMessage("Waiting ...")
                    self.recordPosition=False # STOP TO WAIT FOR THREAD FINISH
                    self.start=time.time()
                    self.thread=AICalculateThread(self.chessBoard)
                    self.thread._signal.connect(self.aiCallback)
                    self.thread.start()