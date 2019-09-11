import numpy as np
import random as r
from chessJudger import ChessJudger
from eprogress import LineProgress
from chessTypes import *
from functools import cmp_to_key

# CHESS AI CLASS

WHITE=True
BLACK=False

class ChessAI(object):
    """chess AI for playing"""
    def __init__(self,nSteps=3):
        self.nSteps=nSteps # consider at most N steps

    def findNextMove(self,board):
        """try to find next move based on present board"""
        self.bestMove=[-1,-1,-1,-1]
        self.minMaxSearch(board,BLACK,self.nSteps,-1e9,1e9)
        return self.bestMove

    def endBoard(self,board,color):
        """judge whether the game has ended for color side"""
        if color==WHITE:
            return KING not in board
        else:
            return 64+KING not in board

    def rearrangePositions(self,color,positionsX,positionsY):
        """sort all positions according to their potential in general"""
        pairs=list(zip(positionsX,positionsY))
        # print(pairs,color)
        if color==WHITE:
            pairs=sorted(pairs,key=lambda x:1000*(7-x[0])+min(x[1],7-x[1]),reverse=True)
        elif color==BLACK:
            pairs=sorted(pairs,key=lambda x:1000*x[0]+min(x[1],7-x[1]),reverse=True)  
        return pairs

    def minMaxSearch(self,board,color,nSteps,alpha,beta):
        """search for the best solution within nSteps range"""
        if self.endBoard(board,WHITE):
            return 1e9
        elif self.endBoard(board,BLACK):
            return -1e9
        if nSteps==0:
            return ChessJudger().evaluate(board)
        if color==WHITE: # enemy, min
            ret=1e9
        else: # ai side, max
            ret=-1e9
        if nSteps==self.nSteps:
            nCounter=len(np.where(board>64)[0])
            displayer=LineProgress(total=nCounter)
            nCounter=0
        if color==WHITE:
            iTurn=list(range(8))
        elif color==BLACK:
            iTurn=list(range(7,-1,-1))
        jTurn=[3,4,2,5,1,6,0,7]
        for i in iTurn:
            for j in jTurn:
                if (color==BLACK and board[i][j]>64) or (color==WHITE and board[i][j]>0 and board[i][j]<64): # black
                    tempJudger=ChessJudger()
                    validMoves=tempJudger.validMoves(board,[i,j])
                    positionsX,positionsY=np.where(validMoves==1)
                    rearrangedPairs=self.rearrangePositions(color,positionsX,positionsY)
                    for (x,y) in rearrangedPairs:
                        newBoard=tempJudger.movedSituation(board,[i,j],[x,y])
                        playoutPotential=self.minMaxSearch(newBoard,not color,nSteps-1,alpha,beta)
                        if color==WHITE: # enemy, min
                            ret=min(ret,playoutPotential)
                            beta=min(beta,ret)
                            if beta<=alpha:
                                # print("ALPHA CUT!")
                                return alpha
                        else:
                            if nSteps==self.nSteps:
                                # displayer.update(nCounter+1)
                                # nCounter+=1
                                if ret<playoutPotential: # TOP LAYER, upgrade best move
                                    self.bestMove=[i,j,x,y]
                                    print("FIND:",self.bestMove,playoutPotential)
                            ret=max(ret,playoutPotential)
                            alpha=max(alpha,ret)
                            if beta<=alpha:
                                # print("BETA PRUNED！")
                                return beta
        return ret