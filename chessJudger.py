import numpy as np
from chessTypes import *
from numba import jit

# CHESS JUDGER FILE, mainly judging whether a given move is valid, and compute relevant
# potential values for differnt sides under different circumstances

WHITE=True
BLACK=False

class ChessJudger(object):
    """Judger for a lot of situations in practice"""
    def __init__(self):
        self.board=np.zeros((8,8),dtype=np.int32)
        self.initMovementCalculators()
        self.typeValue=dict([(SOLDIER,100),(HORSE,300),(KNIGHT,325),(CASTLE,500),(QUEEN,900),(KING,20000)])
        self.typeWeightMatrices=dict()
        self.typeWeightMatrices[SOLDIER]=np.array([[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],
                                                   [5.0,5.0,5.0,5.0,5.0,5.0,5.0,5.0],
                                                   [1.0,1.0,2.0,3.0,3.0,2.0,1.0,1.0],
                                                   [0.5,0.5,1.0,2.5,2.5,1.0,0.5,0.5],
                                                   [0.3,0.3,0.6,2.0,2.0,0.6,0.3,0.3],
                                                   [0.2,0.2,0.4,1.2,1.2,0.4,0.2,0.2],
                                                   [0.1,0.1,0.2,1.0,1.0,0.2,0.1,0.1],
                                                   [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]])
        self.typeWeightMatrices[HORSE]=np.array([[0.1,0.2,0.4,0.4,0.4,0.4,0.2,0.1],
                                                 [0.2,0.4,0.7,0.7,0.7,0.7,0.4,0.2],
                                                 [0.4,0.7,1.0,1.5,1.5,1.0,0.7,0.4],
                                                 [0.4,0.7,1.5,2.0,2.0,1.5,0.7,0.4],
                                                 [0.4,0.7,1.5,2.0,2.0,1.5,0.7,0.4],
                                                 [0.4,0.7,1.0,1.5,1.5,1.0,0.7,0.4],
                                                 [0.2,0.4,0.7,0.7,0.7,0.7,0.4,0.2],
                                                 [0.1,0.2,0.4,0.4,0.4,0.4,0.2,0.1]])
        self.typeWeightMatrices[KNIGHT]=np.array([[0.8,1.0,1.0,1.0,1.0,1.0,1.0,0.8],
                                                  [0.9,1.1,1.1,1.1,1.1,1.1,1.1,0.9],
                                                  [0.8,1.2,1.2,1.2,1.2,1.2,1.2,0.8],
                                                  [0.9,1.2,1.2,1.2,1.2,1.2,1.2,0.9],
                                                  [0.9,1.1,1.1,1.1,1.1,1.1,1.1,0.9],
                                                  [0.7,1.0,1.0,1.0,1.0,1.0,1.0,0.7],
                                                  [0.6,0.8,0.8,0.8,0.8,0.8,0.8,0.6],
                                                  [0.5,0.7,0.7,0.7,0.7,0.7,0.7,0.5]])
        self.typeWeightMatrices[CASTLE]=np.array([[1.5,1.6,1.7,3.5,3.5,1.7,1.6,1.5],
                                                  [1.4,1.5,1.6,3.0,3.0,1.6,1.5,1.4],
                                                  [1.4,1.5,1.6,2.8,2.8,1.6,1.5,1.4],
                                                  [1.3,1.4,1.5,2.7,2.7,1.5,1.4,1.3],
                                                  [1.3,1.4,1.5,2.7,2.7,1.5,1.4,1.3],
                                                  [1.2,1.3,1.4,2.6,2.6,1.4,1.3,1.2],
                                                  [1.1,1.2,1.3,2.5,2.5,1.3,1.2,1.1],
                                                  [1.0,1.0,1.0,2.0,2.0,1.0,1.0,1.0]])
        self.typeWeightMatrices[QUEEN]=np.array([[1.7,2.2,3.7,3.7,3.7,3.7,2.2,1.7],
                                                 [1.6,2.1,3.6,3.6,3.6,3.6,2.1,1.6],
                                                 [1.5,2.0,3.5,3.5,3.5,3.5,2.0,1.5],
                                                 [1.2,1.7,3.3,3.3,3.3,3.3,1.5,1.0],
                                                 [1.0,1.5,3.0,3.0,3.0,3.0,1.5,1.0],
                                                 [0.8,1.2,2.5,2.5,2.5,2.5,1.2,0.8],
                                                 [0.7,1.0,2.0,2.0,2.0,2.0,1.0,0.7],
                                                 [0.5,0.7,0.7,0.8,0.8,0.7,0.7,0.5]])
        self.typeWeightMatrices[KING]=np.array([[1.4,1.1,1.1,1.1,1.1,1.1,1.1,1.4],
                                                [1.4,1.1,1.1,1.1,1.1,1.1,1.1,1.4],
                                                [1.4,1.1,1.1,1.1,1.1,1.1,1.1,1.4],
                                                [1.4,1.1,1.1,1.1,1.1,1.1,1.1,1.4],
                                                [1.4,1.1,1.1,1.1,1.1,1.1,1.1,1.4],
                                                [1.4,1.1,1.1,1.1,1.1,1.1,1.1,1.4],
                                                [1.3,1.1,1.1,1.1,1.1,1.1,1.1,1.3],
                                                [1.2,1.0,1.0,1.0,1.0,1.0,1.0,1.2]])

    def _stripSameColor(self,board,validMoves,color):
        """strip the movements which occupies the position of same color chess"""
        positionsX,positionsY=np.where(board>0)
        for k in range(len(positionsX)):
            i=positionsX[k]; j=positionsY[k]
            if board[i][j]>0 and color==self._isWhiteChess(board,i,j):
                validMoves[i][j]=0
    
    def evaluateSide(self,board,color):
        """evaluate single side"""
        ret=0
        positionsX,positionsY=np.where(board>0) 
        for k in range(len(positionsX)):
            i=positionsX[k]; j=positionsY[k]
            if self._isWhiteChess(board,i,j)==color:
                chessType=board[i][j]&0x3f
                if color==WHITE:
                    ret+=self.typeValue[chessType]*(1+self.typeWeightMatrices[chessType][i][j])
                else:
                    # assert(color==BLACK)
                    ret+=self.typeValue[chessType]*(1+self.typeWeightMatrices[chessType][7-i][j])
        return ret

    def evaluate(self,board):
        """evaluate present situation"""
        whiteSide=self.evaluateSide(board,WHITE)
        blackSide=self.evaluateSide(board,BLACK)
        return blackSide-whiteSide

    def _horseValidMoves(self,board,x1,y1):
        """calculate all possible moves for horse"""
        dx=[-2,-2,-1,-1,1,1,2,2]
        dy=[1,-1,2,-2,2,-2,1,-1]
        color=self._isWhiteChess(board,x1,y1)
        validMoves=np.zeros((8,8),dtype=np.uint8)
        for i in range(8):
            x=x1+dx[i]; y=y1+dy[i]
            if x<0 or x>=8 or y<0 or y>=8:
                continue
            validMoves[x][y]=1
        self._stripSameColor(board,validMoves,color)
        return validMoves

    def _knightValidMoves(self,board,x1,y1):
        """calculate all possible moves for knight"""
        dx=[-1,-1,1,1]
        dy=[-1,1,-1,1]
        color=self._isWhiteChess(board,x1,y1)
        validMoves=np.zeros((8,8),dtype=np.uint8)
        for i in range(4):
            x=x1+dx[i]
            y=y1+dy[i]
            while x>=0 and x<8 and y>=0 and y<8:
                validMoves[x][y]=1
                if board[x][y]>0:
                    break
                x+=dx[i]
                y+=dy[i]
        self._stripSameColor(board,validMoves,color)
        return validMoves

    def _castleValidMoves(self,board,x1,y1):
        """calculate all possible moves for castle"""
        dx=[-1,0,0,1]
        dy=[0,1,-1,0]
        color=self._isWhiteChess(board,x1,y1)
        validMoves=np.zeros((8,8),dtype=np.uint8)
        for i in range(4):
            x=x1+dx[i]
            y=y1+dy[i]
            while x>=0 and x<8 and y>=0 and y<8:
                validMoves[x][y]=1
                if board[x][y]>0:
                    break
                x+=dx[i]
                y+=dy[i]
        self._stripSameColor(board,validMoves,color)
        return validMoves
    
    def _queenValidMoves(self,board,x1,y1):
        """calculate all possible moves for queen"""
        dx=[-1,-1,-1,0,0,1,1,1]
        dy=[-1,0,1,-1,1,-1,0,1]
        color=self._isWhiteChess(board,x1,y1)
        validMoves=np.zeros((8,8),dtype=np.uint8)
        for i in range(8):
            x=x1+dx[i]
            y=y1+dy[i]
            while x>=0 and x<8 and y>=0 and y<8:
                validMoves[x][y]=1
                if board[x][y]>0:
                    break
                x+=dx[i]
                y+=dy[i]
        self._stripSameColor(board,validMoves,color)
        return validMoves

    def _kingValidMoves(self,board,x1,y1):
        """calculate all possible moves for king"""
        dx=[-1,-1,-1,0,0,1,1,1]
        dy=[-1,0,1,-1,1,-1,0,1]
        color=self._isWhiteChess(board,x1,y1)
        validMoves=np.zeros((8,8),dtype=np.uint8)
        for i in range(8):
            x=x1+dx[i]; y=y1+dy[i]
            if x<0 or x>=8 or y<0 or y>=8:
                continue
            validMoves[x][y]=1
        self._stripSameColor(board,validMoves,color)
        return validMoves

    def _soldierValidMoves(self,board,x1,y1):
        """calculate all possible move positions for soldier"""
        # print(board,x1,y1)
        color=self._isWhiteChess(board,x1,y1)
        validMoves=np.zeros((8,8),dtype=np.uint8)
        baseLine=6 if(color==WHITE) else 1
        direction=-1 if(color==WHITE) else 1
        # print(baseLine)
        # print(direction)
        if x1==baseLine:
            if board[x1+direction][y1]==0:
                validMoves[x1+direction][y1]=1
                if board[x1+2*direction][y1]==0:
                    validMoves[x1+2*direction][y1]=1
        else:
            if board[x1+direction][y1]==0:
                validMoves[x1+direction][y1]=1
        if y1>=1 and board[x1+direction][y1-1]>0 and self._isWhiteChess(board,x1+direction,y1-1)!=color:
            validMoves[x1+direction][y1-1]=1
        if y1<=6 and board[x1+direction][y1+1]>0 and self._isWhiteChess(board,x1+direction,y1+1)!=color:
            validMoves[x1+direction][y1+1]=1
        return validMoves

    def initMovementCalculators(self):
        """init checkers for different moves"""
        self.calculators=dict()
        self.calculators[SOLDIER]=self._soldierValidMoves
        self.calculators[HORSE]=self._horseValidMoves
        self.calculators[KNIGHT]=self._knightValidMoves
        self.calculators[CASTLE]=self._castleValidMoves
        self.calculators[QUEEN]=self._queenValidMoves
        self.calculators[KING]=self._kingValidMoves
        # print("ALL CHESS TYPE CALCULATORS INITED.")

    def _isEmptyGrid(self,board,x,y):
        return board[x][y]==0

    def _isWhiteChess(self,board,x,y):
        return board[x][y]<64

    def movedSituation(self,board,presentPosition,nextPosition):
        """return the moved board after given operation"""
        x1,y1=presentPosition
        x2,y2=nextPosition
        chessValue=board[x1][y1]
        chessType=board[x1][y1]&0x3f
        color=self._isWhiteChess(board,x1,y1)
        baseLine=0 if color==WHITE else 7
        newBoard=board.copy()
        if chessType==1:
            if x2==baseLine: # UPGRADATION
                newBoard[x1][y1]=0
                newBoard[x2][y2]=QUEEN if color==WHITE else 64+QUEEN
            else:
                newBoard[x1][y1]=0
                newBoard[x2][y2]=chessValue
        else:
            newBoard[x1][y1]=0
            newBoard[x2][y2]=chessValue
        return newBoard

    def validMoves(self,board,presentPosition):
        """calculate all valid movements from present position
            and wrap them in a new board containing only boolean values
        """
        x1,y1=presentPosition
        # assert(self._isEmptyGrid(board,x1,y1)==False)
        # chessType=board[x1][y1]&0x3f
        # print("Chess type:",chessType,board[x1][y1])
        return self.calculators[board[x1][y1]&0x3f](board,x1,y1)
        


        
    