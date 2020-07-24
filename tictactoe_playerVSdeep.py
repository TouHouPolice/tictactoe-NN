# Tic-Tac-Toe Program using 
# random number in Python 
  
# importing all necessary libraries 
import numpy as np 
import random 
from time import sleep 
import torch
import torchvision
from torchvision import transforms, datasets
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from tictacNet import Net
import math

# Creates an empty board 
def create_board(): 
    return(np.array([[0.01, 0.01, 0.01], 
                     [0.01, 0.01, 0.01], 
                     [0.01, 0.01, 0.01]])) 
  
# Check for empty places on board 
def possibilities(board): 
    l = [] 
      
    for i in range(len(board)): 
        for j in range(len(board)): 
              
            if board[i][j] == 0.01: 
                l.append((i, j)) 
    return(l) 

def checkandplace(i,j,player,board):
    i=i-1
    j=j-1
    if board[i][j]==0.01:
        board[i][j]=player
        return True,board
    else:
        return False,board
def possibleboards(player,board):
    boards=[]
    counter=0
    for i in range(len(board)): 
        for j in range(len(board)):            
            if board[i][j] == 0.01: 
                tempboard=board.copy()
                tempboard[i][j]=player
                boards.append(tempboard) 
                counter+=1
    return counter,boards
# Select a random place for the player 
def random_place(board, player): 
    selection = possibilities(board) 
    current_loc = random.choice(selection) 
    
    board[current_loc] = player 
    return(board) 
  
# Checks whether the player has three  
# of their marks in a horizontal row 
def row_win(board, player): 
    for x in range(len(board)): 
        win = True
          
        for y in range(len(board)): 
            if board[x, y] != player: 
                win = False
                continue
                  
        if win == True: 
            return(win) 
    return(win) 
  
# Checks whether the player has three 
# of their marks in a vertical row 
def col_win(board, player): 
    for x in range(len(board)): 
        win = True
          
        for y in range(len(board)): 
            if board[y][x] != player: 
                win = False
                continue
                  
        if win == True: 
            return(win) 
    return(win) 
  
# Checks whether the player has three 
# of their marks in a diagonal row 
def diag_win(board, player): 
    win = True
    y = 0
    for x in range(len(board)): 
        if board[x, x] != player: 
            win = False
    win = True
    if win: 
        for x in range(len(board)): 
            y = len(board) - 1 - x 
            if board[x, y] != player: 
                win = False
    return win 
  
# Evaluates whether there is 
# a winner or a tie  
def evaluate(board): 
    winner = 0
      
    for player in [1, -1]: 
        if (row_win(board, player) or
            col_win(board,player) or 
            diag_win(board,player)): 
            
            winner=player
             
    return winner 

def board_AI_perspective(player,boards,count):
    
    temp=np.array(boards)
    
    processedboards=np.reshape(temp,(count,9))
    if player==-1:
        for i in range (0,count):
            for j in range(0,9):              
                if processedboards[i,j]!=0.01:
                    processedboards[i,j]=-processedboards[i,j]

    return processedboards
        
        

checkpoint_path="Checkpoints/checkpoint90.pt"
checkpoint=torch.load(checkpoint_path)
net=Net()
optimizer= optim.Adam(net.parameters(),lr=0.001)
net.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
net=net.eval()

def play_game(): 
    player_role=0
    while player_role==0:
        player_role=input("Do you want to play on the offensive? y/n")
        if player_role=="y" or player_role=="Y":
            player_role=1
        elif player_role=="n" or player_role=="N":
            player_role=-1
        else:
            print ("Invalid input")
            player_role=0
    deep_role=-player_role
    board, winner, counter = create_board(), 0, 0
    
    while winner == 0 and counter<9:  
        for player in [1, -1]: 
            if player==deep_role:
                
                [count,options]=possibleboards(player,board)
                processedoptions=board_AI_perspective(player,options,count)
                
                with torch.no_grad():
                    X_current=torch.Tensor(processedoptions).to(net.device)
                    predictions=net(X_current)
                    #print(predictions)
                   
                    decisionindex=0
                    if deep_role==1:
                        decisionindex=torch.argmin(predictions[:,1])
                    elif deep_role==-1:
                        decisionindex=torch.argmin(predictions[:,0])
                    print(decisionindex)
                board=options[decisionindex]
                

            else:
                valid_input=False
                while valid_input==False:
                    row=input("Enter row (1-3):")
                    col=input("Enter column (1-3):")

                    try:
                        row=int(row)
                        col=int(col)
                    except:
                        print("Input is not a number")
                        continue         
                    if row>0 and row<4 and col>0 and col<4:
                        [success,board]=checkandplace(row,col,player,board)
                        if success==False:
                            print("Invalid move")
                            valid_input=False                           
                        else:
                            valid_input=True
                            break
                    else:
                        print("Out of range")
            print("")
            print(board)
            counter += 1
          
            winner = evaluate(board) 
            if winner != 0 : 
                break
            if counter==9 and winner==0:
                
                winner=0
                break

    if winner==player_role:
        print("You win!")
    elif winner==deep_role:
        print("You lose!")
    else:
        print ("Draw")
    return(winner) 



next_game=True
i=0
while next_game==True:
    i+=1
    print("Game: "+str(i))
    play_game()
    replay=input("Replay? y/n")
    if replay.lower()=="y":
        next_game=True
    elif replay.lower()=="n":
        next_game=False
    else:
        print ("Invalid input, quitting")
        next_game=False