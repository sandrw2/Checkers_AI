from random import randint
from BoardClasses import Move
from BoardClasses import Board
from collections import defaultdict
from datetime import datetime
import math 
from copy import deepcopy

#The following part should be completed by students.
#Students can modify anything except the class name and exisiting functions and varibles.

class MonteCarlo():
    def __init__(self, board, color, parent, parent_action, depth):
        self.time_start = datetime.now()
        self.max_turn_time = 10
        self.board = board
        self.color = color
        self.parent = parent
        self.parent_action = parent_action
        self.children = []
        self.num_visits = 0
        self.reward = 0
        self.available_actions = self.flatten_list(self.board.get_all_possible_moves(self.color))
        self.opponent = {1:2,2:1}
        self.depth = depth

    def flatten_list(self, alist):
        flat_list = []
        for element in alist:
            if type(element) is list:
                for item in element:
                    flat_list.append(item)
            else:
                flat_list.append(element)
        return flat_list

    def mcts_process(self):
        #print("MCTS")
        #returns a state
        current_state = self
        while (not current_state.is_terminal() and current_state.time_left):
            if not current_state.fully_expanded():
                return current_state.expand()
            else:
                current_state = current_state.select_child()
        return current_state

    def expand(self):
        #print("EXPAND")
        move = self.available_actions.pop()
        child = deepcopy(self)
        child.board.make_move(move, self.color)
        child_board = MonteCarlo(child.board, self.opponent[self.color], self, move, self.depth+1)
        self.children.append(child_board)
        return child_board 
    

    def ucb_value(self, child):
        #print("UCB_VAL")
        return (child.reward/child.num_visits) + math.sqrt((2* math.log(self.num_visits)/child.num_visits))
    
    def select_child(self):
        best_child = self.children[0]
        max_ucb = -1000000
        for i in self.children:
            if(self.ucb_value(i) > max_ucb):
                max_ucb = self.ucb_value(i)
                best_child = i
        return best_child

    def select_move(self):
        # print("HERE UR KIDS:", len(self.children))
        best_child = self.children[0]
        max_win = -1000000
        for i in self.children:
            temp = i.reward/i.num_visits
            if(temp > max_win):
                max_win = temp
                best_child = i
        return best_child

    def tempVal(self, board):
        if self.color == 1:
            return board.black_count - board.white_count
        else:
            return board.white_count - board.black_count
   
    def rollout(self, board):
        #print("ROLLOUT")
        current_rollout_state = deepcopy(board)
        switch = 0
        depth = 0 

        temp_V = self.tempVal(current_rollout_state.board)
        # make the move at rollout stage
        while (not (current_rollout_state.is_terminal()) and self.time_left):
            if(switch):
                my_possible_moves = self.flatten_list(current_rollout_state.board.get_all_possible_moves(self.color))
                my_move = self.rollout_policy(my_possible_moves)
                current_rollout_state.board.make_move(my_move, self.color)
                switch = 0
                depth +=1
            else:
                op_possible_moves = self.flatten_list(current_rollout_state.board.get_all_possible_moves(self.opponent[self.color]))
                op_move = self.rollout_policy(op_possible_moves)
                current_rollout_state.board.make_move(op_move, self.opponent[self.color])
                switch = 1
                depth += 1
            if depth == 2 and board.depth == 1:
                temp_V2 = self.tempVal(current_rollout_state.board)
                #print("AT:", temp_V, temp_V2)
                if temp_V > temp_V2:
                    # print("ROLLING ON DEPTH:", current_rollout_state.depth)
                    # print("COLOR:", current_rollout_state.color, self.color)
                    # print("BAD MOVE")
                    return "AVOID"


            
        color = None
        if (self.color == 1):
            color = "B"
        else:
            color = "W"
        if(current_rollout_state.is_terminal()):
            #print("TERMINATED STATE")
            if (color == "W" and current_rollout_state.board.is_win("W")) or (color == "B" and current_rollout_state.board.is_win('B')):
                return "win"
            elif (color == "W" and current_rollout_state.board.is_win("B")) or (color == "B" and current_rollout_state.board.is_win('W')):
                return "lose"
            else:
                return "tie"
        else:
            #print("TERMINATED EARLY")
            #terminated by time --> evaluate unfinished game using heuristic 
            white_score, black_score = current_rollout_state.eval()
            if(color == "W" and (white_score > black_score)) | (color == "B" and (black_score > white_score)):
                return "win"
            elif(color == "B" and (white_score > black_score)) | (color == "W" and (black_score > white_score)):
                return "lose"
            else:
                return "tie"

            
        
    def rollout_policy(self, possible_moves):
        return  possible_moves[randint(0, len(possible_moves)-1)]

    def eval(self):
        king_val = 10
        men_val = 1
        white_score = 0
        black_score = 0 
        
        board_row = self.board.row
        board_col = self.board.col
        board = self.board.board

        for r in range(board_row):
            for c in range(board_col):
                piece = board[r][c]
                if piece.color == "B":
                    if piece.is_king:
                        black_score += king_val
                    else:
                        black_score += r*men_val
                if piece.color == "W": 
                    if piece.is_king:
                        white_score += king_val
                    else:
                        white_score += (board_row-r-1)*men_val
        return white_score, black_score

    def backpropogate(self, result):
        self.num_visits += 1
        if(result == "win"):
            self.reward += 1
        elif (result == "lose"):
            self.reward -=1
        elif (result == "AVOID"):
            self.reward -= 100
        else: 
            self.reward += 0.5

        if self.parent != None:
            self.parent.backpropogate(result)

    def best_action(self):
        while(self.time_left()):
            # 
            rollout_node = self.mcts_process()
            result = self.rollout(rollout_node)
            rollout_node.backpropogate(result)
        return self.select_move()
    
    def fully_expanded(self):
        return len(self.available_actions) == 0
    
    def get_visits(self):
        return self.number_visits

    def time_left(self):
        #returns true if there is time left
        time = (datetime.now() - self.time_start).seconds
        return time < self.max_turn_time

    def is_terminal(self):
        return self.board.is_win('W') | self.board.is_win('B')
    


class StudentAI():

    def __init__(self,col,row,p):
        self.col = col
        self.row = row
        self.p = p
        self.board = Board(col,row,p)
        self.board.initialize_game()
        self.color = ''
        self.opponent = {1:2,2:1}
        self.color = 2
    
    def get_move(self,move):
        #move object has length indicating how many spots it must traverse for specific move 
        # print("AI BOARD==============================")
        # self.board.show_board()
        # print("AI BOARD==============================")

        if len(move) != 0:
            self.board.make_move(move,self.opponent[self.color])
        else:
            self.color = 1
       
        # #moves --> list of moves
        # moves = self.board.get_all_possible_moves(self.color)
        # #choose random move from list of moves
        # #moves --> (Move piece 1: [(()-()), (()-()), (()-())], Move piece 2j: [(), ()], Move: [(),()])
        # move = self.minimax_search(4)
        # self.board.make_move(move, self.color)



        # if there is only one move to make, dont go into monte carlo
        all_moves = self.board.get_all_possible_moves(self.color)
        if len(all_moves) == 1:
            if len(all_moves[0]) == 1:
                self.board.make_move(all_moves[0][0], self.color)
                return all_moves[0][0]

        root = MonteCarlo(deepcopy(self.board), self.color, None, None, 0)
        #print(root.best_action().parent_action)
        # print("NUM MOVES:", sum([len(all_moves[y]) for y in range(0, len(all_moves))]))
        try:
            self.board.make_move(root.best_action().parent_action, self.color)
            return root.best_action().parent_action
        except:
            #print("oops")
            self.board.make_move(all_moves[0][0], self.color)
            return all_moves[0][0]
