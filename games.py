"""Games or Adversarial Search (Chapter 5)"""

import copy
import itertools
import random
from collections import namedtuple

import numpy as np

#from utils import vector_add

GameState = namedtuple('GameState', 'to_move, utility, board, moves')

def gen_state(to_move='X', x_positions=[], o_positions=[], h=3, v=3):
    """Given whose turn it is to move, the positions of X's on the board, the
    positions of O's on the board, and, (optionally) number of rows, columns
    and how many consecutive X's or O's required to win, return the corresponding
    game state"""

    moves = set([(x, y) for x in range(1, h + 1) for y in range(1, v + 1)]) - set(x_positions) - set(o_positions)
    moves = list(moves)
    board = {}
    for pos in x_positions:
        board[pos] = 'X'
    for pos in o_positions:
        board[pos] = 'O'
    return GameState(to_move=to_move, utility=0, board=board, moves=moves)


# ______________________________________________________________________________
# MinMax Search


def minmax(game, state):
    """Given a state in a game, calculate the best move by searching
    forward all the way to the terminal states. [Figure 5.3]"""

    player = game.to_move(state)

    def max_value(state):
        if game.terminal_test(state):
            return game.utility(state, player)
        v = -np.inf
        for a in game.actions(state):
            v = max(v, min_value(game.result(state, a)))
        return v

    def min_value(state):
        if game.terminal_test(state):
            return game.utility(state, player)
        v = np.inf
        for a in game.actions(state):
            v = min(v, max_value(game.result(state, a)))
        return v

    # Body of minmax_decision:
    return max(game.actions(state), key=lambda a: min_value(game.result(state, a)))


def minmax_cutoff(game, state):
    """Given a state in a game, calculate the best move by searching
    forward all the way to the cutoff depth. At that level use evaluation func.
    
    The depth works better for 2, when the size of the board is large, if the board size is less than 6 then it 
    also works for depth 3
    
    For this function to work you will still have to change the depth in the UI so it calls this function.
    I've implemented the cutoff function differently so in order to call this the depth value has to be changed above  or equal to 0 
    """
    
    depth = 2
    player = game.to_move(state)

    def max_value(state, depth):
        if depth == 0 or game.terminal_test(state):
            return game.evaluation_func(state)

        v = -np.inf
        for a in game.actions(state):
            v = max(v, min_value(game.result(state, a), depth - 1))
        return v

    def min_value(state, depth):
        if depth == 0 or game.terminal_test(state):
            return game.evaluation_func(state)

        v = np.inf
        for a in game.actions(state):
            v = min(v, max_value(game.result(state, a), depth - 1))
        return v

    return max(game.actions(state), key=lambda a: min_value(game.result(state, a), depth-1))

# ______________________________________________________________________________


def expect_minmax(game, state):
    """
    [Figure 5.11]
    Return the best move for a player after dice are thrown. The game tree
	includes chance nodes along with min and max nodes.
	"""
    player = game.to_move(state)
    
    def max_value(state):
        v = -np.inf
        for a in game.actions(state):
            v = max(v, chance_node(state, a))
        return v
    
    def min_value(state):
        v = np.inf
        for a in game.actions(state):
            v = min(v, chance_node(state, a))
        return v
    
    def chance_node(state, action):
        res_state = game.result(state, action)
        if game.terminal_test(res_state):
            return game.utility(res_state, player)
        sum_chances = 0
        num_chances = len(game.chances(res_state))
        
        # Equal probability for all chance nodes
        probability = 1 / num_chances

        for r in game.chances(res_state):
            sum_chances += probability * expect_node(game.result(res_state, r))

        return sum_chances
        
    def expect_node(state):
        if game.terminal_test(state):
            return game.utility(state, player)
        if game.to_move(state) == player:
            return max_value(state)
        else:
            return min_value(state)

    return max(game.actions(state), key=lambda a: chance_node(state, a), default=None)
    
    


def expect_minmax_cutoff(game, state):
    """
    The game works better at a depth of 2
    Works better for the board size of 3
    
    
    For this function to work you will still have to change the depth in the UI so it calls this function.
    I've implemented the cutoff function differently so in order to call this the depth value has to be changed above  or equal to 0 
    """
    
    
    player = game.to_move(state)
    depth = 2
    
    def max_value(state, depth):
        v = -np.inf
        for a in game.actions(state):
            v = max(v, chance_node(state, a, depth - 1))
        return v
    
    def min_value(state, depth):
        v = np.inf
        for a in game.actions(state):
            v = min(v, chance_node(state, a, depth-1))
        return v
    
    def chance_node(state, action, depth):
        res_state = game.result(state, action)
        if depth == 0 or game.terminal_test(res_state):
            return game.evaluation_func(res_state)
        sum_chances = 0
        num_chances = len(game.chances(res_state))
        
        # Equal probability for all chance nodes
        probability = 1 /num_chances

        for r in game.chances(res_state):
            sum_chances += probability * expect_node(game.result(res_state, r), depth)
            print("sum chances are : ", sum_chances)
        return sum_chances

    def expect_node(state, depth):
        if depth == 0 or game.terminal_test(state):
            return game.evaluation_func(state)
        if game.to_move(state) == player:
            return max_value(state, depth)
        else:
            return min_value(state, depth)

    return max(game.actions(state), key=lambda a: chance_node(state, a, depth), default=None)

    
def alpha_beta_search(game, state):
    """Search game to determine best action; use alpha-beta pruning.
    As in [Figure 5.7], this version searches all the way to the leaves.
    
    
    """

    player = game.to_move(state)

    # Functions used by alpha_beta
    def max_value(state, alpha, beta):
        if game.terminal_test(state):
            return game.utility(state, player)
        v = -np.inf
        for a in game.actions(state):
            v = max(v, min_value(game.result(state, a), alpha, beta))
            if v >= beta:
                return v  # Prune remaining branches
            alpha = max(alpha, v)
        return v

    def min_value(state, alpha, beta):
        if game.terminal_test(state):
            return game.utility(state, player)
        v = np.inf
        for a in game.actions(state):
            v = min(v, max_value(game.result(state, a), alpha, beta))
            if v <= alpha:
                return v  # Prune remaining branches
            beta = min(beta, v)
        return v

    # Body of alpha_beta_search:
    best_action = None
    alpha = -np.inf
    beta = np.inf

    return max(game.actions(state), key=lambda a: min_value(game.result(state, a), alpha, beta))



#def alpha_beta_cutoff_search(game, state, d=4, cutoff_test=None, eval_fn=None):
    
def alpha_beta_cutoff_search(game, state):
    """Search game to determine best action; use alpha-beta pruning.
    This version cuts off search and uses an evaluation function.
    
    The game works better for the board size of 2
    It also works for the depth of 3 when the board size is less than 5 or 6
    
    
    For this function to work you will still have to change the depth in the UI so it calls this function.
    I've implemented the cutoff function differently so in order to call this the depth value has to be changed above or equal to 0 
    """
    depth = 2
    player = game.to_move(state)

    # Functions used by alpha_beta
    def max_value(state, alpha, beta, depth):
        if depth == 0 or game.terminal_test(state):
            return game.evaluation_func(state)
        v = -np.inf
        for a in game.actions(state):
            v = max(v, min_value(game.result(state, a), alpha, beta, depth - 1))
            if v >= beta:
                return v  # Prune remaining branches
            alpha = max(alpha, v)
        return v

    def min_value(state, alpha, beta, depth):
        if depth == 0 or game.terminal_test(state):
            return game.evaluation_func(state)
        v = np.inf
        for a in game.actions(state):
            v = min(v, max_value(game.result(state, a), alpha, beta, depth - 1))
            if v <= alpha:
                return v  # Prune remaining branches
            beta = min(beta, v)
        return v

    # Body of alpha_beta_search:
    alpha = -np.inf
    beta = np.inf

    return max(game.actions(state), key=lambda a: min_value(game.result(state, a), alpha, beta, depth)) 
    


# ______________________________________________________________________________
# Players for Games


def query_player(game, state):
    """Make a move by querying standard input."""
    print("current state:")
    game.display(state)
    print("available moves: {}".format(game.actions(state)))
    print("")
    move = None
    if game.actions(state):
        move_string = input('Your move? ')
        try:
            move = eval(move_string)
        except NameError:
            move = move_string
    else:
        print('no legal moves: passing turn to next player')
    return move


def random_player(game, state):
    """A player that chooses a legal move at random."""
    return random.choice(game.actions(state)) if game.actions(state) else None


def alpha_beta_player(game, state):
    if( game.d == -1):
        print("running the alpha beta search but not with the cutoff")
        return alpha_beta_search(game, state)
    elif(game.d >= 0):
        return alpha_beta_cutoff_search(game, state)


def minmax_player(game,state):
    if( game.d == -1):
        print("running the minmax and not with the cutoff")
        return minmax(game, state)
    elif(game.d >= 0):
        return minmax_cutoff(game, state)


def expect_minmax_player(game, state):
    if( game.d == -1):
        print("running the expect_minimax and not with the cutoff")
        return expect_minmax(game, state)
    elif(game.d >= 0):
        return expect_minmax_cutoff(game, state)


# ______________________________________________________________________________
# 


class Game:
    """A game is similar to a problem, but it has a utility for each
    state and a terminal test instead of a path cost and a goal
    test. To create a game, subclass this class and implement actions,
    result, utility, and terminal_test. You may override display and
    successors or you can inherit their default methods. You will also
    need to set the .initial attribute to the initial state; this can
    be done in the constructor."""

    def actions(self, state):
        """Return a list of the allowable moves at this point."""
        raise NotImplementedError

    def result(self, state, move):
        """Return the state that results from making a move from a state."""
        raise NotImplementedError

    def utility(self, state, player):
        """Return the value of this final state to player."""
        raise NotImplementedError

    def terminal_test(self, state):
        """Return True if this is a final state for the game."""
        return not self.actions(state)

    def to_move(self, state):
        """Return the player whose move it is in this state."""
        return state.to_move

    def display(self, state):
        """Print or otherwise display the state."""
        print(state)

    def __repr__(self):
        return '<{}>'.format(self.__class__.__name__)

    def play_game(self, *players):
        """Play an n-person, move-alternating game."""
        state = self.initial
        while True:
            for player in players:
                move = player(self, state)
                state = self.result(state, move)
                if self.terminal_test(state):
                    self.display(state)
                    return self.utility(state, self.to_move(self.initial))



class TicTacToe(Game):
    """Play TicTacToe on an h x v board, with Max (first player) playing 'X'.
    A state has the player to_move, a cached utility, a list of moves in
    the form of a list of (x, y) positions, and a board, in the form of
    a dict of {(x, y): Player} entries, where Player is 'X' or 'O'.
    depth = -1 means max search tree depth to be used."""

    def __init__(self, h=3, v=3, k=3, d=-1):
        self.h = h
        self.v = v
        self.k = k
        self.depth = d
        moves = [(x, y) for x in range(1, h + 1)
                 for y in range(1, v + 1)]
        self.initial = GameState(to_move='X', utility=0, board={}, moves=moves)

    def actions(self, state):
        """Legal moves are any square not yet taken."""
        return state.moves

    def result(self, state, move):
        if move not in state.moves:
            return state  # Illegal move has no effect
        board = state.board.copy()
        board[move] = state.to_move
        moves = list(state.moves)
        moves.remove(move)
        return GameState(to_move=('O' if state.to_move == 'X' else 'X'),
                         utility=self.compute_utility(board, move, state.to_move),
                         board=board, moves=moves)

    def utility(self, state, player):
        """Return the value to player; 1 for win, -1 for loss, 0 otherwise."""
        return state.utility if player == 'X' else -state.utility

    def terminal_test(self, state):
        """A state is terminal if it is won or there are no empty squares."""
        return state.utility != 0 or len(state.moves) == 0

    def display(self, state):
        board = state.board
        for x in range(1, self.h + 1):
            for y in range(1, self.v + 1):
                print(board.get((x, y), '.'), end=' ')
            print()

    def compute_utility(self, board, move, player):
        """If 'X' wins with this move, return 1; if 'O' wins return -1; else return 0."""
        if (self.k_in_row(board, move, player, (0, 1)) or
                self.k_in_row(board, move, player, (1, 0)) or
                self.k_in_row(board, move, player, (1, -1)) or
                self.k_in_row(board, move, player, (1, 1))):
            return self.k if player == 'X' else -self.k
        else:
            return 0

    def evaluation_func(self, state):
        """computes value for a player on board after move.
            Likely it is better to conside the board's state from 
            the point of view of both 'X' and 'O' players and then subtract
            the corresponding values before returning."""
        
        """Compute the value for a player on the board after a move."""
        player = state.to_move
        opponent = 'O' if player == 'X' else 'X'
        
        # Check for winning moves
        if state.utility == self.k:
            return float('inf') if player == 'X' else float('-inf')
        elif state.utility == -self.k:
            return float('-inf') if player == 'X' else float('inf')
        
        # Evaluate based on the number of player's symbols in rows, columns, and diagonals
        player_score = self.evaluate_player_score(state.board, player)
        
        opponent_score = self.evaluate_player_score(state.board, opponent)
        
        return player_score - opponent_score

    def evaluate_player_score(self, board, player):
        """Evaluate the player's score based on the number of symbols in rows, columns, and diagonals."""
        score = 0

        # Check rows and columns
        for i in range(0, self.h):
            row_symbols = [board.get((i, j), '.') for j in range(0, self.v)]
            col_symbols = [board.get((j, i), '.') for j in range(0, self.h)]
            score += self.evaluate_line(row_symbols, player)
            score += self.evaluate_line(col_symbols, player)

        # Check diagonals
        for i in range(0, self.h - self.k + 1):
            for j in range(0, self.v - self.k + 1):
                diagonal_symbols = [board.get((i + d, j + d), '.') for d in range(self.k)]
                anti_diagonal_symbols = [board.get((i + d, j + self.k - 1 - d), '.') for d in range(self.k)]
                score += self.evaluate_line(diagonal_symbols, player)
                score += self.evaluate_line(anti_diagonal_symbols, player)

        return score

    def evaluate_line(self, line_symbols, player):
        """Evaluate a line based on the number of player's symbols."""
        count = line_symbols.count(player)
        return count ** 2        
       
		
    def k_in_row(self, board, move, player, delta_x_y):
        """Return true if there is a line through move on board for player.
        hint: This function can be extended to test of n number of items on a line 
        not just self.k items as it is now. """
        (delta_x, delta_y) = delta_x_y
        x, y = move
        n = 0  # n is number of moves in row
        while board.get((x, y)) == player:
            n += 1
            x, y = x + delta_x, y + delta_y
        x, y = move
        while board.get((x, y)) == player:
            n += 1
            x, y = x - delta_x, y - delta_y
        n -= 1  # Because we counted move itself twice
        return n >= self.k


    #made changes to this function so the chances list does not stay empty
    def chances(self, state):
        """Return a list of all possible moves."""
        chances = []
        for m in state.moves:
            chances.append(m)
        return chances
    
class Gomoku(TicTacToe):
    """Also known as Five in a row."""

    def __init__(self, h=15, v=16, k=5):
        TicTacToe.__init__(self, h, v, k)
