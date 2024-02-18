"""
Tic Tac Toe Player
"""

import math

X = "X"
O = "O"
EMPTY = None


def initial_state():
    """
    Returns starting state of the board.
    """
    return [[EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY]]


def player(board):
    """
    Returns player who has the next turn on a board.
    """
    num_X, num_O = 0, 0
    for line in board:
        for cell in line:
            if cell == X:
                num_X += 1
            if cell == O:
                num_O += 1
    if num_X == num_O:
        return X
    return O
    raise NotImplementedError


def actions(board):
    """
    Returns set of all possible actions (i, j) available on the board.
    """
    possible_actions = set()
    for i in range(len(board)):
        for j in range(len(board[i])):
            if board[i][j] == EMPTY:
                possible_actions.add((i, j))
    return possible_actions

    raise NotImplementedError


def result(board, action):
    """
    Returns the board that results from making move (i, j) on the board.
    """
    if action is None:
        return board
    if board[action[0]][action[1]] != EMPTY:
        raise moveTaken
    if not (0 <= action[0] < 3 and 0 <= action[1] < 3):
        raise outOfBoundMove

    current_player = player(board)
    new_board_state = [row[:] for row in board]
    new_board_state[action[0]][action[1]] = current_player
    return new_board_state


def winner(board):
    """
    Returns the winner of the game, if there is one.
    """
    # Check rows
    for row in board:
        if all(cell == 'X' for cell in row):
            return 'X'
        elif all(cell == 'O' for cell in row):
            return 'O'

    # Check columns
    for col in range(3):
        if all(board[row][col] == 'X' for row in range(3)):
            return 'X'
        elif all(board[row][col] == 'O' for row in range(3)):
            return 'O'

    # Check diagonals
    if all(board[i][i] == 'X' for i in range(3)) or all(board[i][2 - i] == 'X' for i in range(3)):
        return 'X'
    elif all(board[i][i] == 'O' for i in range(3)) or all(board[i][2 - i] == 'O' for i in range(3)):
        return 'O'

    # If no winner or tie yet
    return None
    raise NotImplementedError


def terminal(board):
    """
    Returns True if game is over, False otherwise.
    """
    if all(cell != EMPTY for row in board for cell in row) or winner(board) is not None:
        return True
    return False
    raise NotImplementedError


def utility(board):
    """
    Returns 1 if X has won the game, -1 if O has won, 0 otherwise.
    """
    if winner(board) == X:
        return 1
    if winner(board) == O:
        return -1
    return 0
    raise NotImplementedError

def minimax(board):
    """
    Returns the optimal action for the current player on the board.
    """
    return minimax_recursive(board, player(board))[0]
    raise NotImplementedError


def minimax_recursive(board, current_player):
    """
    Returns the optimal action, the ultility for the current player on the board.
    """
    if terminal(board):
        return None, utility(board)

    if current_player == X:
        max_eval = -math.inf
        best_action = None

        for action in actions(board):
            new_board_state = result(board, action)
            _, eval = minimax_recursive(new_board_state, O)
            
            if eval > max_eval:
                max_eval = eval
                best_action = action

        return best_action, max_eval

    else:  # current_player == O
        min_eval = math.inf
        best_action = None

        for action in actions(board):
            new_board_state = result(board, action)
            _, eval = minimax_recursive(new_board_state, X)
            
            if eval < min_eval:
                min_eval = eval
                best_action = action

        return best_action, min_eval
