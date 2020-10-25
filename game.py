import numpy as np
from collections import defaultdict

DEFAULT_SHAPE = (19, 19)
directions = [(0,1), (1,1), (1,0), (1,-1)]

def is_in_range(position, shape):
    if len(position) != len(shape):
        return False
    for p, s in zip(position, shape):
        if p<0 or p>=s:
            return False
    return True

class State:
    def __init__(self, 
            shape = None,
            board = None,
            isNextBlack = None,
            validMoves = None,
            consecutives = None,
            isEnd = None,
            endResult = None):
        self.shape = shape
        self.board = board
        self.isNextBlack = isNextBlack
        self.validMoves = validMoves
        self.consecutives = consecutives
        self.isEnd = isEnd
        self.endResult = endResult

    @staticmethod
    def init(shape = DEFAULT_SHAPE):
        shape = tuple(shape)
        return State(
            shape = shape,
            board = np.zeros((2,) + shape, dtype=bool),
            isNextBlack = True,
            validMoves = np.ones(shape, dtype=bool),
            consecutives = np.zeros((2, 4) + shape, dtype=int),
            isEnd = False,
            endResult = 0,
        )

    def copy(self):
        return State(
            shape = self.shape,
            board = self.board.copy(),
            isNextBlack = self.isNextBlack,
            validMoves = self.validMoves.copy(),
            consecutives = self.consecutives.copy(),
            isEnd = self.isEnd,
            endResult = self.endResult,
        )

def update_consecutive(board, consecutives, position):
    max_len = 0
    for i, d in enumerate(directions):
        prev_pos = _list_subtract(position, d)
        consecutive = consecutives[i]
        prev = 0 if not is_in_range(prev_pos, board.shape) else consecutive[prev_pos]
        l = _update_consecutive_dir(board, consecutive, position, d, prev)
        max_len = max(max_len, l)
    return max_len

def _list_add(position, delta):
    return tuple(p + d for p, d in zip(position, delta))

def _list_subtract(position, delta):
    return tuple(p - d for p, d in zip(position, delta))

def _update_consecutive_dir(board, consecutive, position, direction, prev):
    if not is_in_range(position, board.shape):
        return 0
    if board[position] == 0:
        return 0
    c = prev + 1
    consecutive[position] = c
    next = _list_add(position, direction)
    return max(_update_consecutive_dir(board, consecutive, next, direction, c), c)

def transition(state, action):
    assert(state.validMoves[action])
    next_state = state.copy()
    next_state.isNextBlack = not next_state.isNextBlack
    index = 0 if state.isNextBlack else 1

    # update board configuration
    next_state.board[index][action] = 1

    # update consecutive 
    max_con = update_consecutive(next_state.board[index], next_state.consecutives[index], action)

    # update validMoves
    next_state.validMoves = np.logical_not(np.logical_or(next_state.board[0], next_state.board[1]))

    # check ending
    if max_con == 5:
        next_state.isEnd = True
        next_state.endResult = 1 - 2*index
    elif not np.any(next_state.validMoves):
        next_state.isEnd = True
        next_state.endResult = 0

    return next_state


def validate_state(state):
    assert(type(state) is State)
    black = state.black
    white = state.white
    width = state.width
    height = state.height

    assert(len(black.shape) == 2)
    assert(black.shape == [height, width])
    assert(white.shape == [height, width])
    # white and black not overlap
    # if( #black == #white) => isNextBlack = True
    # elif( #black == #white + 1) => isNextBlack = False
    # else => raise


class Game:
    def __init__(self):
        self.dimension = 2
        self.board_size = [15, 15]
        self.directions = self._generateDirections()
        self.win_len = 5
        self.reset()

    def isBlackMove(self) -> bool:
        return self.is_next_black

    def isEnd(self):
        return self.is_end
    
    def reset(self):
        self.is_next_black = True
        self.is_end = False
        self.is_black_win = None
        self.board = defaultdict(int)
        self.length_continuous = defaultdict(int)

    def isBlackWin(self):
        return self.is_black_win
    def getBoardDict(self):
        return dict(self.board)

    def getBoardList(self):
        ret = [[0] * self.board_size[1] for _ in range(self.board_size[0]) ]
        for i in range(len(ret)):
            for j in range(len(ret[0])):
                ret[i][j] = self.board[(i,j)]
        return ret
        
    def getState(self):
        grid = self.getBoardList()
        return dict(
            width = self.board_size[0],
            height = self.board_size[1],
            board = grid,
            is_next_black = self.is_next_black,
            is_end = self.is_end,
            is_black_win = self.is_black_win,
            error = "",
        )


    def getBoardNumpy(self):
        board = np.zeros(self.board_size, dtype=int)
        for pos, val in self.board.items():
            board[pos] = val
        return board

    def move(self, position) -> bool:
        self._validateMove(position)
        position = tuple(position)
        self.board[position] = 1 if self.is_next_black else -1
        self.is_next_black = not self.is_next_black
        self._updateContinuous(position)
        return self.is_end

    def _generateDirections(self):
        result = []
        self._dfsGenerateDirections(result, 0, [], True, self.dimension)
        return {index:direction for index, direction in zip(range(len(result)), result)}

    def _dfsGenerateDirections(self, result, i, current, is_leading_zero, dimension):
        if i == dimension:
            if not is_leading_zero:
                result.append(tuple(current))
            return
        delta_list = [0, 1] if is_leading_zero else [0, 1, -1]
        for delta in delta_list:
            current.append(delta)
            self._dfsGenerateDirections(result, i+1, current, is_leading_zero and delta==0, dimension)
            current.pop()

    def _updateContinuous(self, position):
        if self.board[position] == 0:
            return 
        this_value = self.board[position]
        for i, delta in self.directions.items():
            old_len = self.length_continuous[(i, position)]
            prev_position = self._positionSubtract(position, delta)
            if self.board[prev_position] == this_value:
                new_len = self.length_continuous[(i, prev_position)] + 1
            else:
                new_len = 1
            self.length_continuous[(i, position)] = new_len
            if old_len != new_len:
                next_position = self._positionAdd(position, delta)
                self._updateContinuous(next_position)
            if new_len == self.win_len:
                self.is_end = True
                self.is_black_win = this_value == 1

    
    def _positionAdd(self, position, delta):
        return tuple(p + d for p, d in zip(position, delta))

    def _positionSubtract(self, position, delta):
        return tuple(p - d for p, d in zip(position, delta))

    def _validateMove(self, position):
        if self.isEnd():
            raise Exception("game has ended")
        if type(position) not in (list, tuple):
            raise Exception("position should be of type list or tuple")
        position = tuple(position)
        if len(position) != self.dimension:
            raise Exception("position has wrong dimension")
        for x, length in zip(position, self.board_size):
            if type(x) != int:
                raise Exception("position should be all int")
            if x <0 or x >= length:
                raise Exception("position is out of range")
        if self.board[position] != 0:
            raise Exception("this position is occupied")

if __name__ == "__main__":
    s = State.init()
    print(s.board)
    s = transition(s, (7,7))
    print(s.board)
    s = transition(s, (7,8))
    print(s.board)
    print(s.validMoves)
    print(s.consecutives)
