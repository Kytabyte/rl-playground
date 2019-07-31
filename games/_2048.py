import random
import numpy as np

UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3

class Play2048():
    def __init__(self, size=4):
        self._size = size
        self._move_func = {
            UP: self._move_up,
            DOWN: self._move_down,
            LEFT: self._move_left,
            RIGHT: self._move_right
        }
        self._is_terminate = False
        self._set_game()
    
    def _set_game(self):
        self._grid = [([0] * self._size) for _ in range(self._size)]
        self._score = 0
        self._available = set(i for i in range(self._size ** 2))
        self._fill()
        self._fill()
    
    def __str__(self):
        ret = 'Panel:\n' 
        for row in self._grid:
            ret += ' {}\n'.format(row)
        ret += ('\nScore:\n{}\n'.format(self._score))
        ret += '\nAvailable Grid:\n'
        if len(self._available) == 0:
            ret += 'None'
        else:
            prev = -1
            for num in self._available:
                if (prev > -1 and prev // 4 != num // 4):
                    ret += '\n'
                ret += ('({}, {})'.format(num // self._size, str(num % self._size)))  
                prev = num
        return ret
    
    def __repr__(self):
        return self.__str__()
        
    def _fill(self):
        idx = random.sample(self._available,1)[0]
        num = 2 if random.random() < 0.9 else 4
        self._grid[idx // self._size][idx % self._size] = num
        self._available.remove(idx)
        self._is_terminate = self._check_terminate()
        
    def _check_terminate(self):
        if len(self._available) != 0:
            return False
        for i in range(self._size):
            for j in range(self._size):
                if i < self._size-1 and j < self._size-1:
                    if self._grid[i][j] == self._grid[i][j+1] or self._grid[i][j] == self._grid[i+1][j]:
                        return False
                elif i == self._size-1 and j < self._size-1:
                    if self._grid[i][j] == self._grid[i][j+1]:
                        return False
                elif i < self._size-1 and j == self._size-1:
                    if self._grid[i][j] == self._grid[i+1][j]:
                        return False
        return True
        
    def _can_move(self, direction):
        if direction == LEFT:
            for i in range(self._size):
                for j in range(self._size-1):
                    if (self._grid[i][j] == 0 and self._grid[i][j+1] > 0) or \
                    (self._grid[i][j] != 0 and self._grid[i][j] == self._grid[i][j+1]):
                        return True
        elif direction == RIGHT:
            for i in range(self._size):
                for j in range(self._size-1):
                    if (self._grid[i][j+1] == 0 and self._grid[i][j] > 0) or \
                    (self._grid[i][j] != 0 and self._grid[i][j] == self._grid[i][j+1]):
                        return True
        elif direction == UP:
            for i in range(self._size):
                for j in range(self._size-1):
                    if (self._grid[j][i] == 0 and self._grid[j+1][i]) or \
                    (self._grid[j][i] != 0 and self._grid[j][i] == self._grid[j+1][i]):
                        return True
        elif direction == DOWN:
            for i in range(self._size):
                for j in range(self._size-1):
                    if (self._grid[j+1][i] == 0 and self._grid[j][i]) or \
                    (self._grid[j][i] != 0 and self._grid[j+1][i] == self._grid[j][i]):
                        return True
        else:
            raise Exception("Invalid move")
        return False
                
    def _merge(self, before_move):
        if len(before_move) < 2:
            return before_move.copy()
        
        after_move = []
        i = 0     
        while i < len(before_move):
            if i < len(before_move)-1 and before_move[i] == before_move[i+1]:
                after_move.append(before_move[i] * 2)
                self._score += (before_move[i] * 2)
                i += 2
            else:
                after_move.append(before_move[i])
                i += 1
        
        return after_move
    
    def _move_left(self):
        for i in range(self._size):
            before_move = []
            for j in range(self._size):
                if self._grid[i][j] != 0:
                    before_move.append(self._grid[i][j])
            after_move = self._merge(before_move)
            for j in range(len(after_move)):
                if i * self._size + j in self._available:
                    self._available.remove(i * self._size + j)
                self._grid[i][j] = after_move[j]
            for j in range(len(after_move), self._size):
                if self._grid[i][j] != 0:
                    self._grid[i][j] = 0
                self._available.add(i * self._size + j)
        
    
    def _move_right(self):
        for i in range(self._size):
            before_move = []
            for j in reversed(range(self._size)):
                if self._grid[i][j] != 0:
                    before_move.append(self._grid[i][j])
            after_move = self._merge(before_move)
            for j in reversed(range(self._size - len(after_move), self._size)):
                if i * self._size + j in self._available:
                    self._available.remove(i * self._size + j)
                self._grid[i][j] = after_move[self._size-j-1]
            for j in reversed(range(self._size-len(after_move))):
                if self._grid[i][j] != 0:
                    self._grid[i][j] = 0
                self._available.add(i * self._size + j)
    
    def _move_up(self):
        for i in range(self._size):
            before_move = []
            for j in range(self._size):
                if self._grid[j][i] != 0:
                    before_move.append(self._grid[j][i])
            after_move = self._merge(before_move)
            for j in range(len(after_move)):
                if j * self._size + i in self._available:
                    self._available.remove(j * self._size + i)
                self._grid[j][i] = after_move[j]
            for j in range(len(after_move), self._size):
                if self._grid[j][i] != 0:
                    self._grid[j][i] = 0
                self._available.add(j * self._size + i) 
    
    def _move_down(self):
        for i in range(self._size):
            before_move = []
            for j in reversed(range(self._size)):
                if self._grid[j][i] != 0:
                    before_move.append(self._grid[j][i])  
            after_move = self._merge(before_move)
            for j in reversed(range(self._size-len(after_move), self._size)):
                if j * self._size + i in self._available:
                    self._available.remove(j * self._size + i)
                self._grid[j][i] = after_move[self._size-j-1]
            
            for j in reversed(range(self._size-len(after_move))):
                if self._grid[j][i] != 0:
                    self._grid[j][i] = 0
                self._available.add(j * self._size + i)
        
    def move(self, direction):
        if self._can_move(direction):
            self._move_func[direction]()
            self._fill()

    def can_move(self):
        return [self._can_move(direction) for direction in sorted(self._move_func.keys())]
    
    def status(self):
        return np.array(self._grid)
    
    def score(self):
        return self._score
    
    def reset(self):
        self._set_game()
        
    def is_terminate(self):
        return self._is_terminate
        
    def _terminate(self):
        print('Game Over!\nYour score is ' + str(self._score) + '\n')
        raise Exception()