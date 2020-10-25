import unittest
from game import Game

class TestGameMethods(unittest.TestCase):
    def test_validateMove(self):
        game = Game()
        with self.assertRaises(Exception):
            game.move([-1,0])
        with self.assertRaises(Exception):
            game.move([0,0])
            game.move([0,0])
    
    def test_gameLogic(self):
        game = Game()
        self.assertTrue(game.isBlackMove())
        self.assertFalse(game.isEnd())
        game.move([0,0])
        self.assertFalse(game.isBlackMove())
        self.assertFalse(game.isEnd())
        game.move([0,1])
        black_moves = [(y+1, y) for y in range(5)]
        white_moves = [(y+2, y) for y in range(5)]
        for i in range(4):
            game.move(black_moves[i])
            self.assertFalse(game.isBlackMove())
            self.assertFalse(game.isEnd())
            game.move(white_moves[i])
            self.assertTrue(game.isBlackMove())
            self.assertFalse(game.isEnd())
        game.move(black_moves[4])
        self.assertTrue(game.isEnd())
        self.assertTrue(game.isBlackWin())


        

if __name__ == '__main__':
    unittest.main()