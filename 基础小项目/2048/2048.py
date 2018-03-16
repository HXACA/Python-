#-*- coding:utf-8 _*-
"""
@author:KING
@file: 2048.py
@time: 2018/03/16
"""

import curses
#图形函数库，用于绘制界面
from random import randrange,choice
from collections import defaultdict
#集合模块

letter_codes = [ord(ch) for ch in 'WASDRQwasdrq']
actions = ['Up', 'Left', 'Down', 'Right', 'Restart', 'Exit']
#ord():返回对于ASCII值
actions_dict = dict(zip(letter_codes, actions * 2))
#dict：构造字典，zip（）：打包为元组

def get_user_action(keyboard):
    char = "N"
    while char not in actions_dict:
        char = keyboard.getch()
    return actions_dict[char]

def transpose(field):
    #转置
    return [list(row) for row in zip(*field)]
    #zip（*field):可变参数传递的使用

def invert(field):
    #逆向转置
    return [row[::-1] for row in field]
    #row[::-1]：将row内容翻转 1234->4321

class GameField(object):
    def __init__(self, height=4, width=4, win=2048):
        self.height = height
        self.width = width
        self.win_value = win
        self.score = 0
        self.highscore = 0
        self.reset()

    def reset(self):
        if self.score > self.highscore:
            self.highscore = self.score
        self.score = 0
        self.field = [[0 for i in range(self.width)] for j in range(self.height)]
        self.spawn()
        self.spawn()

    def move(self, direction):
        def move_row_left(row):
            def tighten(row):
                new_row = [i for i in row if i != 0]
                # 取出非0
                new_row += [0 for i in range(len(row) - len(new_row))]
                # 补0
                return new_row

            def merge(row):
                pair = False
                new_row = []
                for i in range(len(row)):
                    if pair:
                        new_row.append(2 * row[i])
                        self.score += 2 * row[i]
                        pair = False
                    else:
                        if i + 1 < len(row) and row[i] == row[i + 1]:
                            pair = True
                            new_row.append(0)
                        else:
                            new_row.append(row[i])
                assert len(new_row) == len(row)
                # 断言 判断是否正确
                return new_row
            # 先挤一起，再合并，再挤一起
            return tighten(merge(tighten(row)))

        # 利用转置，翻转，实现其余三方向的操作
        moves = {}
        moves['Left'] = lambda field: \
            [move_row_left(row) for row in field]
        moves['Right'] = lambda field: \
            invert(moves['Left'](invert(field)))
        moves['Up'] = lambda field: \
            transpose(moves['Left'](transpose(field)))
        moves['Down'] = lambda field: \
            transpose(moves['Right'](transpose(field)))

        # 具体移动
        if direction in moves:
            if self.move_is_possible(direction):
                self.field = moves[direction](self.field)
                self.spawn()
                return True
            else:
                return False

    def is_win(self):
        return any(any(i>=self.win_value for i in row) for row in self.field)

    def is_gameover(self):
        return not any(self.move_is_possible(move) for move in actions)

    def draw(self,screen):
        help_string1 = 'WASD Move'
        help_string2 = '(R) Restart (Q)Quit'
        gameover_string = 'GameOver'
        win_string = 'You Win'

        def cast(string):
            screen.addstr(string+'\n')

        def draw_hor_separator():
            #绘制水平分割线
            line = '+' + ('+------' * self.width + '+')[1:]
            separator = defaultdict(lambda: line)
            if not hasattr(draw_hor_separator,"counter"):
                draw_hor_separator.counter = 0
            cast(separator[draw_hor_separator.counter])
            draw_hor_separator.counter += 1

        def draw_row(row):
            cast(''.join('|{: ^5} '.format(num) if num > 0 else '|      ' for num in row) + '|')
        screen.clear()
        cast('score:'+str(self.score))
        if 0!=self.highscore:
            cast('highScore'+str(self.highscore))

        for row in self.field:
            draw_hor_separator()
            draw_row(row)
        draw_hor_separator()

        if self.is_win():
            cast(win_string)
        else:
            if self.is_gameover():
                cast(gameover_string)
            else:
                cast(help_string1)
        cast(help_string2)

    def spawn(self):
        #随机生成2或者4
        new_element = 4 if randrange(100) >89 else 2
        (i,j) = choice([(i,j) for i in range(self.width) for j in range(self.height) if self.field[i][j]==0])
        #choice 返回一个列表，元组或字符串的随机项
        self.field[i][j] = new_element

    def move_is_possible(self,direction):
        def row_is_left_moveable(row):
            def change(i):
                if row[i]==0 and row[i+1]!=0:
                    return True
                if row[i]!=0 and row[i] == row[i+1]:
                    return True
                return False
            return any(change(i) for i in range(len(row)-1))
            #any() 函数用于判断给定的可迭代参数 iterable 是否全部为空对象，如果都为空、0、false，则返回 False，
            # 如果不都为空、0、false，则返回 True。
        check = {}
        check['Left'] = lambda field:\
            any(row_is_left_moveable(row) for row in field)
        check['Right'] = lambda field:\
            check['Left'](invert(field))
        check['Up'] = lambda field:\
            check['Left'](transpose(field))
        check['Down'] = lambda field:\
            check['Right'](transpose(field))

        if direction in check:
            return check[direction](self.field)
        else:
            return False



def main(stdscr):
    def init():
        #重置棋盘
        game_field.reset()
        return 'Game'

    def not_game(state):
        #游戏结束，绘制出结果
        #获得游戏状态，Restart/Exit
        game_field.draw(stdscr)
        action = get_user_action(stdscr)
        response = defaultdict(lambda :state)#未定义的字典，返回当前状态
        response['Restart'],response['Exit'] = 'Init','Exit'
        return response[action]

    def game():
        game_field.draw(stdscr)
        action = get_user_action(stdscr)
        if action == 'Restart':
            return 'Init'
        if action == 'Exit':
            return 'Exit'
        if game_field.move(action):
            if game_field.is_win():
                return 'Win'
            if game_field.is_gameover():
                return 'Gameover'
        return 'Game'

    state_actions = {
        'Init': init,
        'Win': lambda: not_game('Win'),
        'Gameover': lambda: not_game('Gameover'),
        'Game': game
    }

    curses.use_default_colors()
    game_field = GameField(win=2048)

    state = 'Init'
    while state!='Exit':
        state = state_actions[state]()

curses.wrapper(main)

