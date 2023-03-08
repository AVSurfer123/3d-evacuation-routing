import pygame
import math
from queue import PriorityQueue


pygame.display.set_caption("A* Path Finding Algorithm")

# set colors for cells
YELLOW = (255, 255, 0)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
PURPLE = (128, 0, 128)
ORANGE = (255, 165 ,0)
GREY = (128, 128, 128)
TURQUOISE = (64, 224, 208)

class Spot:
    def __init__(self, row, col, width, total_rows):
        self.row = row
        self.col = col
        self.x = row * width
        self.y = col * width
        self.color = WHITE
        self.neighbors = []
        self.width = width
        self.total_rows = total_rows

    def get_pos(self):
        return self.row, self.col

    def is_barrier(self):
        return self.color == BLACK

    def is_start(self):
        return self.color == ORANGE

    def is_end(self):
        return self.color == TURQUOISE

    def reset(self):
        self.color = WHITE

    def make_start(self):
        self.color = ORANGE

    def make_barrier(self):
        self.color = BLACK

    def make_end(self):
        self.color = TURQUOISE

    def make_path(self):
        self.color = PURPLE

    def draw(self, win):
        pygame.draw.rect(win, self.color, (self.x, self.y, self.width, self.width))

    def __lt__(self, other):
        return False

def make_grid(rows, width):
    grid = []
    gap = width // rows
    for i in range(rows):
        grid.append([])
        for j in range(rows):
            spot = Spot(i, j, gap, rows)
            grid[i].append(spot)

    return grid


def draw_grid(win, rows, width):
    gap = width // rows
    for i in range(rows):
        pygame.draw.line(win, GREY, (0, i * gap), (width, i * gap))
        for j in range(rows):
            pygame.draw.line(win, GREY, (j * gap, 0), (j * gap, width))

# draw the grid
def draw(win, grid, rows, width):
    win.fill(WHITE)

    for row in grid:
        for spot in row:
            spot.draw(win)

    draw_grid(win, rows, width)
    pygame.display.update()


    
def main():
    # create size of grid <== connect to other file
    width = 800
    length = width
    win = pygame.display.set_mode((width, length))
    rows = 10
    grid = make_grid(rows, width)
    start = None
    end = None
    run = True
    while run:
        draw(win, grid, rows, width)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False

            # set this to the start, and, and wall cells <== connect to other file
            start_cell = (3,3)
            end_cells = [(1,2),(5,5)]
            wall_cells = [(0,0),(1,1),(2,2)]
            
            path_cells = [(2,3),(1,3)]
            
            # setting the start
            start = grid[start_cell[0]][start_cell[1]]
            start.make_start()
            for cell in end_cells:
                end = grid[cell[0]][cell[1]]
                end.make_end()
            for cell in wall_cells:
                wall = grid[cell[0]][cell[1]]
                wall.make_barrier()
            for cell in path_cells:
                path = grid[cell[0]][cell[1]]
                path.make_path()
            

    pygame.quit()

main()