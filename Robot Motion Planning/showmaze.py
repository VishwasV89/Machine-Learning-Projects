from maze import Maze
import turtle
import sys

class Showmaze(object):
    def __init__(self, maze_dim, tracer):
        '''
        This function uses Python's turtle library to draw a picture of the maze
        given as an argument when running the script.
        '''

        # Create a maze based on input argument on command line.
        testmaze = maze_dim

        # Intialize the window and drawing turtle.
        window = turtle.Screen()
        wally = turtle.Turtle()
        wally.speed(0)
        wally.hideturtle()
        wally.penup()

        window.addshape("images/cheese.gif")
        cheezy = turtle.Turtle()
        cheezy.shape("images/cheese.gif")
        cheezy.showturtle()

        window.addshape("images/mouse.gif")
        penny = turtle.Turtle()
        penny.hideturtle()
        penny.speed(0)
        penny.shape("images/mouse.gif")
        penny.color("green")
        penny.pensize(2)
        penny.penup()

        # maze centered on (0,0), squares are 40 units in length.
        sq_size = 40
        origin = testmaze.dim * sq_size / -2

        penny.goto(origin + sq_size / 2, origin + sq_size / 2)
        penny.showturtle()
        penny.setheading(90)
        penny.pendown()

        # iterate through squares one by one to decide where to draw walls
        for x in range(testmaze.dim):
            for y in range(testmaze.dim):
                if not testmaze.is_permissible([x,y], 'up'):
                    wally.goto(origin + sq_size * x, origin + sq_size * (y+1))
                    wally.setheading(0)
                    wally.pendown()
                    wally.forward(sq_size)
                    wally.penup()

                if not testmaze.is_permissible([x,y], 'right'):
                    wally.goto(origin + sq_size * (x+1), origin + sq_size * y)
                    wally.setheading(90)
                    wally.pendown()
                    wally.forward(sq_size)
                    wally.penup()

                # only check bottom wall if on lowest row
                if y == 0 and not testmaze.is_permissible([x,y], 'down'):
                    wally.goto(origin + sq_size * x, origin)
                    wally.setheading(0)
                    wally.pendown()
                    wally.forward(sq_size)
                    wally.penup()

                # only check left wall if on leftmost column
                if x == 0 and not testmaze.is_permissible([x,y], 'left'):
                    wally.goto(origin, origin + sq_size * y)
                    wally.setheading(90)
                    wally.pendown()
                    wally.forward(sq_size)
                    wally.penup()

        for i in range(len(tracer)):
            if tracer[i] == 'r':
                penny.right(90)
            elif tracer[i] == 'l':
                penny.left(90)
            elif tracer[i] == 'f':
                penny.forward(sq_size)
            elif tracer[i] == 'b':
                penny.backward(sq_size)

        window.exitonclick()