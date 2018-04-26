import numpy as np
import random

dir_sensors = {'u': ['l', 'u', 'r'], 'r': ['u', 'r', 'd'],
               'd': ['r', 'd', 'l'], 'l': ['d', 'l', 'u'],
               'up': ['l', 'u', 'r'], 'right': ['u', 'r', 'd'],
               'down': ['r', 'd', 'l'], 'left': ['d', 'l', 'u']}
dir_move = {'u': [0, 1], 'r': [1, 0], 'd': [0, -1], 'l': [-1, 0],
            'up': [0, 1], 'right': [1, 0], 'down': [0, -1], 'left': [-1, 0]}
dir_reverse = {'u': 'd', 'r': 'l', 'd': 'u', 'l': 'r',
               'up': 'd', 'right': 'l', 'down': 'u', 'left': 'r'}
dir_rotation = {'u': {'l':-90, 'u':0, 'r':90}, 'r': {'u':-90, 'r':0, 'd':90},
               'd': {'r':-90, 'd':0, 'l':90}, 'l': {'d':-90, 'l':0, 'u':90},
               'up': {'l':-90, 'u':0, 'r':90}, 'right': {'u':-90, 'r':0, 'd':90},
               'down': {'r':-90, 'd':0, 'l':90}, 'left': {'d':-90, 'l':0, 'u':90}}

class Robot(object):
    def __init__(self, maze_dim):
        '''
        Use the initialization function to set up attributes that your robot
        will use to learn and navigate the maze. Some initial attributes are
        provided based on common information, including the size of the maze
        the robot is placed in.
        '''
        self.heading = 'up'
        self.maze_dim = maze_dim
        self.location = [0, 0]
        self.run = 0
        self.closed = [[0]*maze_dim for col in range(maze_dim)] # grid for keeping track of cells mapped while navigation
        self.explored = [[0]*maze_dim for col in range(maze_dim)] # grid for keeping track of visited cells while performing search in the second run
        self.maze_grid = [[0]*maze_dim for col in range(maze_dim)] # grid for the mapped maze storing walls and open space
        self.path_grid = [[-1]*maze_dim for col in range(maze_dim)] # grid for keeping track of action performed to reach a cell
        self.policy = [[' ']*maze_dim for col in range(maze_dim)] # grid for the optimal policy to the goal
        self.dist = [[float('inf')]*maze_dim for col in range(maze_dim)] # distance grid for dijkstra algorithm
        self.heuristics = [[-1]*maze_dim for col in range(maze_dim)] # heuristic grid for A-Star algorithm
        self.value = [[99]*maze_dim for col in range(maze_dim)] # value grid for the no of steps to the goal from each cell
        self.new_grid = [[0]*maze_dim for col in range(maze_dim)] # placeholder grid to print other grids for proper view in the shell
        self.goal_bounds = [maze_dim/2 - 1, maze_dim/2] # goal boundary
        self.goal_found = False
        self.search = False
        self.steps = 0

    def next_move(self, sensors):
        '''
        Use this function to determine the next move the robot should make,
        based on the input from the sensors after its previous move. Sensor
        inputs are a list of three distances from the robot's left, front, and
        right-facing sensors, in that order.

        Outputs should be a tuple of two values. The first value indicates
        robot rotation (if any), as a number: 0 for no rotation, +90 for a
        90-degree rotation clockwise, and -90 for a 90-degree rotation
        counterclockwise. Other values will result in no rotation. The second
        value indicates robot movement, and the robot will attempt to move the
        number of indicated squares: a positive number indicates forwards
        movement, while a negative number indicates backwards movement. The
        robot may move a maximum of three units per turn. Any excess movement
        is ignored.

        If the robot wants to end a run (e.g. during the first training run in
        the maze) then returing the tuple ('Reset', 'Reset') will indicate to
        the tester to end the run and return the robot to the start.
        '''
        # keep a count of steps
        self.steps += 1

        # change sensor values to 0 and 1; 0 for close and 1 for open
        for i in range(len(sensors)):
            if sensors[i] > 0:
                sensors[i] = 1

        if self.run == 1:
            rotation, movement = self.optimized_robot()

        if self.run == 0:
            rotation, movement = self.explorer_robot(sensors)

        return rotation, movement

    def explorer_robot(self, sensors):
        '''
        Uses random movement algorithm to determine next action to explore and map the maze using sensor values as input. 
        '''
        x, y = self.location
        self.wall_finder(sensors) # function to calculate and store the wall value for the current location

        # Check if the goal has been reached
        if self.location[0] in self.goal_bounds and self.location[1] in self.goal_bounds:
            self.goal_found = True

        # Check for the dead ends, block them by putting -1 in the maze map and go reverse
        if self.heading in ['up', 'u'] and self.maze_grid[x][y] == 4: # for heading 'up' cell value 4 is a dead end
            self.maze_grid[x][y] = -1
            movement = -1
            self.location[0] += dir_move[dir_reverse[self.heading]][0]
            self.location[1] += dir_move[dir_reverse[self.heading]][1]
            if self.maze_grid[x][y-1] == 5: # prior cell with only one possible action leading to the dead end
                self.maze_grid[x][y-1] = -1
                movement = -2
                self.location[0] += dir_move[dir_reverse[self.heading]][0]
                self.location[1] += dir_move[dir_reverse[self.heading]][1]
            rotation = 0

        elif self.heading in ['right', 'r'] and self.maze_grid[x][y] == 8: # for heading 'right' cell value 8 is a dead end
            self.maze_grid[x][y] = -1
            movement = -1
            self.location[0] += dir_move[dir_reverse[self.heading]][0]
            self.location[1] += dir_move[dir_reverse[self.heading]][1]
            if self.maze_grid[x-1][y] == 10: # prior cell with only one possible action leading to the dead end
                self.maze_grid[x-1][y] = -1
                movement = -2
                self.location[0] += dir_move[dir_reverse[self.heading]][0]
                self.location[1] += dir_move[dir_reverse[self.heading]][1]
            rotation = 0

        elif self.heading in ['down', 'd'] and self.maze_grid[x][y] == 1: # for heading 'down' cell value 1 is a dead end
            self.maze_grid[x][y] = -1
            movement = -1
            self.location[0] += dir_move[dir_reverse[self.heading]][0]
            self.location[1] += dir_move[dir_reverse[self.heading]][1]
            if self.maze_grid[x][y+1] == 5: # prior cell with only one possible action leading to the dead end
                self.maze_grid[x][y+1] = -1
                movement = -2
                self.location[0] += dir_move[dir_reverse[self.heading]][0]
                self.location[1] += dir_move[dir_reverse[self.heading]][1]
            rotation = 0
            self.maze_grid[0][0] = 1 # in case starting cell is marked -1
            self.maze_grid[0][1] = 5 # in case starting cell is marked -1

        elif self.heading in ['left', 'l'] and self.maze_grid[x][y] == 2: # for heading 'left' cell value 2 is a dead end
            self.maze_grid[x][y] = -1
            movement = -1
            self.location[0] += dir_move[dir_reverse[self.heading]][0]
            self.location[1] += dir_move[dir_reverse[self.heading]][1]
            if self.maze_grid[x+1][y] == 10: # prior cell with only one possible action leading to the dead end
                self.maze_grid[x+1][y] = -1
                movement = -2
                self.location[0] += dir_move[dir_reverse[self.heading]][0]
                self.location[1] += dir_move[dir_reverse[self.heading]][1]
            rotation = 0

        # perform rotation and movement if the robot is not at a dead end
        else:
            actions = []
            # select valid actions based on heading and sensor readings
            for i in range(len(sensors)):
                if sensors[i] == 1:
                    if self.maze_grid[x + dir_move[dir_sensors[self.heading][i]][0]]\
                                    [y + dir_move[dir_sensors[self.heading][i]][1]] != -1: # restricting actions leading to blocked deadends
                        actions.append(dir_sensors[self.heading][i])

            # if robot is heading to a blocked cell, reverse and block the path
            if len(actions) == 0:
                rotation = 0
                movement = -1
                action = dir_reverse[self.heading]
                self.maze_grid[x][y] = -1
                self.location[0] += dir_move[action][0]
                self.location[1] += dir_move[action][1]
                if self.maze_grid[x + dir_move[action][0]][y + dir_move[action][1]] in [5, 10]:
                    rotation = 0
                    movement = -2
                    self.maze_grid[x + dir_move[action][0]][y + dir_move[action][1]] = -1
                    self.location[0] += dir_move[action][0]
                    self.location[1] += dir_move[action][1]
                    if self.maze_grid[x + dir_move[action][0]*2][y + dir_move[action][1]*2] in [5, 10]:
                        rotation = 0
                        movement = -3
                        self.maze_grid[x + dir_move[action][0]*2][y + dir_move[action][1]*2] = -1
                        self.location[0] += dir_move[action][0]
                        self.location[1] += dir_move[action][1]

            # move to open space
            else:
                action = random.choice(actions) # randomly choose an action

                # if cell is not explored perform action
                if self.closed[x + dir_move[action][0]][y + dir_move[action][1]] == 0:

                    rotation = dir_rotation[self.heading][action] # perform rotation based on action
                    self.heading = action # new heading after rotation, same as action
                    movement = 1

                    self.location[0] += dir_move[self.heading][0]
                    self.location[1] += dir_move[self.heading][1]

                # if cell is already explored
                elif self.closed[x + dir_move[action][0]][y + dir_move[action][1]] == 1:
                    # select an action leading to unexplored cell
                    # if all actions lead to explored cells, keep the previous action
                    if len(actions)>1:
                        for i in range(len(actions)):
                            if  self.closed[x + dir_move[actions[i]][0]][y + dir_move[actions[i]][1]] == 0:
                                action = actions[i]

                    rotation = dir_rotation[self.heading][action]
                    self.heading = action
                    movement = 1

                    self.location[0] += dir_move[self.heading][0]
                    self.location[1] += dir_move[self.heading][1]

        # Check if the goal has been found and if enough maze has been explored
        # 'Reset' if both are true
        visited = 0
        for x in range(self.maze_dim):
            for y in range(self.maze_dim):
                if self.closed[x][y] > 0:
                    visited += 1
        area = (visited/(self.maze_dim*self.maze_dim))*100 # Calculate explored area
        if area >= 70 and self.goal_found == True:
            print("Goal found; Maze area mapped : {}".format(area))
            print("Steps taken in the exploratory run : {}".format(self.steps))
            
            rotation = 'Reset'
            movement = 'Reset'
            self.location = [0, 0]
            self.heading = 'up'
            self.run = 1
            self.goal_found = False
            self.steps = 0

        return rotation, movement

    def optimized_robot(self):
        '''
        Generates a policy and applies one of the available algorithms to reach the goal in shortest time steps on the mapped maze
        '''

        # generate a policy by applying an algorithm
        if self.search == False:
            self.policy = self.flood_fill() # algorithm function called to return the policy
            self.search = True
            print("Policy Obtained! Starting the optimized run...")

        x, y = self.location

        # perform rotation and movement
        # returns movement = 3 for three consecutive actions, 2 for 2 consecutive actions and one otherwise
        rotation = dir_rotation[self.heading][self.policy[x][y]] # perform rotation based on current heading and intended action
        self.heading = self.policy[x][y]
        movement = 1

        if self.policy[x + dir_move[self.heading][0]][y + dir_move[self.heading][1]] == self.heading:
            movement = 2

            if self.policy[x + dir_move[self.heading][0]*2][y + dir_move[self.heading][1]*2] == self.heading:
                movement = 3

        self.location[0] += dir_move[self.heading][0]*movement
        self.location[1] += dir_move[self.heading][1]*movement

        # check if the goal has been reached and print no of steps taken
        if self.location[0] in self.goal_bounds and self.location[1] in self.goal_bounds:
            print("Steps taken in the optimized run : {}".format(self.steps))

        return rotation, movement


    def breadth_first_search(self):
        '''
        Breadth First Search Algorithm
        '''
        queue = [(self.heading, self.location)] # queue for breadth first search, maintaining each node and its heading
        parent = {} # dictionary to maintain the parent of each cell
        x, y = self.location
        self.path_grid[x][y] = self.heading
        self.explored[x][y] = 1

        while self.goal_found == False:

            node = queue.pop(0) # pop the first element from the queue for breadth first search
            heading = node[0] # heading for the current location
            vertex = (node[1][0], node[1][1]) # current location to extend

            actions = self.actions(vertex) # check valid actions

            # extend the current location
            for a in range(len(actions)):
                x2 = vertex[0] + dir_move[actions[a]][0]
                y2 = vertex[1] + dir_move[actions[a]][1]

                # check if the extended location is not blocked & is unexplored
                if self.maze_grid[x2][y2] != -1 and self.explored[x2][y2] == 0:
                    self.explored[x2][y2] = 1

                    # check if goal has been found
                    if x2 in self.goal_bounds and y2 in self.goal_bounds:
                        self.goal_found = True
                        parent[(x2, y2)] = vertex
                        self.path_grid[x2][y2] = actions[a]
                        source_reached = False
                        n = (x2, y2) # goal as current cell for beginning to update the policy

                        # creating a policy to reach the goal using parent dictionary and path grid
                        while source_reached == False:
                            p = parent[n] # parent of the current cell
                            self.policy[p[0]][p[1]] = self.path_grid[n[0]][n[1]] # put parent of current node in the policy grid
                            # check if starting position is reached
                            if p[0] == 0 and p[1] == 0:
                                source_reached = True
                            else:
                                n = p # change parent into next cell for updating the policy
                    else:
                        # update queue, path grid and parent dictionary
                        queue.append((actions[a], [x2, y2]))
                        parent[(x2, y2)] = vertex
                        self.path_grid[x2][y2] = actions[a]

        return self.policy

    def depth_first_search(self):
        '''
        Depth First Search Algorithm
        '''
        stack = [(self.heading, self.location)] # stack for depth first search, maintaining each node and its heading
        parent = {} # parent dictionary for each cell
        x, y = self.location
        self.path_grid[x][y] = self.heading
        self.explored[x][y] = 1

        while self.goal_found == False:

            node = stack.pop() # pop from the stack for depth first search
            heading = node[0] # heading for current location
            vertex = (node[1][0], node[1][1]) # current location to extend

            actions = self.actions(vertex) # check valid actions

            # extend the current location
            for a in range(len(actions)):
                x2 = vertex[0] + dir_move[actions[a]][0]
                y2 = vertex[1] + dir_move[actions[a]][1]

                # check if the extended location is not blocked & is unexplored
                if self.maze_grid[x2][y2] != -1 and self.explored[x2][y2] == 0 :
                    self.explored[x2][y2] = 1

                    # check if goal has been found
                    if x2 in self.goal_bounds and y2 in self.goal_bounds:
                        print("Goal Found ")
                        self.goal_found = True
                        parent[(x2, y2)] = vertex
                        self.path_grid[x2][y2] = actions[a]
                        source_reached = False
                        n = (x2, y2) # goal as current cell for beginning to update the policy

                        # creating a policy to reach the goal using parent dictionary and path grid
                        while source_reached == False:
                            p = parent[n] # parent of current cell
                            self.policy[p[0]][p[1]] = self.path_grid[n[0]][n[1]]# put parent of current node in the policy grid

                            # check if starting position is reached
                            if p[0] == 0 and p[1] == 0:
                                source_reached = True
                            else:
                                n = p # change parent into next cell for updating the policy

                    else:
                        # update stack, path grid and parent dictionary
                        stack.append((actions[a], [x2, y2]))
                        parent[(x2, y2)] = vertex
                        self.path_grid[x2][y2] = actions[a]

        return self.policy

    def dijkstra_algorithm(self):
        '''
        Dijkstra's Algorithm
        '''
        x, y = self.location
        self.dist[self.location[0]][self.location[1]] = 0 # distance grid, put 0 at the starting position
        self.path_grid[x][y] = self.heading
        self.explored[x][y] = 1

        g = 0 # cost value for the current node
        cost = 1 # cost for one step

        queue = [(g, self.heading, self.location)] # queue maintaining positions, heading and total cost

        while self.goal_found == False:

            queue.sort() # sort the queue
            node = queue.pop(0) # pop the cell with lowest cost to extend first
            g = node[0] # cost of current position
            vertex = node[2] # current location to extend

            actions = self.actions(vertex) # check for valid actions

            # extend the current location
            for a in range(len(actions)):
                x2 = vertex[0] + dir_move[actions[a]][0]
                y2 = vertex[1] + dir_move[actions[a]][1]

                # check if extended location is not blocked & is unexplored
                if self.maze_grid[x2][y2] != -1 and self.dist[x2][y2] == float('inf'):

                    # check if goal has been reached
                    if x2 in self.goal_bounds and y2 in self.goal_bounds:
                        print("Goal Found ")
                        self.goal_found = True
                        self.path_grid[x2][y2] = actions[a]
                        source_reached = False

                        # creating the policy
                        while source_reached == False:
                            x = x2 - dir_move[self.path_grid[x2][y2]][0]
                            y = y2 - dir_move[self.path_grid[x2][y2]][1]
                            self.policy[x][y] = self.path_grid[x2][y2]
                            x2 = x
                            y2 = y
                            if x == 0 and y == 0:
                                source_reached = True

                    else:
                        # update queue and path grid
                        g2 = g + cost
                        self.dist[x2][y2] = g2
                        queue.append((g2, actions[a], [x2, y2]))
                        self.path_grid[x2][y2] = actions[a]

        return self.policy

    def a_star_search(self):
        '''
        A Star Search Algorithm
        '''
        x, y = self.location
        self.path_grid[x][y] = self.heading
        self.explored[x][y] = 1

        for x in range(self.maze_dim):
            for y in range(self.maze_dim):
                # positive minimum distance from a goal cell to any other cell
                dx = min(abs(x - int(self.maze_dim/2 - 1)), abs(x - int(self.maze_dim/2)))
                dy = min(abs(y - int(self.maze_dim/2 - 1)), abs(y - int(self.maze_dim/2)))
                self.heuristics[x][y] = dx + dy # heuristic value for the cell

        cost = 1 # cost for each step

        g = 0 # cost of a cell calculated from the starting location; 0 for starting location
        h = self.heuristics[x][y] # heuristic value for starting location
        f = g + h # total cost for a cell

        open = [(f, g, h, self.location)] # maintaining a list of tuples containing f, g and h values for each cell expanded

        # A_star search loop
        while self.goal_found == False:
            open.sort() # sort the open list
            next = open.pop(0) # pop the tuple with minimum total cost

            g = next[1] # g value of the current location
            vertex = next[3] # setting current location as vertex
            actions = self.actions(vertex) # check for valid actions

            # extend the current location
            for a in range(len(actions)):
                x2 = vertex[0] + dir_move[actions[a]][0]
                y2 = vertex[1] + dir_move[actions[a]][1]

                # check if extended location is not blocked & is unexplored
                if self.maze_grid[x2][y2] != -1 and self.explored[x2][y2] == 0:
                    self.explored[x2][y2] = 1

                    # check if goal has been reached
                    if x2 in self.goal_bounds and y2 in self.goal_bounds:
                        print("Goal Found ")
                        self.goal_found = True
                        self.path_grid[x2][y2] = actions[a]
                        source_reached = False

                        # creating policy
                        while source_reached == False:
                            x = x2 - dir_move[self.path_grid[x2][y2]][0]
                            y = y2 - dir_move[self.path_grid[x2][y2]][1]
                            self.policy[x][y] = self.path_grid[x2][y2]
                            x2 = x
                            y2 = y
                            if x == 0 and y == 0:
                                source_reached = True

                    else:
                        g2 = g + cost # update g value for extended location
                        h2 = self.heuristics[x2][y2] # heuristic value for the extended location
                        f2 = g2 + h2 # total cost for the extended location
                        open.append((f2, g2, h2, [x2, y2])) # update the list open for the extended location
                        self.path_grid[x2][y2] = actions[a] # update path grid

        return self.policy

    def flood_fill(self):
        '''
        Flood fill / Dynamic programming for calculating no of steps to the goal from each cell
        '''
        change = True # change set to true whenever a value is updated in the value grid
        cost = 1

        while change:
            change = False # set change to false for each iteration
            for x in range(self.maze_dim):
                for y in range(self.maze_dim):
                    # check if goal has been reached
                    if x in self.goal_bounds and y in self.goal_bounds:
                        # if value for goal is greater than 0, set it to 0
                        if self.value[x][y] > 0:
                            self.value[x][y] = 0
                            change = True
                    else:
                        actions = self.actions([x,y]) # check for valid actions
                        # change the value for the neighboring cells
                        for i in range(len(actions)):
                            x2 = x + dir_move[actions[i]][0]
                            y2 = y + dir_move[actions[i]][1]
                            #if x2 >= 0 and x2 < self.maze_dim and y2 >=0 and y2 < self.maze_dim:
                            v2 = self.value[x2][y2] + cost # new value
                            # change, if new value is less than current value
                            if v2 < self.value[x][y]:
                                change = True
                                self.value[x][y] = v2 # update value grid
                                self.policy[x][y] = actions[i] # update policy

        return self.policy

    def wall_finder(self, sensors):
        '''
        Calculates the walls in a cell using sensor inputs
        '''
        headings_1 = ['up', 'right', 'down', 'left']
        headings_2 = ['u', 'r', 'd', 'l']
        front = [1, 2, 4, 8] # bitwise representation of the headings
        x, y = self.location

        # if no walls has been added to the location
        if self.maze_grid[x][y] == 0:
            # if current location is starting position, set it to 1
            if x == 0 and y == 0:
                wall = 1
                self.maze_grid[x][y] = wall
            else:
                # calculate walls in a cell and update maze map
                for i in range(len(headings_1)):
                    if self.heading == headings_1[i] or self.heading == headings_2[i]:
                        back = front[(i+2)%4]
                        wall = sensors[0]*front[i-1] + sensors[1]*front[i] + sensors[2]*front[(i+1)%4] + back

                        self.maze_grid[x][y] = wall # update maze map
                        self.closed[x][y] = 1 # set cell as explored

    def actions(self, vertex):
        '''
        Returns a list of valid actions based on walls in a cell
        '''
        actions = []

        walls = self.maze_grid[vertex[0]][vertex[1]]

        # select valid actions based on wall value
        if walls in [1, 3, 5, 7, 9, 11, 13, 15] :
            actions.append('u')

        if walls in [2, 3, 6, 7, 10, 11, 14, 15] :
            actions.append('r')

        if walls in [8, 9, 10, 11, 12, 13, 14, 15] :
            actions.append('l')

        if walls in [4, 5, 6, 7, 12, 13, 14, 15]:
            actions.append('d')

        return actions