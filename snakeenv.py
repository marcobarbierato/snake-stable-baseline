import numpy as np
import gymnasium as gym
from gymnasium import spaces

class SnakeEnv(gym.Env):
    """
    Custom Environment that follows gym interface.
    """

    # Because of google colab, we cannot implement the GUI ('human' render mode)
    metadata = {"render_modes": ["console"]}

    # Define constants for clearer code
    HEAD = 4
    BODY = 3
    FRUIT = 2
    EMPTY = 1
    WALL = 0

    HEAD = (4-2)/2
    BODY = (3-2)/2
    FRUIT = (2-2)/2
    EMPTY = (1-2)/2
    WALL = (0-2)/2

    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3
    NONE = 4

    def __init__(self, board_size=10, max_steps=1000,render_mode="console"):
        super().__init__()

        self.WIN_REWARD = 1.
        self.FRUIT_REWARD = .5
        self.STEP_REWARD = 0.
        self.ATE_HIMSELF_REWARD = -.2
        self.HIT_WALL_REWARD = -.1

        self.render_mode = render_mode

        # Size of the 1D-grid
        self.board_size = board_size
        # Initialize the agent at the right of the grid
        # self.agent_pos = grid_size - 1

        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions, we have two: left and right
        n_actions = 4
        self.action_space = spaces.Discrete(n_actions)
        # The observation will be the coordinate of the agent
        # this can be described both by Discrete and Box space
        #self.observation_space = spaces.Box(
        #    low=0, high=self.grid_size, shape=(1,), dtype=np.float32
        #)
        #self.observation_space = spaces.MultiDiscrete(
        #    np.ones((self.board_size, self.board_size)) * 4
        #)

        #self.observation_space = spaces.Box(low=0, high=4, shape=(self.board_size, self.board_size), dtype=np.int32)
        self.observation_space = spaces.Box(low=-1, high=1, shape=(self.board_size, self.board_size), dtype=np.float32)
        self.max_steps = max_steps
        self.steps = 0
        


    def reset_board(self):

        board = np.ones((self.board_size, self.board_size)) * self.EMPTY
        board[[0, -1], :] = self.WALL
        board[:, [0, -1]] = self.WALL
        # add head, add fruit
        available = np.argwhere(board == self.EMPTY)
        ind = available[np.random.choice(range(len(available)))]
        board[ind[0], ind[1]] = self.HEAD
        available = np.argwhere(board == self.EMPTY)
        ind = available[np.random.choice(range(len(available)))]
        board[ind[0], ind[1]] = self.FRUIT

        return board.astype('float32')

    def reset(self, seed=None, options=None):
        """
        Important: the observation must be a numpy array
        :return: (np.array)
        """
        super().reset(seed=seed, options=options)
        #self.rng = np.random.default_rng(seed)
        self.board=self.reset_board()
        self.body = []
        return self.board, {}


    def step(self, action):
        self.steps +=1
        #print(self.steps)
        if self.steps >= self.max_steps:
          truncated = True
          self.steps=0
        else:
          truncated = False

        head = np.argwhere(self.board == self.HEAD)[0]
        # action offsets
        dx, dy = 0, 0
        if action == self.UP:
            dx=1
        elif action == self.DOWN:
            dx = -1
        elif action == self.RIGHT:
            dy = 1
        elif action == self.LEFT:
            dy = -1
        new_head = head + np.array([dx, dy])

        if self.board[new_head[0], new_head[1]] == self.WALL: #hits wall
            terminated=False
            #truncated=False

            # return unchanged board with wall reward
            return  (
                self.board,
                self.HIT_WALL_REWARD,
                terminated,
                truncated,
                {}
            )

        #check this
        fruit = np.argwhere(self.board == self.FRUIT)[0]
        fruit_eaten = np.all(new_head == fruit, axis=-1)

        ate_self = False
        if self.board[new_head[0], new_head[1]] == self.BODY: #ate self
            ate_self = True
            index = int( np.argwhere( np.all(np.array(self.body)==new_head, axis=1) )[0][0] )
            del self.body[index:]
            # maybe should be index-1?


        # remove last piece of body (if fruit not eaten), add head
        self.body.insert(0, head)
        self.board[np.where(self.board==self.BODY)] = self.EMPTY
        self.board[head[0], head[1]] = self.EMPTY
        if not fruit_eaten:
            self.body.pop()
        #check these two
        if self.body:
            for i in self.body:
                self.board[i[0], i[1]] = self.BODY

        # is the body updated?

        #set new head
        self.board[new_head[0], new_head[1]] = self.HEAD

        if ate_self:
            terminated=False
            #truncated=False
            return  (
                    self.board,
                    self.ATE_HIMSELF_REWARD,
                    terminated,
                    truncated,
                    {}
                )

        if fruit_eaten:
            available=np.argwhere(self.board==self.EMPTY)

            if len(available) ==0: # win game
                self.board = self.reset_board()
                self.body = []

                terminated=False
                #truncated=False

                # return unchanged board with will reward
                return  (
                    self.board,
                    self.WIN_REWARD,
                    terminated,
                    truncated,
                    {}
                )
            #add fruit
            new_fruit = available[np.random.choice(range(len(available)))]
            self.board[new_fruit[0], new_fruit[1]] = self.FRUIT

            terminated=False
            #truncated=False

            return  (
                    self.board,
                    self.FRUIT_REWARD,
                    terminated,
                    truncated,
                    {}
                )


        else:
            terminated=False
            #truncated=False
            return  (
                    self.board,
                    self.STEP_REWARD,
                    terminated,
                    truncated,
                    {}
                )

    def render(self):
        pass

    def close(self):
        pass


class SnakeEnvPartial(SnakeEnv):
    def __init__(self, board_size=10, max_steps=1000, mask_size=2, render_mode="console"):
        self.mask_size = mask_size
        super().__init__(board_size=board_size, max_steps=max_steps, render_mode=render_mode)
        
        self.observation_space = spaces.Box(low=-1, high=1, shape=(2*self.mask_size+1, 2*self.mask_size+1), dtype=np.float32)

        self.max_steps = max_steps
        self.steps = 0


    def observation_from_board(self):
        bs=np.pad(self.board, self.mask_size, constant_values=self.WALL)
        h= np.argwhere(bs==self.HEAD)[0]
        
        observed =  bs[(h[0]-self.mask_size):(h[0]+self.mask_size+1), (h[1]-self.mask_size):(h[1]+self.mask_size+1)]

        return observed
    def reset(self, seed=None, options=None):
        """
        Important: the observation must be a numpy array
        :return: (np.array)
        """
        super().reset(seed=seed, options=options)

        self.board=self.reset_board()
        self.body = []

        
        return self.observation_from_board(), {}

    def step(self, action):
        self.steps +=1
        #print(self.steps)
        if self.steps >= self.max_steps:
          truncated = True
          self.steps=0
        else:
          truncated = False

        head = np.argwhere(self.board == self.HEAD)[0]
        # action offsets
        dx, dy = 0, 0
        if action == self.UP:
            dx=1
        elif action == self.DOWN:
            dx = -1
        elif action == self.RIGHT:
            dy = 1
        elif action == self.LEFT:
            dy = -1
        new_head = head + np.array([dx, dy])

        if self.board[new_head[0], new_head[1]] == self.WALL: #hits wall
            terminated=False
            #truncated=False

            # return unchanged board with wall reward
            return  (
                self.observation_from_board(),
                self.HIT_WALL_REWARD,
                terminated,
                truncated,
                {}
            )

        #check this
        fruit = np.argwhere(self.board == self.FRUIT)[0]
        fruit_eaten = np.all(new_head == fruit, axis=-1)

        ate_self = False
        if self.board[new_head[0], new_head[1]] == self.BODY: #ate self
            ate_self = True
            index = int( np.argwhere( np.all(np.array(self.body)==new_head, axis=1) )[0][0] )
            del self.body[index:]
            # maybe should be index-1?


        # remove last piece of body (if fruit not eaten), add head
        self.body.insert(0, head)
        self.board[np.where(self.board==self.BODY)] = self.EMPTY
        self.board[head[0], head[1]] = self.EMPTY
        if not fruit_eaten:
            self.body.pop()
        #check these two
        if self.body:
            for i in self.body:
                self.board[i[0], i[1]] = self.BODY

        # is the body updated?

        #set new head
        self.board[new_head[0], new_head[1]] = self.HEAD

        if ate_self:
            terminated=False
            #truncated=False
            return  (
                    self.observation_from_board(),
                    self.ATE_HIMSELF_REWARD,
                    terminated,
                    truncated,
                    {}
                )

        if fruit_eaten:
            available=np.argwhere(self.board==self.EMPTY)

            if len(available) == 0: # win game
                self.board = self.reset_board()
                self.body = []

                terminated=False
                #truncated=False

                # return unchanged board with wall reward
                return  (
                    self.observation_from_board(),
                    self.WIN_REWARD,
                    terminated,
                    truncated,
                    {}
                )
            #add fruit
            new_fruit = available[np.random.choice(range(len(available)))]
            self.board[new_fruit[0], new_fruit[1]] = self.FRUIT

            terminated=False
            #truncated=False

            return  (
                    self.observation_from_board(),
                    self.FRUIT_REWARD,
                    terminated,
                    truncated,
                    {}
                )


        else:
            terminated=False
            #truncated=False
            return  (
                    self.observation_from_board(),
                    self.STEP_REWARD,
                    terminated,
                    truncated,
                    {}
                )


        pass

class SnakeEnvBonus(SnakeEnv):
    def __init__(self, board_size=10, max_steps=1000, render_mode="console"):
        super().__init__(board_size, max_steps, render_mode)

        self.WIN_REWARD = 100
        self.FRUIT_REWARD = 1
        self.STEP_REWARD = 0.
        self.ATE_HIMSELF_REWARD = -10
        self.HIT_WALL_REWARD = -10

    def step(self, action):
        self.steps +=1
        #print(self.steps)
        if self.steps >= self.max_steps:
          truncated = True
          self.steps=0
        else:
          truncated = False

        head = np.argwhere(self.board == self.HEAD)[0]
        # action offsets
        dx, dy = 0, 0
        if action == self.UP:
            dx=1
        elif action == self.DOWN:
            dx = -1
        elif action == self.RIGHT:
            dy = 1
        elif action == self.LEFT:
            dy = -1
        new_head = head + np.array([dx, dy])

        if self.board[new_head[0], new_head[1]] == self.WALL: #hits wall
            terminated=True
            #truncated=False

            # return unchanged board with wall reward
            return  (
                self.board,
                self.HIT_WALL_REWARD,
                terminated,
                truncated,
                {}
            )

        #check this
        fruit = np.argwhere(self.board == self.FRUIT)[0]
        fruit_eaten = np.all(new_head == fruit, axis=-1)

        ate_self = False
        if self.board[new_head[0], new_head[1]] == self.BODY: #ate self
            ate_self = True
            index = int( np.argwhere( np.all(np.array(self.body)==new_head, axis=1) )[0][0] )
            del self.body[index:]
            # maybe should be index-1?


        # remove last piece of body (if fruit not eaten), add head
        self.body.insert(0, head)
        self.board[np.where(self.board==self.BODY)] = self.EMPTY
        self.board[head[0], head[1]] = self.EMPTY
        if not fruit_eaten:
            self.body.pop()
        #check these two
        if self.body:
            for i in self.body:
                self.board[i[0], i[1]] = self.BODY

        # is the body updated?

        #set new head
        self.board[new_head[0], new_head[1]] = self.HEAD

        if ate_self:
            terminated=True
            #truncated=False
            return  (
                    self.board,
                    self.ATE_HIMSELF_REWARD,
                    terminated,
                    truncated,
                    {}
                )

        if fruit_eaten:
            available=np.argwhere(self.board==self.EMPTY)

            if len(available) ==0: # win game
                self.board = self.reset_board()
                self.body = []

                terminated=True
                #truncated=False

                # return unchanged board with will reward
                return  (
                    self.board,
                    self.WIN_REWARD,
                    terminated,
                    truncated,
                    {}
                )
            #add fruit
            new_fruit = available[np.random.choice(range(len(available)))]
            self.board[new_fruit[0], new_fruit[1]] = self.FRUIT

            terminated=False
            #truncated=False

            return  (
                    self.board,
                    self.FRUIT_REWARD,
                    terminated,
                    truncated,
                    {}
                )


        else:
            terminated=False
            #truncated=False
            return  (
                    self.board,
                    self.STEP_REWARD,
                    terminated,
                    truncated,
                    {}
                )


        pass


class SnakeEnvPartialBonus(SnakeEnvPartial):
    def __init__(self, board_size=10, max_steps=1000, mask_size=2, render_mode="console"):
        super().__init__(board_size, max_steps, mask_size, render_mode)
        self.WIN_REWARD = 100
        self.FRUIT_REWARD = 1
        self.STEP_REWARD = 0.
        self.ATE_HIMSELF_REWARD = -10
        self.HIT_WALL_REWARD = -10

    def step(self, action):
        self.steps +=1
        #print(self.steps)
        if self.steps >= self.max_steps:
            truncated = True
            self.steps=0
        else:
            truncated = False

        head = np.argwhere(self.board == self.HEAD)[0]
        # action offsets
        dx, dy = 0, 0
        if action == self.UP:
            dx=1
        elif action == self.DOWN:
            dx = -1
        elif action == self.RIGHT:
            dy = 1
        elif action == self.LEFT:
            dy = -1
        new_head = head + np.array([dx, dy])

        if self.board[new_head[0], new_head[1]] == self.WALL: #hits wall
            terminated=True
            #truncated=False

            # return unchanged board with wall reward
            return  (
                self.observation_from_board(),
                self.HIT_WALL_REWARD,
                terminated,
                truncated,
                {}
            )

        #check this
        fruit = np.argwhere(self.board == self.FRUIT)[0]
        fruit_eaten = np.all(new_head == fruit, axis=-1)

        ate_self = False
        if self.board[new_head[0], new_head[1]] == self.BODY: #ate self
            ate_self = True
            index = int( np.argwhere( np.all(np.array(self.body)==new_head, axis=1) )[0][0] )
            del self.body[index:]
            # maybe should be index-1?


        # remove last piece of body (if fruit not eaten), add head
        self.body.insert(0, head)
        self.board[np.where(self.board==self.BODY)] = self.EMPTY
        self.board[head[0], head[1]] = self.EMPTY
        if not fruit_eaten:
            self.body.pop()
        #check these two
        if self.body:
            for i in self.body:
                self.board[i[0], i[1]] = self.BODY

        # is the body updated?

        #set new head
        self.board[new_head[0], new_head[1]] = self.HEAD

        if ate_self:
            terminated=True
            #truncated=False
            return  (
                    self.observation_from_board(),
                    self.ATE_HIMSELF_REWARD,
                    terminated,
                    truncated,
                    {}
                )

        if fruit_eaten:
            available=np.argwhere(self.board==self.EMPTY)

            if len(available) == 0: # win game
                self.board = self.reset_board()
                self.body = []

                terminated=True
                #truncated=False

                # return unchanged board with wall reward
                return  (
                    self.observation_from_board(),
                    self.WIN_REWARD,
                    terminated,
                    truncated,
                    {}
                )
            #add fruit
            new_fruit = available[np.random.choice(range(len(available)))]
            self.board[new_fruit[0], new_fruit[1]] = self.FRUIT

            terminated=False
            #truncated=False

            return  (
                    self.observation_from_board(),
                    self.FRUIT_REWARD,
                    terminated,
                    truncated,
                    {}
                )


        else:
            terminated=False
            #truncated=False
            return  (
                    self.observation_from_board(),
                    self.STEP_REWARD,
                    terminated,
                    truncated,
                    {}
                )


