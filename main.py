import game.wrapped_flappy_bird as game
import numpy as np

ACTIONS = 2
game_state = game.GameState()

nothing = np.array([1.,0.])
jump = np.array([0.,1.])

## 01 - jump
## 10 - nothing

while True:
    if np.random.randint(0,100)%3==0:
        x_t, r_0, terminal = game_state.frame_step(jump)
    else:
        x_t, r_0, terminal = game_state.frame_step(nothing)