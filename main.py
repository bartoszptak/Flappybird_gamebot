import numpy as np
import argparse

import game.wrapped_flappy_bird as game
from model import get_model
from gamebot import Flappybird as gamebot

def main(train=False):
    game_state = game.GameState()
    
    bot = gamebot(get_model(), train)


    next_state, reward, terminal = game_state.frame_step(bot.NOTHING)
    next_state = bot.image_preprocessing(next_state)

    state = np.stack((next_state, next_state, next_state, next_state), axis=2)
    state = np.reshape(state, (1,*state.shape))

    while True:
        action, action_index = bot.make_action(state)

        next_state, reward, terminal = game_state.frame_step(action)
        next_state = bot.image_preprocessing(next_state)

        next_state = next_state.reshape(1, *next_state.shape, 1)
        next_state = np.append(next_state, state[:, :, :, :3], axis=3)

        if train:
                bot.make_buffer(state, action_index, reward, next_state, terminal)

                print('Epoch: {} - loss: {}'.format(*bot.make_train()))

        state = next_state



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-t','--train', help='Make training', required=False, action='store_true')
    args = vars(parser.parse_args())
    main(args['train'])