import numpy as np
import cv2
from collections import deque
import random

class Flappybird:

    def __init__(self, model, train):
        self.NOTHING = np.array([1., 0.])
        self.JUMP = np.array([0., 1.])
        self.action_size = 2

        self.model = model
        self.deque = deque(maxlen=20000)
        self.INITIAL_EPSILON = 0.1
        self.FINAL_EPSILON = 0.0001
        self.EXPLORE = 3000000.
        self.GAMMA = 0.99
        
        self.batch_size = 32
        self.train_index = 0

        self.train = train

        if train:
            self.epsilon = self.INITIAL_EPSILON
        else:
            self.epsilon = self.FINAL_EPSILON
            self.model.load_weights('model/weights.h5')

    @staticmethod
    def image_preprocessing(im):
        im = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
        im = np.transpose(im, (1,0))
        im = cv2.resize(im, (64,64))
        im = im * 1./255.

        return im

    def make_action(self, state):
        if random.random() < self.epsilon:
            index = random.randrange(self.action_size)
        else:
            index = np.argmax(self.model.predict(state)[0])

        action = np.zeros(self.action_size)
        action[index] = 1.

        if self.epsilon > self.FINAL_EPSILON:
            self.epsilon -= (self.INITIAL_EPSILON - self.FINAL_EPSILON) / self.EXPLORE

        return action, index

    def make_buffer(self, state, action_index, reward, next_state, terminal):
        self.deque.append((state, action_index, reward, next_state, terminal))

    def make_train(self):
        if len(self.deque) < self.batch_size:
            return 0, 'WAIT'

        minibatch = random.sample(self.deque, self.batch_size)

        state, action_index, reward, next_state, terminal = zip(*minibatch)

        state, next_state = np.concatenate(state), np.concatenate(next_state)

        targets, Q = self.model.predict(state), self.model.predict(next_state)

        targets[range(self.batch_size), action_index] = reward + self.GAMMA*np.max(Q, axis=1)*np.invert(terminal)

        if self.train_index%1000==0:
            self.model.save_weights('model/weights.h5')

        self.train_index+=1

        return self.train_index, self.model.train_on_batch(state, targets)