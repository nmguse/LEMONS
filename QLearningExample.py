# Cartpole DQN

import os
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"] = ""
from keras.models import Sequential
#from tensorflow.keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import random
import gym
import numpy as np
from collections import deque # just like a list, but you can add things from the top or bottom

# setting parameters --
env = gym.make('CartPole-v0')
state_size = env.observation_space.shape[0]
action_size = env.action_space.n
batch_size = 32
n_episodes = 1001
output_dir = "model_output/cartpole"

if not os.path.exists(output_dir):
    print('no output directory exists currently; making one in current working directory.')
    os.makedirs(output_dir)

# formal definition of agent
# Events from previous trial don't provide very  much information if they are adjacent to each other, so the total amount of trials ran increases by randomizing the iterations which are actually interpreted
class SwitchyBoi:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size

# using a deque allows to keep adding new elements to the list, but after length surpasses 2000, oldest elements are removed
        self.memory = deque(maxlen=2000)

# gamma parameter weights the 'trustability' of upcoming rewards expected (a function of # of steps it takes to get the reward)
        self.gamma = 0.95

# epsilon parameter is the exploration rate (the number of actions from 0-1 that should experiment with 'new' inputs rather than default to the safest option based on old inputs) this will initially be 0 until the network can be trusted with its' knowledge of what to do in the environmnet.
        self.epsilon = 1.0
        self.epsilon_decay = 0.998
        self.epsilon_min = 0.03

# "" stochastic gradient descent learning rage, step size for stochastic gradient descent optomizer ""
        self.learning_rate = 0.001

# private method - can only be used by a particular instance of DNQAgent class within itself.
        self.model = self._build_model()

    def _build_model(self):

#calls on the Sequential keras model, using stochastic gradient ascent template.
        model = Sequential()

# all hidden layers will be dense with 24 neurons, one dimension for each state type (angular vel, lin vel, angle, pos), and "" activation "" is "" rectified linear unit ""
# decided on this number of layers // neurons (first argument) pretty arbitrarily. Try other numbers if you feel like it, but #neurons should always be an integer coefficient of a power of 2
        model.add(Dense(24,input_dim = self.state_size, activation='relu'))
        model.add(Dense(24, activation = 'relu'))
# output layer has the same number of neurons as possible actions (2: right and left), and have linear activation because actions aren't "" probabilistic "" and are "" direct "" out of this neural network.
# probably, if the activation was nonlinear, there wouldn't be a binary output (right or left), and instead there would be a ratio output (like 0.72 in the direction of right if left = 0 and right = 1).
        model.add(Dense(self.action_size, activation = 'linear'))

# mse is "mean squared error"
        model.compile(loss = "mse", optimizer = Adam(lr=self.learning_rate))

        return model

# Takes in state at the current timestep, action at the current timestep, reward at the current timestep, and the next state, and enables us to model the circumstances and rewards expected in the next state. Done notifies when the episode has ended.
    def remember(self,state,action,reward,next_state,done):
        self.memory.append((state,action,reward,next_state,done))

#figures out what action to take based on states
    def act(self,state):
# this if statement randomly selects a value between 0 and 1 anc compares it to the exlore/exploit ratio parameter. Because this is initially 1, it will obviously always execute the instructions to explore...
        if np.random.rand() <= self.epsilon:
# returns a random number between 0 and the number of possible actions (2 in this case)
            return random.randrange(self.action_size)
# following is the exploitation option
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])
    def replay(self,batch_size):
#takes a random sample from the table made by the memory method
        minibatch = random.sample(self.memory, batch_size)

        for state, action, reward, next_state, done in minibatch:
# if max timesteps has been reached, or if game ends because cartpole angle exceeds maximum or cart moves off screen, done == true and target is equal to reward
            target = reward
# otherwise, if the game is still in progress, target is the sum of current reward and the predicted future reward scaled down by gamma parameter
            if not done:
                target = (reward + self.gamma * np.amax(self.model.predict(next_state)[0]))
            target_f = self.model.predict(state)
            target_f[0][action] = target

            self.model.fit(state,target_f, epochs = 1, verbose = 0)

# decreases epsilon value for the next cycle as long as it is greater than the minimum epsilon value
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self,name):
        self.model.load_weights(name)

    def save(self,name):
        self.model.save_weights(name)

agent = SwitchyBoi(state_size, action_size)

done = False

# for each episode in the total number of episodes defined (currently 1001)... each episode starts in a 'reset' state (a random configuration of the four possible states)
for e in range(n_episodes):
    state = env.reset()
# transposes from row to column
    state = np.reshape(state, [1,state_size])

# the maximum game time the cartpole can remain 'alive' before game ends is 5000 timesteps.
    for time in range(9000):
# render the environment for every timestep
        env.render()
# set the local action variable to the decision made by the agent (right or left), as a function of the previous state
        action = agent.act(state)
# collect the new state, reward and 'done' boolean from the environment given the action set in the previous line
        next_state, reward, done, _=env.step(action)
# set local reward variable equal to reward returned in this timestep. If agent failed on this timestep, penalize 10 points
        reward = reward if not done else -10
# transpose array of new states (after last timestep) from row to column
        next_state = np.reshape(next_state, [1,state_size])
# record step taken in this timestep, the action done in this timestep, the reward returned in this timestep, and whether or not this timestep caused the game to fail.
        agent.remember(state, action, reward, next_state, done)
# set the new state to the current state for the next timestep
        state = next_state
# if the agent failed on previous timestep, print diagnostic information
        if done:
            print("episode: {}/{}, score: {}, epsilon: {:.2}".format(e,n_episodes,time,agent.epsilon))
            break
# updates theta weights while interacting with the environment
        if len(agent.memory) > batch_size:
            agent.replay(batch_size)

# saves model weights every 50 episodes
        if e % 50 == 0:
            agent.save(output_dir + "Weights_"+'{:04d}'.format(e) + ".hdf5")
