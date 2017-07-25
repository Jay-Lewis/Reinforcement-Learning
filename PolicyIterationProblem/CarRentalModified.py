__author__ = 'Justin'

import numpy as np
import copy
from matplotlib import cm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Car Rental Problem

# ---Problem Description-----------------

# Environment:
#   -two car rental lots (poisson rates for rental and return)
#   -a maximum of twenty cars are held at each location

# State Space:
#   -number of cars at each lot

# Action Space:
#   -number of cars sent between each lot

# Reward:
#   -amount of money made renting cars ($10 per car rented)

# -----------------------------------------


# Initialize Value Function Table
lotsize = 20
maxval = 5
values = [[0 for x in range(0,lotsize+1)]for y in range(0,lotsize+1)]

# Initialize Policy Table

#note may have reversed x and y
actions = [[[1 if z == 0 else 0 for z in range(-1*min(maxval,x),min(maxval,y)+1)] for x in range(0,lotsize+1)]for y in range(0,lotsize+1)]


# State Transition Function

def next_state(previous,delta):
    next = np.add(np.asarray(previous),np.asarray([-delta,delta]))

    for index in range(0,len(next)):
        if next[index] < 0:
            next[index] = 0
            print('ERROR')
        elif next[index] > lotsize:
            next[index] = lotsize

    return list(next)

# Policy Iteration Loop
loopiter = 5

for _ in range(0,loopiter):

    # Policy Evaluation
    threshold = 0.1
    max_error = copy.deepcopy(threshold)
    step = 0
    discount = 0.9

    while(max_error >= threshold):
        step += 1
        temp_values = copy.deepcopy(values)
        for cars_a in range(0,lotsize+1):
            for cars_b in range(0,lotsize+1):
                action = [z for z in range(-1*min(maxval,cars_b),min(maxval,cars_a)+1) if actions[cars_a][cars_b][z+min(maxval,cars_b)] == 1][0]

                state_next = next_state([cars_a,cars_b],action)

                # Calculate E[immediate return]
                lambda_a = 4; lambda_b = 4
                lambda_a_return = 3; lambda_b_return = 2
                imm_cost = -2*abs(action)
                imm_benefit_a = 10*lambda_a if state_next[0] >= lambda_a else 10*state_next[0]
                imm_benefit_b = 10*lambda_b if state_next[1] >= lambda_b else 10*state_next[1]
                imm_return = imm_cost+imm_benefit_a+imm_benefit_b

                values[cars_a][cars_b] = imm_return+discount*temp_values[cars_a][cars_b]
        error = np.absolute(np.subtract(np.matrix(values),np.matrix(temp_values)))
        max_error = np.matrix.max(error)

    # Policy Iteration
    for cars_a in range(0, lotsize + 1):
        for cars_b in range(0, lotsize + 1):
            possible_actions = np.arange(-1*min(maxval,cars_b),min(maxval,cars_a)+1)
            possible_next_states = [next_state([cars_a,cars_b],action) for action in possible_actions]

            best_state = [];    best_value = 0;     best_action = 0;

            for new_state,new_action in zip(possible_next_states,possible_actions):
                new_value = values[new_state[0]][new_state[1]]
                if(new_value > best_value):
                    best_value = new_value
                    best_state = new_state
                    best_action = new_action

            # update best action
            best_index = list(possible_actions).index(best_action)
            new_action_vector = [0 if index != best_index else 1 for index in range(0,len(possible_actions))]
            actions[cars_a][cars_b] = new_action_vector

    print('Number of Evaluation Iterations:',step)
    print(np.max(values))



# Plot Actions

fig = plt.figure()
ax = fig.gca(projection='3d')
ax1 = fig.add_subplot(111, projection='3d')

xpos = np.array([]); ypos = np.array([])
for i in range(0,lotsize+1):
    xpos = np.append(xpos,np.linspace(0,lotsize,lotsize+1))
for i in range(0,lotsize+1):
    ypos = np.append(ypos,np.linspace(i,i,lotsize+1))
zpos = np.zeros([1,pow(lotsize+1,2)])[0]
dx = np.ones([1,pow(lotsize+1,2)])[0]
dy = np.ones([1,pow(lotsize+1,2)])[0]
dz = []
for cars_a in range(0, lotsize + 1):
    for cars_b in range(0, lotsize + 1):
        action = [z for z in range(-1*min(maxval,cars_b),min(maxval,cars_a)+1) if actions[cars_a][cars_b][z+min(maxval,cars_b)] == 1][0]
        dz.append(abs(action))

ax1.bar3d(ypos, xpos, zpos, dy, dx, dz, color='#00ceaa')
ax1.set_xlabel('Cars in Lot B')
ax1.set_ylabel('Cars in Lot A')



# Plot Value Function

fig = plt.figure()
ax = fig.gca(projection='3d')
ax1 = fig.add_subplot(111, projection='3d')

xpos = np.array([]); ypos = np.array([])
for i in range(0,lotsize+1):
    xpos = np.append(xpos,np.linspace(0,lotsize,lotsize+1))
for i in range(0,lotsize+1):
    ypos = np.append(ypos,np.linspace(i,i,lotsize+1))
zpos = np.zeros([1,pow(lotsize+1,2)])[0]
dx = np.ones([1,pow(lotsize+1,2)])[0]
dy = np.ones([1,pow(lotsize+1,2)])[0]
dz = np.zeros([1,0])
for j in range(0,len(values)):
    dz = np.append(dz,values[j])
X, Y = np.meshgrid(np.linspace(0,lotsize,lotsize+1),np.linspace(0,lotsize,lotsize+1))
values = np.asarray(values)
# surf = ax.plot_surface(X,Y,np.asarray(values),cmap=cm.coolwarm,
#                        linewidth=0, antialiased=False)
ax1.bar3d(ypos, xpos, zpos, dy, dx, dz, color='#00ceaa')
ax1.set_xlabel('Cars in Lot B')
ax1.set_ylabel('Cars in Lot A')
plt.show()

print('-------------Training Complete------------')



