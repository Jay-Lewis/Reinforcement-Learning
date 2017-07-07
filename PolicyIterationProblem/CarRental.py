__author__ = 'Justin'

import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

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
action_range = [-5,5]
values = [[0 for x in range(0,lotsize+1)]for y in range(0,lotsize+1)]

# Initialize Policy Table
actions = [[[0 for z in range(action_range[0],action_range[1]+1)] for x in range(0,lotsize+1)]for y in range(0,lotsize+1)]

for x in range(0,lotsize+1):
    for y in range(0,lotsize+1):
        randnum = round(np.random.uniform(action_range[0],action_range[1]))
        actions[x][y][randnum] = 1


# State Transition Function

def next_state(previous,delta):
    next = np.add(np.asarray(previous),np.asarray([-delta,delta]))

    for index in range(0,len(next)):
        if next[index] <0:
            next[index] = 0
        elif next[index] > lotsize:
            next[index] = lotsize

    return list(next)

# Policy Iteration Loop

# Policy Evaluation
threshold = 0.1
max_error = threshold
# print(np.matrix(values))

while(max_error >= threshold):

    temp_values = values
    discount = 0.9

    for cars_a in range(0,lotsize+1):
        for cars_b in range(0,lotsize+1):

            action = [z for z in range(action_range[0],action_range[1]+1) if actions[cars_a][cars_b][z-action_range[0]] == 1][0]
            state_next = next_state([cars_a,cars_b],action)

            # Calculate E[immediate return]
            lambda_a = 3; lambda_b = 4
            imm_cost = -2*abs(action)
            imm_benefit_a = 10*lambda_a if state_next[0] >=lambda_a else 10*state_next[0]
            imm_benefit_b = 10*lambda_b if state_next[1] >=lambda_b else 10*state_next[1]
            imm_return = imm_cost+imm_benefit_a+imm_benefit_b

            values[cars_a][cars_b] = imm_return+discount*temp_values[cars_a][cars_b]

    error = np.absolute(np.subtract(np.matrix(values),np.matrix(temp_values)))
    max_error = np.matrix.max(error)

print(np.matrix(values))


# Plot Value Function

fig = plt.figure()
ax1 = fig.add_subplot(111, projection='3d')
xpos = np.array([]); ypos = xpos
for i in range(0,lotsize+1):
    xpos = np.append(xpos,np.linspace(0,lotsize,lotsize+1))
for i in range(0,lotsize+1):
    ypos = np.append(ypos,np.linspace(i,i,lotsize+1))
zpos = np.zeros([1,pow(lotsize+1,2)])[0]
dx = np.ones([1,pow(lotsize+1,2)])[0]
dy = np.ones([1,pow(lotsize+1,2)])[0]
dz = []
for j in range(0,len(values)):
    dz = dz + values[j]

ax1.bar3d(xpos, ypos, zpos, dx, dy, dz, color='#00ceaa')
plt.show()

# Policy Iteration
