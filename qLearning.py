import numpy as np
import random

#SAMPLE Q-LEARNING CODE

# Environment settings
gamma = 0.9  # Discount factor
alpha = 0.1  # Learning rate
epsilon = 0.1  # Exploration rate

# Define the environment: a 4x4 grid world
grid_size = 4
goal_state = (3, 3)

# Initialize the Q-table
q_table = np.zeros((grid_size, grid_size, 4))  # 4 actions: up, down, left, right

# Actions
actions = ['up', 'down', 'left', 'right']
action_dict = {
    'up': (-1, 0),
    'down': (1, 0),
    'left': (0, -1),
    'right': (0, 1)
}

# Function to choose an action using the epsilon-greedy strategy
def choose_action(state):
    if random.uniform(0, 1) < epsilon:
        return random.choice(actions)  # Explore
    else:
        return actions[np.argmax(q_table[state])]  # Exploit

# Function to get the next state given the current state and action
def next_state(state, action):
    row, col = state
    row_change, col_change = action_dict[action]
    next_row = min(max(row + row_change, 0), grid_size - 1)
    next_col = min(max(col + col_change, 0), grid_size - 1)
    return next_row, next_col

# Function to update the Q-value
def update_q_value(state, action, reward, next_state):
    action_idx = actions.index(action)
    best_next_action = np.max(q_table[next_state])
    q_table[state][action_idx] = (1 - alpha) * q_table[state][action_idx] + alpha * (reward + gamma * best_next_action)

# Function to run the Q-learning algorithm
def q_learning(episodes):
    for episode in range(episodes):
        state = (0, 0)  # Start state
        while state != goal_state:
            action = choose_action(state)
            next_s = next_state(state, action)
            reward = 1 if next_s == goal_state else -0.1  # Reward for reaching the goal or penalty otherwise
            update_q_value(state, action, reward, next_s)
            state = next_s

# Run the Q-learning algorithm
q_learning(1000)

# Print the learned Q-table
print("Learned Q-Table:")
for row in range(grid_size):
    for col in range(grid_size):
        print(f"State ({row}, {col}): {q_table[row, col]}")