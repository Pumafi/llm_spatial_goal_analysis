import json
import string
import random
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch


def make_simple_prompt(agent, goal):
    return f"""
You are solving a grid navigation task.

The grid is 5x5.
Coordinates are (x, y).
x increases to the RIGHT.
y increases DOWN.
The top-left cell is (0, 0).

Agent position: {agent}
Goal position: {goal}

Reason step by step about the spatial relationship.
Then write:

ACTION:
""".strip()

def make_one_action_prompt(agent, goal):
    return f"""
You are solving a grid navigation task.

The grid is 5x5.
Coordinates are (x, y).
x increases to the RIGHT.
y increases DOWN.
The top-left cell is (0, 0).

Agent position: {agent}
Goal position: {goal}

Reason step by step about the spatial relationship.

Choose optimal move among: [LEFT (-1, 0), RIGHT (+1, 0), UP (0, -1), DOWN (0, +1)]
Then write:

ACTION:
""".strip()

def make_one_cardinal_prompt(agent, goal):
    return f"""
You are solving a grid navigation task.

The grid is 5x5.
Coordinates are (x, y).
x decreases to the WEST
x increases to the EAST.
y decreases to the NORTH.
y increases to the SOUTH.
The top-left cell is (0, 0).

The agent can only move along one coordinate, one cell at a time.

Agent position: {agent}
Goal position: {goal}

Reason step by step about the spatial relationship.
Then write:

ACTION:
""".strip()


def generate_example(agent, goal, height=5, width=5):
    ar, ac = agent
    gr, gc = goal

    cardinal = make_one_cardinal_prompt(agent, goal)
    action = make_one_action_prompt(agent, goal)
    simple = make_simple_prompt(agent, goal)

    agent_grid = [0 for _ in range(width * height)]
    goal_grid  = [0 for _ in range(width * height)]

    agent_idx = ar * width + ac
    goal_idx  = gr * width + gc

    agent_grid[agent_idx] = 1
    goal_grid[goal_idx] = 1

    return {
        "inputs": {
            "cardinal": json.dumps(cardinal, indent=2),
            "action": json.dumps(action, indent=2),
            "simple": simple,
        },
        "output": {
            "agent_grid": agent_grid,
            "goal_grid": goal_grid,
        },
    }

def generate_simple_example(agent, goal, height=5, width=5):
    ar, ac = agent
    gr, gc = goal

    simple = make_simple_prompt(agent, goal)

    agent_grid = [0 for _ in range(width * height)]
    goal_grid  = [0 for _ in range(width * height)]

    agent_idx = ar * width + ac
    goal_idx  = gr * width + gc

    agent_grid[agent_idx] = 1
    goal_grid[goal_idx] = 1

    return {
        "inputs": {
            "simple": json.dumps(simple, indent=2),
        },
        "output": {
            "agent_grid": agent_grid,
            "goal_grid": goal_grid,
        },
    }

def make_random_grid_dataset(nb_samples, height=5, width=5):

    dataset = []
    for _ in range(nb_samples):
        agent = (0, 0) #(random.randrange(height), random.randrange(width))

        while True:
            goal = (random.randrange(height), random.randrange(width))
            if goal != agent:
                break

        example = generate_example(
            height=height,
            width=width,
            agent=agent,
            goal=goal,
        )

        dataset.append(example)

    return dataset


def make_determinist_grid_dataset(nb_samples, agent, height=5, width=5):

    dataset = []
    for k in range(height):
        for h in range(width):
            agent = (k, h)
            for i in range(height):
                for j in range(width):
                    goal = (i, j)
                    if goal != agent:
                        example = generate_simple_example(
                            height=height,
                            width=width,
                            agent=agent,
                            goal=goal,
                        )
                
                        dataset.append(example)
    return dataset

def visualize_binary_grid(goal_grid, agent_grid, h=5, w=5):
    goal_grid_2d = np.array(goal_grid).reshape(h, w)
    agent_grid_2d = np.array(agent_grid).reshape(h, w)
    
    display_grid = np.zeros((h, w))
    for i in range(h):
        for j in range(w):
            if goal_grid_2d[i, j] == 1 and agent_grid_2d[i, j] == 1:
                display_grid[i, j] = 3
            elif goal_grid_2d[i, j] == 1:
                display_grid[i, j] = 1
            elif agent_grid_2d[i, j] == 1:
                display_grid[i, j] = 2
    
    # Define colors and norm
    colors = ['white', 'green', 'blue']  # empty, goal, agent, both
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(boundaries=[-0.5, 0.5, 1.5, 2.5, 3.5], ncolors=4)
    
    fig, ax = plt.subplots()
    ax.imshow(display_grid, cmap=cmap, norm=norm, origin='upper')
    
    # Grid lines
    ax.set_xticks(np.arange(-0.5, w, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, h, 1), minor=True)
    ax.grid(which='minor', color='black', linestyle='-', linewidth=1)
    ax.tick_params(which='minor', size=0)
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Legend
    legend_elements = [
        Patch(facecolor='white', edgecolor='black', label='Empty'),
        Patch(facecolor='green', edgecolor='black', label='Goal'),
        Patch(facecolor='blue', edgecolor='black', label='Agent'),
    ]
    ax.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.show()



def update_agent_grid(agent_grid, h, w, action_json):
    action_data = json.loads(action_json)
    action = action_data.get("action", "").strip().upper()
    if action not in ["UP", "DOWN", "LEFT", "RIGHT"]:
        raise ValueError(f"Invalid action: {action}")
    
    grid_2d = np.array(agent_grid).reshape(h, w)
    
    pos = np.argwhere(grid_2d == 1)
    if len(pos) == 0:
        raise ValueError("No agent found in the grid")
    if len(pos) > 1:
        raise ValueError("Multiple agents found in the grid")
    
    i, j = pos[0]
    
    new_i, new_j = i, j
    if action == "UP":
        new_i = max(i - 1, 0)
    elif action == "DOWN":
        new_i = min(i + 1, h - 1)
    elif action == "LEFT":
        new_j = max(j - 1, 0)
    elif action == "RIGHT":
        new_j = min(j + 1, w - 1)
    
    grid_2d[i, j] = 0
    grid_2d[new_i, new_j] = 1
    
    return grid_2d.flatten().tolist()