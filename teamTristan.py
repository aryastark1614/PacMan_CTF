# teamTristan.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).

import mcts
from captureAgents import CaptureAgent
import random, time, util
import math
from game import Directions
import game
from mcts import MCTS


#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed,
               first = 'attackingAgent', second = 'attackingAgent'):
  """
  This function should return a list of two agents that will form the
  team, initialized using firstIndex and secondIndex as their agent
  x numbers.  isRed is True if the red team is being created, and
  will be False if the blue team is being created.

  As a potentially helpful development aid, this function can take
  additional string-valued keyword arguments ("first" and "second" are
  such arguments in the case of this function), which will come from
  the --redOpts and --blueOpts command-line arguments to capture.py.
  For the nightly contest, however, your team will be created without
  any extra arguments, so you should make sure that the default
  behavior is what you want for the nightly contest.
  """

  # The following line is an example only; feel free to change it.
  return [eval(first)(firstIndex), eval(second)(secondIndex)]

##########
# Agents #
##########

class Agents(CaptureAgent):

  def registerInitialState(self, gameState):
    """
    This method handles the initial setup of the
    agent to populate useful fields (such as what team
    we're on).

    A distanceCalculator instance caches the maze distances
    between each pair of positions, so your agents can use:
    self.distancer.getDistance(p1, p2)

    IMPORTANT: This method may run for at most 15 seconds.
    """

    '''
    Make sure you do not delete the following line. If you would like to
    use Manhattan distances instead of maze distances in order to save
    on initialization time, please take a look at
    CaptureAgent.registerInitialState in captureAgents.py.
    '''
    CaptureAgent.registerInitialState(self, gameState)

    '''
    Your initialization code goes here, if you need any.
    '''
    self.simulation_time = 0.3
    self.max_depth = 1000

    self.start = gameState.getAgentPosition(self.index)
    self.child_state = {}
    self.current_root = mcts.MCTSNode(gameState)
    self.numberOfAgents = gameState.getNumAgents()
    self.me = self.index
    self.mate = self.getTeam(gameState)[1] if self.getTeam(gameState)[0] == self.index else self.getTeam(gameState)[0]

  def get_index(self, start = 0, n = 4):
    while True:
        for i in range(n):
            yield (start + i) % n

  def next_index(self, x):
    if x < 0 or x > 3:
      raise ValueError("Input must be an integer between 0 and 3 inclusive.")
    return (x + 1) % 4

  def is_terminal(self, state):
    if hasattr(state.data, '_win') and state.data._win:
      return True
    if hasattr(state.data, '_lose') and state.data._lose:
      return True
    return False


class attackingAgent(Agents):

  def chooseAction(self, gameState):

    start_time = time.time()

    sindex = self.get_index(self.index, self.numberOfAgents)
    sim_index = next(sindex)

    self.current_root = mcts.MCTSNode(gameState, None, None, 1, self.index)

    iteration = 0
    treelevel = 0

    #Start MCTS Tree Loop
    while time.time() - start_time < self.simulation_time:
      iteration += 1
      #Traverse the tree
      current_node = self.current_root
      is_leaf = False

      #Check if current node is a leaf
      while current_node.children:

          #sim_index = next(sindex)
          sim_index = self.next_index(current_node.agentIndex)
          uct_values = {child: mcts.MCTSNode.uct_value(child) for child in current_node.children}
          current_node = max(uct_values, key=uct_values.get)
          treelevel += 1


      #Expand
      if not self.is_terminal(current_node.state) and current_node.visits > 0:
        actions = current_node.state.getLegalActions(sim_index)
        if actions:  # Make sure there are legal actions
          for action in actions:
            next_state = current_node.state.generateSuccessor(sim_index, action)
            new_child = mcts.MCTSNode(next_state, current_node, action, 0, self.next_index(self.index))
            current_node.children.append(new_child)
          if current_node.children:
            current_node = current_node.children[0]

      #Simulate
      sim_gameState = current_node.state
      current_sim_index = sim_index
      current_depth = 0

      while not self.is_terminal(sim_gameState) and current_depth < self.max_depth:
        actions = sim_gameState.getLegalActions(current_sim_index)
        if not actions:
          break
        action = random.choice(actions)
        sim_gameState = sim_gameState.generateSuccessor(current_sim_index, action)
        current_sim_index = (current_sim_index + 1) % self.numberOfAgents
        current_depth += 1

      # Evaluate
      if not sim_gameState:
        return 0

      if hasattr(sim_gameState.data, '_win') and sim_gameState.data._win:
        return 1000

      foodList = sim_gameState.getRedFood().asList() if self.red else sim_gameState.getBlueFood().asList()
      if not foodList:
        return 1000

      pacmanState = sim_gameState.getAgentState(self.me)
      pacmanPosition = pacmanState.getPosition()

      foodDistances = [self.distancer.getDistance(pacmanPosition, food) for food in foodList]
      minFoodDist = min(foodDistances) if foodDistances else 0
      foodCarrying = pacmanState.numCarrying

      enemies = self.getOpponents(sim_gameState)
      enemy_positions = [sim_gameState.getAgentPosition(i) for i in enemies if
                         sim_gameState.getAgentPosition(i) is not None]

      max_maze_dist = 20
      enemy_distance = min([self.distancer.getDistance(pacmanPosition, pos) for pos in
                            enemy_positions]) if enemy_positions else max_maze_dist

      home_positions = []
      if self.red:
        home_x = (sim_gameState.getWalls().width // 2) - 1
      else:
        home_x = (sim_gameState.getWalls().width // 2)

      for y in range(sim_gameState.getWalls().height):
        if not sim_gameState.hasWall(home_x, y):
          home_positions.append((home_x, y))

      home_distances = [self.distancer.getDistance(pacmanPosition, pos) for pos in home_positions]
      min_home_dist = min(home_distances) if home_distances else 0

      # Weights
      w_food_dist = 40.0
      w_food_carrying = 20.0
      w_enemy_dist = 5.0
      w_home_dist = 5.0

      food_reward = w_food_dist * (1 - (minFoodDist / max_maze_dist))
      carrying_reward = w_food_carrying * (foodCarrying / 10.0)  # Assuming 10 is max food to carry
      enemy_penalty = w_enemy_dist * (1 - (enemy_distance / max_maze_dist))

      home_reward = 0
      if foodCarrying > 0:
        home_reward = w_home_dist * (1 - (min_home_dist / max_maze_dist)) * foodCarrying

      raw_reward = food_reward + carrying_reward + home_reward - enemy_penalty
      reward = max(0, min(1000, raw_reward * 10))

      #Backpropagate
      while current_node.parent:
        current_node.visits += 1
        current_node.reward += reward
        current_node = current_node.parent

    best_node = max(self.current_root.children, key=lambda node: node.reward)
    if not best_node.action or best_node.action not in gameState.getLegalActions(self.index):
        print(f"{best_node.action} is not part of {gameState.getLegalActions(self.index)}")
        print("No best node found, returning random action")
        return random.choice(gameState.getLegalActions(self.index))
    print(f"Executing action {best_node.action}")
    return best_node.action