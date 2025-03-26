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
    self.simulation_time = 0.4
    self.max_depth = 60
    self.treelevel = 0
    self.state_hash_mapping = {}

    self.current_root = None
    self.current_node = None
    self.sim_gameState = None
    self.reuse = 0

    self.start = gameState.getAgentPosition(self.index)
    self.numberOfAgents = gameState.getNumAgents()
    self.me = self.index
    self.mate = self.getTeam(gameState)[1] if self.getTeam(gameState)[0] == self.index else self.getTeam(gameState)[0]
    self.max_food_carry = 2

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

  def get_state_hash(self, gameState):
    positions = tuple(gameState.getAgentPosition(i) for i in range(gameState.getNumAgents()))
    return hash(positions)

  def getRoot(self, gameState):
    found_existing = False
    new_state_hash = self.get_state_hash(gameState)
    for node in self.state_hash_mapping.values():
      if new_state_hash == self.get_state_hash(node.state):
        found_existing = True
        self.current_root = self.state_hash_mapping[new_state_hash]
        self.reuse +=1
        break

    if not found_existing:
      self.current_root = mcts.MCTSNode(gameState, None, None, 1, self.index)
      self.state_hash_mapping[new_state_hash] = self.current_root

  def traverse(self):
    while self.current_node.children:
      self.next_index(self.current_node.agentIndex)
      uct_values = {child: mcts.MCTSNode.uct_value(child) for child in self.current_node.children}
      self.current_node = max(uct_values, key=uct_values.get)
      self.treelevel += 1

  def expand(self, sim_index):
    if not self.is_terminal(self.current_node.state) and self.current_node.visits > 0:
      actions = self.current_node.state.getLegalActions(sim_index)
      if actions:  # Make sure there are legal actions
        for action in actions:
          next_state = self.current_node.state.generateSuccessor(sim_index, action)
          state_hash = self.get_state_hash(next_state)
          new_child = mcts.MCTSNode(next_state, self.current_node, action, 0, self.next_index(self.index))
          self.state_hash_mapping[state_hash] = new_child
          self.current_node.children.append(new_child)
        if self.current_node.children:
          uct_values = {child: mcts.MCTSNode.uct_value(child) for child in self.current_node.children}
          self.current_node = max(uct_values, key=uct_values.get)

  def simulate(self, sim_index):
    self.sim_gameState = self.current_node.state
    current_sim_index = sim_index
    current_depth = 0

    while not self.is_terminal(self.sim_gameState) and current_depth < self.max_depth:
      actions = self.sim_gameState.getLegalActions(current_sim_index)
      if not actions:
        break
      action = random.choice(actions)
      self.sim_gameState = self.sim_gameState.generateSuccessor(current_sim_index, action)
      current_sim_index = (current_sim_index + 1) % self.numberOfAgents
      current_depth += 1

  def backpropagate(self, reward):
    while self.current_node.parent:
      self.current_node.visits += 1
      self.current_node.reward += reward
      self.current_node = self.current_node.parent

class attackingAgent(Agents):

  def chooseAction(self, gameState):

    start_time = time.time()

    sindex = self.get_index(self.index, self.numberOfAgents)
    sim_index = next(sindex)

    #Set the root node
    self.getRoot(gameState)

    iteration = 0
    self.treelevel = 0

    #Start MCTS Tree Loop
    while time.time() - start_time < self.simulation_time:

      iteration += 1
      self.current_node = self.current_root

      #Traverse to leaf
      self.traverse()

      #Expand
      self.expand(sim_index)

      #Simulate
      self.simulate(sim_index)

      #Evaluate
      reward = self.evaluate()

      #Backpropagate
      self.backpropagate(reward)
      print(f'\rCurrently at iteration: {iteration}', end='', flush=True)

    print(f'\rSearched {iteration} iterations in {(time.time() - start_time):.4f} seconds. Reused {self.reuse} nodes. ', end='', flush=True)

    best_node = max(self.current_root.children, key=lambda node: node.reward)
    if not best_node.action or best_node.action not in gameState.getLegalActions(self.index):
        print(f"{best_node.action} is not part of {gameState.getLegalActions(self.index)}")
        print("No best node found, returning random action")
        return random.choice(gameState.getLegalActions(self.index))
    print(f"Executing action {best_node.action}")
    return best_node.action

  def evaluate(self):
    if not self.sim_gameState:
      return 0

    if hasattr(self.sim_gameState.data, '_win') and self.sim_gameState.data._win:
      return 1000

    foodPositions = self.sim_gameState.getRedFood().asList() if self.red else self.sim_gameState.getBlueFood().asList()
    if not foodPositions:
      return 1000


    pacmanState = self.sim_gameState.getAgentState(self.me)
    pacmanPosition = pacmanState.getPosition()
    foodDist = [self.distancer.getDistance(pacmanPosition, food) for food in foodPositions]

    foodCarrying = pacmanState.numCarrying

    enemies = self.getOpponents(self.sim_gameState)
    enemy_positions = [self.sim_gameState.getAgentPosition(i) for i in enemies if
                       self.sim_gameState.getAgentPosition(i) is not None]

    max_maze_dist = self.sim_gameState.data.layout.width - 2
    enemy_distance = min([self.distancer.getDistance(pacmanPosition, pos) for pos in
                          enemy_positions]) if enemy_positions else max_maze_dist

    width = self.sim_gameState.data.layout.width
    height = self.sim_gameState.data.layout.height
    walls = self.sim_gameState.getWalls().asList()

    if self.red:
      home_territory = [(float(x), float(y)) for x in range(1, width // 2)
                        for y in range(1, height)
                        if (x, y) not in walls]
      enemy_territory =  [(float(x), float(y)) for x in range(width // 2, width - 1)
                        for y in range(1, height)
                        if (x, y) not in walls]
    else:
      home_territory = [(float(x), float(y)) for x in range(width // 2, width - 1)
                        for y in range(1, height)
                        if (x, y) not in walls]
      enemy_territory = [(float(x), float(y)) for x in range(1, width // 2)
                         for y in range(1, height)
                         if (x, y) not in walls]

    # Weights
    w_food_dist = 40.0
    w_food_carrying = 20.0
    w_enemy_dist = 5.0
    w_home_dist = 5.0

    distPacEnem = min([self.distancer.getDistance(pacmanPosition, pos) for pos in
                          enemy_positions]) if enemy_positions else max_maze_dist

    distPacFood = min(foodDist) if foodDist else 0
    distPacHome = min([self.distancer.getDistance(pacmanPosition, pos) for pos in home_territory])
    distPacCapt = min([self.distancer.getDistance(pacmanPosition, pos) for pos in enemy_territory])

    position_score = 0
    food_score = 0

    if foodCarrying >= self.max_food_carry and pacmanPosition not in home_territory:
      position_score -= 1000

    if foodCarrying < self.max_food_carry and pacmanPosition in home_territory:
      position_score -= 1000

    if foodCarrying < self.max_food_carry:
      food_score -= 1000

    if foodCarrying >= self.max_food_carry:
      food_score += 1000


    food_reward = w_food_dist * (1 - (distPacFood / max_maze_dist))
    carrying_reward = w_food_carrying * (foodCarrying / self.max_food_carry)
    enemy_penalty = w_enemy_dist * (1 - (distPacEnem / max_maze_dist))
    territory_reward = foodCarrying * (1 - (distPacEnem / max_maze_dist))

    home_reward = 0
    if foodCarrying > 0:
      home_reward = w_home_dist * (1 - (distPacHome / max_maze_dist)) * foodCarrying

    raw_reward = food_reward + carrying_reward + home_reward - enemy_penalty + territory_reward
    reward = max(0, min(1000, raw_reward * 10))

    return reward