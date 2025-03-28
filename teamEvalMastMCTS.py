# teamEvalMastMCTS.py
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
               first = 'attackingAgent', second = 'defendingAgent'):
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
    self.simulation_time = 0.5
    self.max_depth = 12
    self.treelevel = 0
    self.state_hash_mapping = {}

    self.current_root = None
    self.current_node = None
    self.sim_gameState = None
    self.reuse = 0
    self.explored = [(gameState.getAgentPosition(self.index), 0)]

    self.start = gameState.getAgentPosition(self.index)
    self.numberOfAgents = gameState.getNumAgents()
    self.me = self.index
    self.mate = self.getTeam(gameState)[1] if self.getTeam(gameState)[0] == self.index else self.getTeam(gameState)[0]
    self.max_food_carry = 3
    self.mast_stats = {}

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

  def select_action_mast(self, legal_actions, exploration_rate=0.2):
    if random.random() < exploration_rate:
      return random.choice(legal_actions)

    action_scores = {}
    for action in legal_actions:
      if action in self.mast_stats and self.mast_stats[action]['count'] > 0:
        total_reward = self.mast_stats[action]['total']
        action_count = self.mast_stats[action]['count']
        normalized_score = total_reward / (action_count + 1)
        action_scores[action] = normalized_score
      else:
        action_scores[action] = 0

    if action_scores:
      max_score = max(action_scores.values())
      best_actions = [
        action for action, score in action_scores.items()
        if math.isclose(score, max_score, rel_tol=1e-9)
      ]
      return random.choice(best_actions)

    return random.choice(legal_actions)

  def update_mast(self, simulation_actions, reward, decay_factor=0.9):
    for action in simulation_actions:
      if action not in self.mast_stats:
        self.mast_stats[action] = {
          'total': reward,
          'count': 1,
          'last_update': time.time()
        }
      else:
        current_time = time.time()
        time_since_update = current_time - self.mast_stats[action].get('last_update', current_time)
        decay_multiplier = decay_factor ** time_since_update

        self.mast_stats[action]['total'] = (
                self.mast_stats[action]['total'] * decay_multiplier + reward
        )
        self.mast_stats[action]['count'] += 1
        self.mast_stats[action]['last_update'] = current_time

    if len(self.mast_stats) > 100:
      sorted_actions = sorted(
        self.mast_stats.items(),
        key=lambda x: x[1]['count'],
        reverse=True
      )
      self.mast_stats = dict(sorted_actions[:100])

  def traverse(self):
    while self.current_node.children:
      self.next_index(self.current_node.agentIndex)
      uct_values = {child: mcts.MCTSNode.uct_value(child) for child in self.current_node.children}
      self.current_node = max(uct_values, key=uct_values.get)
      self.treelevel += 1

  def expand(self, sim_index):
    if not self.is_terminal(self.current_node.state) and self.current_node.visits > 0:
      actions = self.current_node.state.getLegalActions(self.index)
      actions.remove('Stop')
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
    """
    Simulation method enhanced to work with improved MAST

    Args:
        sim_index (int): Starting agent index for simulation
    """
    self.sim_gameState = self.current_node.state
    current_sim_index = sim_index
    current_depth = 0
    simulation_actions = []

    while not self.is_terminal(self.sim_gameState) and current_depth < self.max_depth:
      actions = self.sim_gameState.getLegalActions(current_sim_index)
      actions = [a for a in actions if a != 'Stop']  # Remove 'Stop' action

      if not actions:
        break

      action = self.select_action_mast(actions)

      simulation_actions.append(action)
      self.sim_gameState = self.sim_gameState.generateSuccessor(current_sim_index, action)

      current_sim_index = (current_sim_index + 1) % self.numberOfAgents
      current_depth += 1

    reward = self.evaluate()

    self.update_mast(simulation_actions, reward)

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

    print(f'\rSearched {iteration} iterations in {(time.time() - start_time):.4f} seconds for Attacker. Reused {self.reuse} nodes. ', end='', flush=True)

    best_node = max(self.current_root.children, key=lambda node: node.reward)
    if not best_node.action or best_node.action not in gameState.getLegalActions(self.index):
        print(f"{best_node.action} is not part of {gameState.getLegalActions(self.index)}")
        print("No best node found, returning random action")
        return random.choice(gameState.getLegalActions(self.index))
    print(f"Executing action {best_node.action}")
    return best_node.action

  # Paste this into the patch instructions
  def evaluate(self):

    # -- Weights --
    W_SCORE = 10.0  # Weight for the current game score
    W_INV_FOOD_DIST = 10.0  # Weight for (1 / distance to food)
    W_CARRY_BONUS = 10  # Weight for each piece of carried food
    W_RETURN_HOME = 50.0  # Extra reward for moving closer to boundary if carrying
    W_GHOST_DANGER = 10.0  # Penalty multiplier for being near ghosts
    W_CAPSULE_INV_DIST = 10.0  # Weight for (1 / distance to capsule)

    if not self.sim_gameState:
      return 0

    if getattr(self.sim_gameState.data, '_win', False):
      return 9999  # large positive
    if getattr(self.sim_gameState.data, '_lose', False):
      return -9999  # large negative

    if self.red:
      game_score = self.getScore(self.sim_gameState) * W_SCORE
    else:
      game_score = -self.getScore(self.sim_gameState) * W_SCORE

    my_state = self.sim_gameState.getAgentState(self.index)
    my_pos = my_state.getPosition()
    if not my_pos:
      return game_score

    if self.red:
      food_list = self.sim_gameState.getBlueFood().asList()
      capsules = self.sim_gameState.getBlueCapsules()
    else:
      food_list = self.sim_gameState.getRedFood().asList()
      capsules = self.sim_gameState.getRedCapsules()

    if len(food_list) > 0:
      closest_food_dist = min(self.distancer.getDistance(my_pos, f) for f in food_list)
      inv_food_dist = (1.0 / (closest_food_dist + 1.0)) * W_INV_FOOD_DIST
    else:
      inv_food_dist = 0


    width = self.sim_gameState.data.layout.width
    height = self.sim_gameState.data.layout.height
    walls = self.sim_gameState.getWalls().asList()

    if self.red:
      home_territory = [(float(x), float(y)) for x in range(1, width // 2)
                        for y in range(1, height)
                        if (x, y) not in walls]
      boundary_positions = [(float(x), float(y)) for x in range(width // 2) for y in range(1, height) if (x, y) not in walls]

      enemy_territory = [(float(x), float(y)) for x in range(width // 2, width - 1)
                         for y in range(1, height)
                         if (x, y) not in walls]
    else:
      home_territory = [(float(x), float(y)) for x in range(width // 2, width - 1)
                        for y in range(1, height)
                        if (x, y) not in walls]
      boundary_positions = [(float(x), float(y)) for x in range(width // 2 + 1) for y in range(1, height) if
                            (x, y) not in walls]
      enemy_territory = [(float(x), float(y)) for x in range(1, width // 2)
                         for y in range(1, height)
                         if (x, y) not in walls]

    carrying = my_state.numCarrying

    if carrying < 3:
      carry_bonus = carrying * W_CARRY_BONUS
    else:
        carry_bonus = -(carrying * W_CARRY_BONUS)

    if boundary_positions:
      closest_boundary_dist = min(self.distancer.getDistance(my_pos, b) for b in boundary_positions)
      if carrying > 2:
        return_home_bonus = (1.0 / (closest_boundary_dist + 1.0)) * (W_RETURN_HOME * 10)
      else:
        ratio = float(carrying) / float(self.max_food_carry)
        return_home_bonus = (1.0 / (closest_boundary_dist + 1.0)) * W_RETURN_HOME * ratio
    else:
      return_home_bonus = 0

    if capsules:
      closest_capsule_dist = min(self.distancer.getDistance(my_pos, cap) for cap in capsules)
      inv_capsule_dist = (1.0 / (closest_capsule_dist + 1.0)) * W_CAPSULE_INV_DIST
    else:
      inv_capsule_dist = 0

    enemies = [self.sim_gameState.getAgentState(i) for i in self.getOpponents(self.sim_gameState)]
    defending_ghosts = [
      e for e in enemies
      if not e.isPacman and e.getPosition() is not None
    ]
    if defending_ghosts:
      ghost_distances = [self.distancer.getDistance(my_pos, g.getPosition()) for g in defending_ghosts]
      closest_ghost_dist = min(ghost_distances)
      if closest_ghost_dist < 5:
        ghost_penalty = -W_GHOST_DANGER * (5.0 - closest_ghost_dist)
      else:
        ghost_penalty = 0
    else:
      ghost_penalty = 0

    reward = 0.0
    reward += game_score
    reward += inv_food_dist
    reward += carry_bonus
    reward += return_home_bonus
    reward += inv_capsule_dist
    reward += ghost_penalty

    return reward


class defendingAgent(Agents):

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

    print(f'\rSearched {iteration} iterations in {(time.time() - start_time):.4f} seconds for Attacker. Reused {self.reuse} nodes. ', end='', flush=True)

    best_node = max(self.current_root.children, key=lambda node: node.reward)
    if not best_node.action or best_node.action not in gameState.getLegalActions(self.index):
        print(f"{best_node.action} is not part of {gameState.getLegalActions(self.index)}")
        print("No best node found, returning random action")
        return random.choice(gameState.getLegalActions(self.index))
    print(f"Executing action {best_node.action}")
    return best_node.action

  # Paste this into the patch instructions
  def evaluate(self):
    """
    A defensive evaluation function that emphasizes:
      - Intercepting enemy Pacmen on our side
      - Guarding capsules (if available)
      - Eating enemy Pacmen quickly when we are not scared
      - Avoiding/fleeing if we are scared

    Weights are made explicit so you can tune them.
    """

    # -- Weights (adjust as desired) --
    W_SCORE               = 50.0   # Weight for the current game score
    W_INV_ENEMY_DIST      = 250.0  # Reward for closing distance to enemy Pacman on our side
    W_ENEMY_CARRY_BONUS   = 250.0   # Additional bonus if that enemy is carrying food
    W_SCARED_PENALTY      = 100.0  # Penalty for engaging if we're scared
    W_CAPSULE_PROTECTION  = 25.0   # Encouragement to guard capsules

    # 1. If there's no simulated state, return 0
    if not self.sim_gameState:
        return 0

    # 2. Win/Lose check
    if getattr(self.sim_gameState.data, '_win', False):
        return 9999  # huge positive
    if getattr(self.sim_gameState.data, '_lose', False):
        return -9999 # huge negative

    # 3. Base game score from our perspective
    game_score = self.getScore(self.sim_gameState) * W_SCORE

    # 4. Gather info about our agent
    my_state = self.sim_gameState.getAgentState(self.index)
    my_pos   = my_state.getPosition()
    if not my_pos:
        # If we don't know our position, fall back to the base score
        return game_score

    # 5. Determine relevant capsules (these are your team's capsules)
    #    On defense, you typically want to guard your capsules so the enemy can't get them.
    if self.red:
        capsules = self.sim_gameState.getRedCapsules()
    else:
        capsules = self.sim_gameState.getBlueCapsules()

    # 6. Identify enemy Pacmen
    #    We want to penalize distance if they're in our territory.
    opponents = [self.sim_gameState.getAgentState(i) for i in self.getOpponents(self.sim_gameState)]
    visible_enemy_pacmen = [
        o for o in opponents
        if o.isPacman and o.getPosition() is not None
    ]
    width = self.sim_gameState.data.layout.width
    height = self.sim_gameState.data.layout.height
    walls = self.sim_gameState.getWalls().asList()
    if self.red:
      home_territory = [(float(x), float(y)) for x in range(1, width // 2)
                        for y in range(1, height)
                        if (x, y) not in walls]
      boundary_positions = [(float(x), float(y)) for x in range(width // 2) for y in range(1, height) if (x, y) not in walls]

      enemy_territory = [(float(x), float(y)) for x in range(width // 2, width - 1)
                         for y in range(1, height)
                         if (x, y) not in walls]
    else:
      home_territory = [(float(x), float(y)) for x in range(width // 2, width - 1)
                        for y in range(1, height)
                        if (x, y) not in walls]
      boundary_positions = [(float(x), float(y)) for x in range(width // 2 + 1) for y in range(1, height) if
                            (x, y) not in walls]
      enemy_territory = [(float(x), float(y)) for x in range(1, width // 2)
                         for y in range(1, height)
                         if (x, y) not in walls]

    # 7. Evaluate distance to visible Pacmen on our side
    #    If an enemy is on our side, we want to be close enough to eat them if we're not scared.
    inv_enemy_dist_total = 0.0

    # Are we currently scared? If our 'scaredTimer' > 0, it means we can't eat them right now.
    we_are_scared = (my_state.scaredTimer > 0)

    for enemy in visible_enemy_pacmen:
        enemy_pos = enemy.getPosition()
        dist = self.distancer.getDistance(my_pos, enemy_pos)

        # The closer we are, the bigger the reward (unless we're scared)
        inv_dist = (1.0 / (dist + 1.0)) if dist > 0 else 1.0

        # If we're scared, there's a penalty for being *too* close
        if we_are_scared:
            # The closer we get, the more we risk being eaten
            inv_enemy_dist_total -= W_SCARED_PENALTY * inv_dist
        else:
            # We want to get closer to eat them
            inv_enemy_dist_total += W_INV_ENEMY_DIST * inv_dist

        # If the enemy is carrying food, we add an extra bonus for chasing them
        if enemy.numCarrying > 0 and not we_are_scared:
            inv_enemy_dist_total += W_ENEMY_CARRY_BONUS * float(enemy.numCarrying)

    if my_pos in enemy_territory:
      pos_penalty = 1000
    else:
      pos_penalty = - 100

    enemy_penalty = 0
    for enemy in visible_enemy_pacmen:
        enemy_pos = enemy.getPosition()
        if enemy_pos in home_territory:
          enemy_penalty = 1000
        else: enemy_penalty = -100

    # 8. Capsule protection
    #    We may want to be near capsules on our side so that enemy Pacmen can’t easily grab them.
    #    The idea: The closer we are to our capsules, the better we can defend them.
    capsule_guard_value = 0.0
    if capsules:
        closest_capsule_dist = min(self.distancer.getDistance(my_pos, c) for c in capsules)
        # The closer we are, the more we are “protecting” it
        inv_capsule_dist = (1.0 / (closest_capsule_dist + 1.0))
        capsule_guard_value += W_CAPSULE_PROTECTION * inv_capsule_dist

    # 9. Sum the final evaluation
    evaluation = 0.0
    evaluation += game_score
    evaluation += inv_enemy_dist_total
    evaluation += capsule_guard_value
    evaluation -= enemy_penalty
    evaluation -= pos_penalty

    return evaluation

