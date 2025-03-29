# teamEvalMCTS.py
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

import math
import random
import time

import game
import mcts
import util
from captureAgents import CaptureAgent
from game import Directions
from mcts import MCTS

#################
# Team creation #
#################

# Creates a team from the attacking agent and the defending agent
def createTeam(firstIndex, secondIndex, isRed, first = 'attackingAgent', second = 'defendingAgent'):
  return [eval(first)(firstIndex), eval(second)(secondIndex)]

##########
# Agents #
##########

class MCTSNode:
    def __init__(self, gameState, parent=None, action=None, visits=0, agentIndex=None):

        self.gameState = gameState
        self.agentIndex = agentIndex
        self.action = action

        self.parent = parent
        self.children = []

        self.visits = visits
        self.reward = 0.0

        self.legalActions = [action for action in gameState.getLegalActions(self.agentIndex) if action != 'Stop']
        self.unexploredActions = self.legalActions[:]


class MCTS:
    def __init__(self, gameState, max_depth=50, exploration_constant=1.41):

        self.simulation_time = 0.9
        self.max_depth = max_depth
        self.depth = 0
        self.exploration = exploration_constant
        self.current_node = None
        self.current_root = None
        self.game_State = gameState
        self.current_game_State = gameState
        self.state_hash_mapping = {}
        self.action_values = {}
        self.action_visits = {}
        self.rave_values = {}
        self.rave_visits = {}

    def uct_best_child(self, exploration_constant=1.41):
        best_value = -float('inf')
        best_child = None
        for child in self.current_node.children:
            if child.visits == 0:
                value = float('inf')
            else:
                value = (child.reward / child.visits) + exploration_constant * math.sqrt(
                    math.log(self.current_node.visits) / child.visits)
            if value > best_value:
                best_value = value
                best_child = child
        return best_child

    def get_state_hash(self, gameState, agent_index=None):
        """Generate a consistent hash for a game state"""
        if agent_index is None:
            agent_index = self.agent_index
        agent_position = gameState.getAgentPosition(agent_index)
        return hash((agent_position, tuple(gameState.data.food.asList())))

    def getRoot(self, gameState, agent_index):
        """Establish or reuse the root node for the search"""
        new_state_hash = self.get_state_hash(gameState, agent_index)

        # Check if we have an existing root that matches this state
        if new_state_hash in self.state_hash_mapping:
            self.current_root = self.state_hash_mapping[new_state_hash]
            self.reuse += 1
        else:
            # Create a new root node
            self.current_root = MCTSNode(gameState, None, None, 1, agent_index)
            self.state_hash_mapping[new_state_hash] = self.current_root

        self.current_node = self.current_root
        return self.current_root

    def traverse(self):
        """Traverse the tree to select the most promising node"""
        self.depth = 0
        if not self.current_node:
            self.current_node = self.current_root

        while self.current_node.children:
            self.current_node = self.uct_best_child(self.exploration)
            self.depth += 1

    def expand(self):
        """Expand the current node by adding child nodes"""
        if (not self.current_node.gameState.data._win and
                not self.current_node.gameState.data._lose and
                self.current_node.visits > 0 and
                self.depth < self.max_depth):

            legal_actions = [action for action in self.current_node.gameState.getLegalActions(self.agent_index) if action != 'Stop']
            unexplored_actions = [action for action in legal_actions
                                  if action not in [child.action for child in self.current_node.children]]

            if unexplored_actions:
                action = random.choice(unexplored_actions)
                print(unexplored_actions)
                next_state = self.current_node.gameState.generateSuccessor(self.current_node.agentIndex, action)


                new_child = MCTSNode(next_state, self.current_node, action, 0, self.current_node.agentIndex)
                self.current_node.children.append(new_child)

                state_hash = self.get_state_hash(next_state)
                self.state_hash_mapping[state_hash] = new_child

                self.current_node = new_child
                self.depth += 1

    def simulate(self):
        self.sim_node = self.current_node

        while (
                not self.sim_node.gameState.data._win
                and not self.sim_node.gameState.data._lose
                and self.depth <= self.max_depth):
            actions = [action for action in self.sim_node.gameState.getLegalActions(self.sim_node.agentIndex) if action != 'Stop']
            if not actions:
                break
            action = random.choice(actions)
            self.sim_node.gameState = self.sim_node.gameState.generateSuccessor(self.sim_node.agentIndex, action)
            #self.sim_node = mcts.MCTSNode(self.sim_node, 0, action, self.sim_node.agentIndex)
            self.depth += 1

        return self.sim_node.gameState

    def backpropagate(self, reward):
        while self.current_node.parent:
            self.current_node.visits += 1
            self.current_node.reward += reward
            self.update_rave(self.current_node.action, reward)
            self.current_node = self.current_node.parent
    
    def update_rave(self, action, reward):
        if action not in self.rave_values:
            self.rave_values[action] = 0
            self.rave_visits[action] = 0

        self.rave_values[action] += reward
        self.rave_visits[action] += 1
        
    def get_rave_value(self, action, num_simulations, score):
        rave_value = self.rave_values.get(action, 0)
        rave_visits = self.rave_visits.get(action, 0)

        combined_rave_value = (1 - self.exploration) * (score / (num_simulations + 0.0001)) + \
                         self.exploration * (rave_value / (rave_visits + 0.0001))
        return combined_rave_value


class Agents(CaptureAgent):

  def registerInitialState(self, gameState):
    CaptureAgent.registerInitialState(self, gameState)

    self.simulation_time = 0.4
    self.max_depth = 8
    self.treelevel = 0
    self.state_hash_mapping = {}

    self.current_root = None
    self.current_node = None
    self.reuse = 0
    self.exploration = 0.1
    self.action_values = {}
    self.action_visits = {}
    self.rave_values = {}
    self.rave_visits = {}
    self.explored = [(gameState.getAgentPosition(self.index), 0)]

    self.start = gameState.getAgentPosition(self.index)
    self.numberOfAgents = gameState.getNumAgents()
    self.me = self.index
    self.mate = self.getTeam(gameState)[1] if self.getTeam(gameState)[0] == self.index else self.getTeam(gameState)[0]
    self.max_food_carry = 3

    self.food_count = int(len(self.getFood(gameState).asList()))
    self.map_width = gameState.data.layout.width
    self.map_height = gameState.data.layout.height
    self.map_walls = gameState.getWalls().asList()
    self.home_territory,\
    self.enem_territory,\
    self.home_bounds,\
    self.enem_bounds= self.getMapProperties()
    self.sim_gameState = gameState

  def getMapProperties(self):
      if self.red:
          home_territory = [(float(x), float(y)) for x in range(1, self.map_width // 2)
                            for y in range(1, self.map_height)
                            if (x, y) not in self.map_walls]
          home_bounds = [(float(x), float(y)) for x in range(self.map_width // 2) for y in range(1, self.map_height) if
                         (x, y) not in self.map_walls]
          enem_bounds = [(float(x), float(y)) for x in range(self.map_width // 2 + 1) for y in range(1, self.map_height)
                         if
                         (x, y) not in self.map_walls]

          enemy_territory = [(float(x), float(y)) for x in range(self.map_width // 2, self.map_width - 1)
                             for y in range(1, self.map_height) if (x, y) not in self.map_walls]
      else:
          home_territory = [(float(x), float(y)) for x in range(self.map_width // 2, self.map_width - 1)
                            for y in range(1, self.map_height) if (x, y) not in self.map_walls]
          enem_bounds = [(float(x), float(y)) for x in range(self.map_width // 2) for y in range(1, self.map_height) if
                         (x, y) not in self.map_walls]
          home_bounds = [(float(x), float(y)) for x in range(self.map_width // 2 + 1) for y in range(1, self.map_height)
                         if
                         (x, y) not in self.map_walls]
          enemy_territory = [(float(x), float(y)) for x in range(1, self.map_width // 2)
                             for y in range(1, self.map_height)
                             if (x, y) not in self.map_walls]

      return home_territory, enemy_territory, home_bounds, enem_bounds


class attackingAgent(Agents):

    def chooseAction(self, gameState):
        start_time = time.time()
        tree = MCTS(gameState)
        tree.agent_index = self.index
        tree.getRoot(gameState, self.index)

        iteration = 0
        while time.time() - start_time < self.simulation_time:
            iteration += 1
            tree.traverse()
            tree.expand()
            self.sim_gameState = tree.simulate()
            reward = self.evaluate()
            tree.backpropagate(reward)
            for node in tree.current_root.children:
                action = node.action
                tree.rave_values[action] = tree.rave_values.get(action, 0) + reward
                tree.rave_visits[action] = tree.rave_visits.get(action, 0) + 1
            print(iteration)

        actions = [node.action for node in tree.current_root.children]
        simulations = {node.action: node.visits for node in tree.current_root.children}
        scores = {node.action: node.reward for node in tree.current_root.children}

        best_action = max(actions, key=lambda a: tree.get_rave_value(a, simulations[a], scores[a]))
        
        return best_action

    def evaluate(self):

        # -- Weights --
        W_SCORE = 10.0  # Weight for the current game score
        W_INV_FOOD_DIST = 10.0  # Weight for (1 / distance to food)
        W_CARRY_BONUS = 10  # Weight for each piece of carried food
        W_RETURN_HOME = 50.0  # Extra reward for moving closer to boundary if carrying
        W_GHOST_DANGER = 10.0  # Penalty multiplier for being near ghosts
        W_CAPSULE_INV_DIST = 10.0  # Weight for (1 / distance to capsule)

        #Extreme Conditions
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

        carrying = my_state.numCarrying

        if carrying < 3:
            carry_bonus = carrying * W_CARRY_BONUS
        else:
            carry_bonus = -(carrying * W_CARRY_BONUS)

        if self.home_bounds:
            closest_boundary_dist = min(self.distancer.getDistance(my_pos, b) for b in self.home_bounds)
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


    def get_enemies(self):
        ghosts= []
        scaredGhosts = []
        pacmans = []
        for enemy in self.getOpponents(self.sim_gameState):
            enemyState = self.sim_gameState.getAgentState(enemy)
            if not enemyState.isPacman and enemyState.scaredTimer == 0:
                enemyPos = self.sim_gameState.getAgentPosition(enemy)
                if enemyPos != None:
                    ghosts.append(enemy)
            if not enemyState.isPacman and enemyState.scaredTimer > 0:
                enemyPos = self.sim_gameState.getAgentPosition(enemy)
                if enemyPos != None:
                    scaredGhosts.append(enemy)
            if enemyState.isPacman:
                enemyPos = self.sim_gameState.getAgentPosition(enemy)
                if enemyPos != None:
                    pacmans.append(enemy)
        return pacmans, ghosts, scaredGhosts




class defendingAgent(Agents):

  def chooseAction(self, gameState):
        start_time = time.time()
        tree = MCTS(gameState)
        tree.agent_index = self.index
        tree.getRoot(gameState, self.index)

        iteration = 0
        while time.time() - start_time < self.simulation_time:
            iteration += 1
            tree.traverse()
            tree.expand()
            self.sim_gameState = tree.simulate()
            reward = self.evaluate()
            tree.backpropagate(reward)
            for node in tree.current_root.children:
                action = node.action
                tree.rave_values[action] = tree.rave_values.get(action, 0) + reward
                tree.rave_visits[action] = tree.rave_visits.get(action, 0) + 1
            print(iteration)

        actions = [node.action for node in tree.current_root.children]
        simulations = {node.action: node.visits for node in tree.current_root.children}
        scores = {node.action: node.reward for node in tree.current_root.children}

        best_action = max(actions, key=lambda a: tree.get_rave_value(a, simulations[a], scores[a]))
        
        return best_action

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

