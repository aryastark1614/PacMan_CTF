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

# Creates a team from the attacking agent and the defending agent
def createTeam(firstIndex, secondIndex, isRed, first = 'attackingAgent', second = 'attackingAgent'):
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
            self.current_node = self.current_node.parent


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
            print(iteration)

        best_node = max(tree.current_root.children, key=lambda node: node.reward)
        return best_node.action

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
      return