# teamHeuristic.py
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


import random
import time

import game
import util
from captureAgents import CaptureAgent
from game import Directions

#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed,
               first = 'AttackingDefenderAgentTop', second = 'AttackingDefenderAgentBottom'):


  # The following line is an example only; feel free to change it.
  return [eval(first)(firstIndex), eval(second)(secondIndex)]

##########
# Agents #
##########

class AttackingDefenderAgentTop(CaptureAgent):

  def registerInitialState(self, gameState):
    
    CaptureAgent.registerInitialState(self, gameState)
    
    self.start = gameState.getAgentPosition(self.index)
    self.food_carrying = 0
    self.max_food_to_carry = 1

  def chooseAction(self, gameState):
    
    if not gameState.getAgentState(self.index).isPacman:
      self.food_carrying = 0
    
    actions = gameState.getLegalActions(self.index)

    scores = {action: 0 for action in actions}


    
    for action in actions:
        
      next_state = gameState.generateSuccessor(self.index, action)
      sim_food_carrying = self.food_carrying
      if len(self.getFood(gameState).asList()) > len(self.getFood(next_state).asList()):
          sim_food_carrying += 1
          self.food_carrying += 1
          scores[action] += 10000
          
      if len(self.getCapsules(gameState)) > len(self.getCapsules(next_state)):
          scores[action] += 100000
      
      reward = self.evaluate(next_state, food_carrying=sim_food_carrying)
      scores[action] += reward
    best_action = max(actions, key=lambda a: scores[a])
    
    if best_action not in actions:
      return random.choice(actions)
    return best_action
  
  def evaluate(self, gameState, food_carrying=None):

    width = gameState.data.layout.width
    height = gameState.data.layout.height
    walls = gameState.getWalls().asList()
    if self.red:

      boundary_positions = [(float(x), float(y)) for x in range(width // 2) for y in range(1, height) if (x, y) not in walls]

    else:

      boundary_positions = [(float(x), float(y)) for x in range(width // 2 + 1) for y in range(1, height) if
                            (x, y) not in walls]

    if food_carrying is None:
        food_carrying = self.food_carrying
        
    position = gameState.getAgentPosition(self.index)
    food_list = self.getFood(gameState).asList()
    
    score = 0
    
    
    if food_carrying >= self.max_food_to_carry:
      closest_boundary_dist = min(self.distancer.getDistance(position, b) for b in boundary_positions)
      return 100000-(10 * closest_boundary_dist)

    opp = self.getOpponents(gameState)
    enemies = [gameState.getAgentPosition(i) for i in opp]
    enemy_pacmans = [en for en in opp if gameState.getAgentState(en).isPacman]
    
    map_width = gameState.data.layout.width

    for en in self.getOpponents(gameState):
      if self.red:
        if gameState.getAgentPosition(en)[0] == map_width - 2 and position[0] != 1:
          score += 1000
      else:
        if gameState.getAgentPosition(en)[0] == 1 and position[0] != map_width - 2:
          score += 1000
          
    for en in self.getOpponents(gameState):
      if self.red:
        if gameState.getAgentPosition(en)[0] != map_width - 2 and position[0] == 1:
          score -= 1000
      else:
        if gameState.getAgentPosition(en)[0] != 1 and position[0] == map_width - 2:
          score -= 1000
          
    if enemy_pacmans and gameState.getAgentState(self.index).scaredTimer == 0:
      closest_pacman = min(enemy_pacmans, key=lambda ep: self.getMazeDistance(position, gameState.getAgentPosition(ep)))
      distance_to_pacman = self.getMazeDistance(position, gameState.getAgentPosition(closest_pacman))
      score += 150 -  3 * distance_to_pacman

    else:
      closest_enemy = min(self.getMazeDistance(position, enemy) for enemy in enemies)
      score += 100 - closest_enemy
    
    if gameState.getAgentState(self.index).isPacman:
      score -= 10
      
    capsules = self.getCapsules(gameState)
    if capsules:
        capsule = capsules[0]
        closest_enemy_to_capsule = min(self.getMazeDistance(capsule, enemy) for enemy in enemies)
        distance_to_capsule = self.getMazeDistance(position, capsule)
        if distance_to_capsule < closest_enemy_to_capsule:
            return 10000 - ( 10 * distance_to_capsule)
        
        
    closest_food = min(food_list, key=lambda food: self.getMazeDistance(position, food))
    closest_enemy_to_closest_food = min(self.getMazeDistance(closest_food, enemy) for enemy in enemies)
    distance_to_closest_food = self.getMazeDistance(position, closest_food)
    if 2 * distance_to_closest_food < closest_enemy_to_closest_food or gameState.getAgentState(((self.index + 1) % 4)).scaredTimer > 0:
      return 5000 - (10 * distance_to_closest_food)
    
    
            
    #Check if teammate is one away from enemy, if so, go near the other enemy
    teammate_indices = [i for i in self.getTeam(gameState) if i != self.index]
    teammate_positions = [gameState.getAgentPosition(i) for i in teammate_indices]
    enemies_positions = [gameState.getAgentPosition(i) for i in self.getOpponents(gameState) if gameState.getAgentPosition(i) is not None]

    for teammate_position in teammate_positions:
        for enemy_position in enemies_positions:
            if self.getMazeDistance(teammate_position, enemy_position) == 1:
                return 10000 - (10 * distance_to_closest_food)
    
    
    

    return score



class AttackingDefenderAgentBottom(CaptureAgent):

  def registerInitialState(self, gameState):
    
    CaptureAgent.registerInitialState(self, gameState)
    
    self.start = gameState.getAgentPosition(self.index)
    self.food_carrying = 0
    self.max_food_to_carry = 1

  def chooseAction(self, gameState):
    start_time = time.time()
    
    if not gameState.getAgentState(self.index).isPacman:
      self.food_carrying = 0
    
    actions = gameState.getLegalActions(self.index)
    actions.remove(Directions.STOP)
    scores = {action: 0 for action in actions}


    
    for action in actions:
        
      next_state = gameState.generateSuccessor(self.index, action)
      sim_food_carrying = self.food_carrying
      if len(self.getFood(gameState).asList()) > len(self.getFood(next_state).asList()):
          sim_food_carrying += 1
          self.food_carrying += 1
          scores[action] += 10000
      if len(self.getCapsules(gameState)) > len(self.getCapsules(next_state)):
          scores[action] += 100000
          
      reward = self.evaluate(next_state, food_carrying=sim_food_carrying)
      scores[action] += reward
    best_action = max(actions, key=lambda a: scores[a])
    
    if best_action not in actions:
      return random.choice(actions)
    return best_action
  
  def evaluate(self, gameState, food_carrying=None):

    width = gameState.data.layout.width
    height = gameState.data.layout.height
    walls = gameState.getWalls().asList()
    if self.red:

      boundary_positions = [(float(x), float(y)) for x in range(width // 2) for y in range(1, height) if (x, y) not in walls]

    else:

      boundary_positions = [(float(x), float(y)) for x in range(width // 2 + 1) for y in range(1, height) if
                            (x, y) not in walls]

    if food_carrying is None:
        food_carrying = self.food_carrying
        
    position = gameState.getAgentPosition(self.index)
    food_list = self.getFood(gameState).asList()
    
    score = 0
    
    
    if food_carrying >= self.max_food_to_carry:
      closest_boundary_dist = min(self.distancer.getDistance(position, b) for b in boundary_positions)
      return 100000-(10 * closest_boundary_dist)

    opp = self.getOpponents(gameState)
    enemies = [gameState.getAgentPosition(i) for i in opp]
    enemy_pacmans = [en for en in opp if gameState.getAgentState(en).isPacman]
    map_width = gameState.data.layout.width
    
    for en in self.getOpponents(gameState):
      if self.red:
        if gameState.getAgentPosition(en)[0] == map_width - 2 and position[0] != 1:
          score += 1000
      else:
        if gameState.getAgentPosition(en)[0] == 1 and position[0] != map_width - 2:
          score += 1000
          
    for en in self.getOpponents(gameState):
      if self.red:
        if gameState.getAgentPosition(en)[0] != map_width - 2 and position[0] == 1:
          score -= 1000
      else:
        if gameState.getAgentPosition(en)[0] != 1 and position[0] == map_width - 2:
          score -= 1000
          
    if enemy_pacmans and gameState.getAgentState(self.index).scaredTimer == 0:
      closest_pacman = min(enemy_pacmans, key=lambda ep: self.getMazeDistance(position, gameState.getAgentPosition(ep)))
      distance_to_pacman = self.getMazeDistance(position, gameState.getAgentPosition(closest_pacman))
      score += 150 -  3 * distance_to_pacman

    else:
      closest_enemy = min(self.getMazeDistance(position, enemy) for enemy in enemies)
      score += 100 - closest_enemy
    
    if gameState.getAgentState(self.index).isPacman:
      score -= 10
      
    capsules = self.getCapsules(gameState)
    if capsules:
        capsule = capsules[0]
        closest_enemy_to_capsule = min(self.getMazeDistance(capsule, enemy) for enemy in enemies)
        distance_to_capsule = self.getMazeDistance(position, capsule)
        if distance_to_capsule < closest_enemy_to_capsule:
            return 10000 - ( 10 * distance_to_capsule)
        
        
    closest_food = min(food_list, key=lambda food: self.getMazeDistance(position, food))
    closest_enemy_to_closest_food = min(self.getMazeDistance(closest_food, enemy) for enemy in enemies)
    distance_to_closest_food = self.getMazeDistance(position, closest_food)
    if 2 * distance_to_closest_food < closest_enemy_to_closest_food or gameState.getAgentState(((self.index + 1) % 4)).scaredTimer > 0:
      return 5000 - (10 * distance_to_closest_food)
            
            
    #Check if teammate is one away from enemy, if so, go away from teammate
    teammate_indices = [i for i in self.getTeam(gameState) if i != self.index]
    teammate_positions = [gameState.getAgentPosition(i) for i in teammate_indices]
    enemies_positions = [gameState.getAgentPosition(i) for i in self.getOpponents(gameState) if gameState.getAgentPosition(i) is not None]
    for teammate_position in teammate_positions:
        for enemy_position in enemies_positions:
            if self.getMazeDistance(teammate_position, enemy_position) == 1:
                get_teammate = self.getTeam(gameState)
                teammate = get_teammate[0] if get_teammate[0] != self.index else get_teammate[1]
                teammate_position = gameState.getAgentPosition(teammate)
                score += 0.4 * self.getMazeDistance(position, teammate_position)

    return score