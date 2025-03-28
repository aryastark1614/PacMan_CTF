# myTeam.py
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
  """
  This function should return a list of two agents that will form the
  team, initialized using firstIndex and secondIndex as their agent
  index numbers.  isRed is True if the red team is being created, and
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

class AttackingDefenderAgentTop(CaptureAgent):

  def registerInitialState(self, gameState):
    
    self.simulation_time = 0.01
    CaptureAgent.registerInitialState(self, gameState)
    
    self.start = gameState.getAgentPosition(self.index)
    self.food_carrying = 0
    self.max_food_to_carry = 1

  def chooseAction(self, gameState):
    start_time = time.time()
    
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
    print(scores)
    best_action = max(actions, key=lambda a: scores[a])
    
    if best_action not in actions:
      return random.choice(actions)
    return best_action
  
  def evaluate(self, gameState, food_carrying=None):

    if food_carrying is None:
        food_carrying = self.food_carrying
        
    position = gameState.getAgentPosition(self.index)
    food_list = self.getFood(gameState).asList()
    
    score = 0
    
    
    if food_carrying >= self.max_food_to_carry:
      dist_to_start = self.getMazeDistance(position, self.start)
      return 100000-(10 * dist_to_start)

    opp = self.getOpponents(gameState)
    enemies = [gameState.getAgentPosition(i) for i in opp]
    enemy_pacmans = [en for en in opp if gameState.getAgentState(en).isPacman]
    

    for en in self.getOpponents(gameState):
      if self.red:
        if gameState.getAgentPosition(en)[0] == 30 and position[0] != 1:
          score += 1000
      else:
        if gameState.getAgentPosition(en)[0] == 1 and position[0] != 30:
          score += 1000
          
    for en in self.getOpponents(gameState):
      if self.red:
        if gameState.getAgentPosition(en)[0] != 30 and position[0] == 1:
          score -= 1000
      else:
        if gameState.getAgentPosition(en)[0] != 1 and position[0] == 30:
          score -= 1000
          
    if enemy_pacmans and gameState.getAgentState(self.index).scaredTimer == 0:
      closest_pacman = min(enemy_pacmans, key=lambda ep: self.getMazeDistance(position, gameState.getAgentPosition(ep)))
      distance_to_pacman = self.getMazeDistance(position, gameState.getAgentPosition(closest_pacman))
      score += 150 -  3 * distance_to_pacman

    else:
      closest_enemy = min(self.getMazeDistance(position, enemy) for enemy in enemies)
      score += 50 - closest_enemy
    
    if gameState.getAgentState(self.index).isPacman:
      score -= 10
      
    capsules = self.getCapsules(gameState)
    if capsules:
        capsule = capsules[0]
        closest_enemy_to_capsule = min(self.getMazeDistance(capsule, enemy) for enemy in enemies)
        distance_to_capsule = self.getMazeDistance(position, capsule)
        if distance_to_capsule < closest_enemy_to_capsule:
            print("CLOSE TO CAPSULE")
            return 10000 - ( 10 * distance_to_capsule)
        
        
    closest_food = min(food_list, key=lambda food: self.getMazeDistance(position, food))
    closest_enemy_to_closest_food = min(self.getMazeDistance(closest_food, enemy) for enemy in enemies)
    distance_to_closest_food = self.getMazeDistance(position, closest_food)
    if 2 * distance_to_closest_food < closest_enemy_to_closest_food or gameState.getAgentState(((self.index + 1) % 4)).scaredTimer > 0:
      print("CLOSE TO FOOD")
      return 10000 - (10 * distance_to_closest_food)
    
    
            
    #Check if teammate is one away from enemy, if so, go near the other enemy
    teammate_indices = [i for i in self.getTeam(gameState) if i != self.index]
    teammate_positions = [gameState.getAgentPosition(i) for i in teammate_indices]
    enemies_positions = [gameState.getAgentPosition(i) for i in self.getOpponents(gameState) if gameState.getAgentPosition(i) is not None]

    for teammate_position in teammate_positions:
        for enemy_position in enemies_positions:
            if self.getMazeDistance(teammate_position, enemy_position) == 1:
                #Try to have one agent go to the other side of the board
                get_teammate = self.getTeam(gameState)
                teammate = get_teammate[0] if get_teammate[0] != self.index else get_teammate[1]
                teammate_position = gameState.getAgentPosition(teammate)
                score += 0.5 * self.getMazeDistance(position, teammate_position)
    
    
    

    return score



class AttackingDefenderAgentBottom(CaptureAgent):

  def registerInitialState(self, gameState):
    
    self.simulation_time = 0.01
    CaptureAgent.registerInitialState(self, gameState)
    
    self.start = gameState.getAgentPosition(self.index)
    self.food_carrying = 0
    self.max_food_to_carry = 1

  def chooseAction(self, gameState):
    start_time = time.time()
    
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
    print(scores)
    best_action = max(actions, key=lambda a: scores[a])
    
    if best_action not in actions:
      return random.choice(actions)
    return best_action
  
  def evaluate(self, gameState, food_carrying=None):

    if food_carrying is None:
        food_carrying = self.food_carrying
        
    position = gameState.getAgentPosition(self.index)
    food_list = self.getFood(gameState).asList()
    
    score = 0
    
    
    if food_carrying >= self.max_food_to_carry:
      dist_to_start = self.getMazeDistance(position, self.start)
      return 1000 - (dist_to_start)

    opp = self.getOpponents(gameState)
    enemies = [gameState.getAgentPosition(i) for i in opp]
    enemy_pacmans = [en for en in opp if gameState.getAgentState(en).isPacman]

    for en in self.getOpponents(gameState):
      if self.red:
        if gameState.getAgentPosition(en)[0] == 30 and position[0] != 1:
          score += 1000
      else:
        if gameState.getAgentPosition(en)[0] == 1 and position[0] != 30:
          score += 1000
          
    for en in self.getOpponents(gameState):
      if self.red:
        if gameState.getAgentPosition(en)[0] != 30 and position[0] == 1:
          score -= 1000
      else:
        if gameState.getAgentPosition(en)[0] != 1 and position[0] == 30:
          score -= 1000
    print(f"ScaredTimer self: {gameState.getAgentState(self.index).scaredTimer}")
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
            print("CLOSE TO CAPSULE")
            return 10000 - ( 10 * distance_to_capsule)
        
        
    closest_food = min(food_list, key=lambda food: self.getMazeDistance(position, food))
    closest_enemy_to_closest_food = min(self.getMazeDistance(closest_food, enemy) for enemy in enemies)
    distance_to_closest_food = self.getMazeDistance(position, closest_food)
    if 2 * distance_to_closest_food < closest_enemy_to_closest_food or gameState.getAgentState(((self.index + 1) % 4)).scaredTimer > 0:
      print("CLOSE TO FOOD")
      return 10000 - (10 * distance_to_closest_food)
            
            
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
                score += 2 * self.getMazeDistance(position, teammate_position)

    return score