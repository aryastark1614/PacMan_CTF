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
               first = 'AttackingAgent', second = 'DefendingAgent'):
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

class AttackingAgent(CaptureAgent):

  def registerInitialState(self, gameState):
    
    self.simulation_time = 0.1
    self.max_depth = 6
    
    CaptureAgent.registerInitialState(self, gameState)
    
    self.start = gameState.getAgentPosition(self.index)
    self.food_carrying = 0
    self.max_food_to_carry = 5


  def chooseAction(self, gameState):
    
    start_time = time.time()
    
    if not gameState.getAgentState(self.index).isPacman:
      self.food_carrying = 0
    
    actions = gameState.getLegalActions(self.index)
    
    simulations = {action: 0 for action in actions}
    scores = {action: 0 for action in actions}

    for action in actions:
        next_state = gameState.generateSuccessor(self.index, action)
        if len(self.getFood(gameState).asList()) > len(self.getFood(next_state).asList()):
            scores[action] += 5000
    
    while time.time() - start_time < self.simulation_time:
      action = random.choice(actions)
      next_state = gameState.generateSuccessor(self.index, action)
      
      sim_food_carrying = self.food_carrying
      if len(self.getFood(gameState).asList()) > len(self.getFood(next_state).asList()):
          sim_food_carrying += 1
      
      reward = self.simulate(next_state, depth=0, food_carrying=sim_food_carrying)
      simulations[action] += 1
      scores[action] += reward
    
    best_action = max(actions, key=lambda a: scores[a] / (simulations[a] + 1e-6))

    successor = gameState.generateSuccessor(self.index, best_action)
    if len(self.getFood(gameState).asList()) > len(self.getFood(successor).asList()):
        self.food_carrying += 1
    
    print(f"Best action: {best_action}, Score: {scores[best_action] / (simulations[best_action] + 1e-6)}")
    return best_action
  
  def simulate(self, gameState, depth, food_carrying):
        if depth >= self.max_depth or gameState.isOver():
            return self.evaluate(gameState, food_carrying) * (0.9 ** depth)

        actions = gameState.getLegalActions(self.index)
        
        if not actions:
            return self.evaluate(gameState, food_carrying)
        
        action_scores = []
        for action in actions:
            next_state = gameState.generateSuccessor(self.index, action)
            
            sim_food_carrying = food_carrying
            if len(self.getFood(gameState).asList()) > len(self.getFood(next_state).asList()):
                sim_food_carrying += 1
                
            score = self.evaluate_action(gameState, action, food_carrying, sim_food_carrying)
            action_scores.append((action, score, sim_food_carrying))
        
        total_score = sum(max(1, score) for _, score, _ in action_scores)
        probabilities = [max(1, score) / total_score for _, score, _ in action_scores]
        
        chosen_idx = random.choices(range(len(action_scores)), weights=probabilities)[0]
        action, _, new_food_carrying = action_scores[chosen_idx]
        
        next_state = gameState.generateSuccessor(self.index, action)
        return self.simulate(next_state, depth + 1, new_food_carrying)
      
  def evaluate_action(self, gameState, action, current_food_carrying, next_food_carrying):
      successor = gameState.generateSuccessor(self.index, action)
      
      if next_food_carrying > current_food_carrying:
          return 15000
      
      return self.evaluate(successor, next_food_carrying)
      
  def evaluate(self, gameState, food_carrying=None):

    if food_carrying is None:
        food_carrying = self.food_carrying
        
    position = gameState.getAgentPosition(self.index)
    food_list = self.getFood(gameState).asList()
    
    score = 0
    
    if food_carrying >= self.max_food_to_carry:
      dist_to_start = self.getMazeDistance(position, self.start)
      return 10000 - (10 * dist_to_start)
    
    score += 2000 * food_carrying
    
    if len(food_list) > 2:
      min_food_dist = min(self.getMazeDistance(position, food) for food in food_list)
      score -= 10 * min_food_dist
    
    else:
      score += 1000 - self.getMazeDistance(position, self.start)
      
    enemies = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]

    if enemies:
      ghost_dists = [self.getMazeDistance(position, enemy.getPosition()) for enemy in enemies]
      if min(ghost_dists) <= 2:
        score -= 1000 * food_carrying
      elif min(ghost_dists) <= 5:
        score -= 100 * food_carrying
        
        
    if gameState.getAgentState(self.index).isPacman:
      score += 100
            
    return score





class DefendingAgent(CaptureAgent):

  def registerInitialState(self, gameState):
      
    self.simulation_time = 0.1
    self.max_depth = 10
    
    CaptureAgent.registerInitialState(self, gameState)

  def chooseAction(self, gameState):
    
    start_time = time.time()
    actions = gameState.getLegalActions(self.index)
    simulations = {action: 0 for action in actions}
    scores = {action: 0 for action in actions}
    
    while start_time < self.simulation_time:
      action = random.choice(actions)
      next_state = gameState.generateSuccessor(self.index, action)
      
      reward = self.simulate(next_state, depth=0)
      simulations[action] += 1
      scores[action] += reward
    
    best_action = max(actions, key=lambda a: scores[a] / (simulations[a] + 1e-6))
    
    
    print(f"Best action: {best_action}, Score: {scores[best_action] / (simulations[best_action] + 1e-6)}")
    return best_action
  
  def simulate(self, gameState, depth):
    
    if depth >= self.max_depth or gameState.isOver():
        return self.evaluate(gameState) * (0.9 ** depth)

    actions = gameState.getLegalActions(self.index)
    
    if not actions:
        return self.evaluate(gameState)
    
    action_scores = []
    for action in actions:
        next_state = gameState.generateSuccessor(self.index, action)
            
        score = self.evaluate_action(gameState, action)
        action_scores.append((action, score))
    
    total_score = sum(max(1, score) for _, score, _ in action_scores)
    probabilities = [max(1, score) / total_score for _, score, _ in action_scores]
    
    chosen_idx = random.choices(range(len(action_scores)), weights=probabilities)[0]
    action, _, new_food_carrying = action_scores[chosen_idx]
    
    next_state = gameState.generateSuccessor(self.index, action)
    return self.simulate(next_state, depth + 1, new_food_carrying)
  

  def evaluate_action(self, gameState, action):
      successor = gameState.generateSuccessor(self.index, action)
      
      return self.evaluate(successor)
    
  def evaluate(self, gameState):

    position = gameState.getAgentPosition(self.index)
    
    score = 0   
    score += 1000 - self.getMazeDistance(position, self.start)
      
    enemies = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]

    if enemies:
      ghost_dists = [self.getMazeDistance(position, enemy.getPosition()) for enemy in enemies]
      if min(ghost_dists) <= 2:
        score += 1000
      elif min(ghost_dists) <= 5:
        score += 100
        
    #penalty for being on the opponents side
    if gameState.getAgentState(self.index).isPacman:
      score -= 100
            
    return score