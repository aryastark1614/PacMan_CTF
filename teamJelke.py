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
               first = 'AttackingDefenderAgentTop', second = 'AttackingDefenderAgentTop'):
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
    
    self.simulation_time = 0.21
    self.max_depth = 3
    self.exploration = 0   
    CaptureAgent.registerInitialState(self, gameState)
    
    self.start = gameState.getAgentPosition(self.index)
    self.food_carrying = 0
    self.max_food_to_carry = 1

    self.action_values = {}
    self.action_visits = {}
    self.rave_values = {}
    self.rave_visits = {}

  def chooseAction(self, gameState):
    start_time = time.time()
    
    if not gameState.getAgentState(self.index).isPacman:
      self.food_carrying = 0
    
    actions = gameState.getLegalActions(self.index)
    print(actions)
    simulations = {action: 0 for action in actions}
    scores = {action: 0 for action in actions}

    # for action in actions:
    #     next_state = gameState.generateSuccessor(self.index, action)
    #     if len(self.getFood(gameState).asList()) > len(self.getFood(next_state).asList()):
    #         scores[action] += 1000
    
    while time.time() - start_time < self.simulation_time:
      action = random.choice(actions)

      next_state = gameState.generateSuccessor(self.index, action)
      sim_food_carrying = self.food_carrying
      if len(self.getFood(gameState).asList()) > len(self.getFood(next_state).asList()):
          sim_food_carrying += 1
      
      reward = self.simulate(next_state, depth=0, food_carrying=sim_food_carrying)
      simulations[action] += 1
      scores[action] += reward
      self.updateRAVE(action, reward)

    print("\nAction Analysis:")
    for action in actions:
        direct_score = scores[action] / (simulations[action] + 1e-6)
        rave_value = self.rave_values.get(action, 0) / (self.rave_visits.get(action, 1))
        combined_value = self.getCombinedValue(action, simulations[action], scores[action])
        
        print(f"Action: {action}")
        print(f"  Direct Score: {direct_score:.2f}")
        print(f"  Simulations: {simulations[action]}")
        print(f"  RAVE Value: {rave_value:.2f}")
        print(f"  Combined Value: {combined_value:.2f}")
    best_action = max(actions, key=lambda a: self.getCombinedValue(a, simulations[a], scores[a]))
    
    successor = gameState.generateSuccessor(self.index, best_action)
    if len(self.getFood(gameState).asList()) > len(self.getFood(successor).asList()):
        self.food_carrying += 1
    
    print(f"Best action: {best_action}, Score: {scores[best_action] / (simulations[best_action] + 1e-6)}")
    if best_action not in actions:
      return random.choice(actions)
    return best_action
  
  def simulate(self, gameState, depth, food_carrying):
        if depth >= self.max_depth or gameState.isOver():
            return self.evaluate(gameState, food_carrying)

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
        
        chosen_idx = max(range(len(action_scores)), key=lambda i: action_scores[i][1])

        action, _, new_food_carrying = action_scores[chosen_idx]
        
        next_state = gameState.generateSuccessor(self.index, action)
    
        all_indexes = [0, 1, 2, 3]
        if self.red:
          all_indexes = [1, 3]
        else:
          all_indexes = [1, 3]
        #all_indexes.remove(self.index)
        
        
        for agent in all_indexes:
            if next_state.isOver():
                break
                    
            next_state = next_state.generateSuccessor(agent, random.choice(gameState.getLegalActions(agent)))
                
        return self.simulate(next_state, depth + 1, new_food_carrying)
      
  def getCombinedValue(self, action, num_simulations, score):
        rave_value = self.rave_values.get(action, 0)
        rave_visits = self.rave_visits.get(action, 0)

        combined_value = (1 - self.exploration) * (score / (num_simulations + 1e-6)) + \
                         self.exploration * (rave_value / (rave_visits + 1e-6))
        return combined_value
    
  def updateRAVE(self, action, reward):
        if action not in self.rave_values:
            self.rave_values[action] = 0
            self.rave_visits[action] = 0

        self.rave_values[action] += reward
        self.rave_visits[action] += 1
        
  def evaluate_action(self, gameState, action, current_food_carrying, next_food_carrying):
      successor = gameState.generateSuccessor(self.index, action)
      
      # if next_food_carrying > current_food_carrying:
      #     return 1000
      
      return self.evaluate(successor, next_food_carrying)
      
  def evaluate(self, gameState, food_carrying=None):

    if food_carrying is None:
        food_carrying = self.food_carrying
        
    position = gameState.getAgentPosition(self.index)
    food_list = self.getFood(gameState).asList()
    
    score = 0
    
    
    if food_carrying >= self.max_food_to_carry:
      dist_to_start = self.getMazeDistance(position, self.start)
      return -(10 * dist_to_start)

    opp = self.getOpponents(gameState)
    enemies = [gameState.getAgentPosition(i) for i in opp]
    enemy_pacmans = [en for en in opp if gameState.getAgentState(en).isPacman]

    # for en in self.getOpponents(gameState):
    #   if self.red:
    #     if gameState.getAgentPosition(en)[0] == 30 and position[0] != 1:
    #       score += 100
    #   else:
    #     if gameState.getAgentPosition(en)[0] == 1 and position[0] != 30:
    #       score += 100
          
    if enemy_pacmans:
      closest_pacman = min(enemy_pacmans, key=lambda ep: self.getMazeDistance(position, gameState.getAgentPosition(ep)))
      distance_to_pacman = self.getMazeDistance(position, gameState.getAgentPosition(closest_pacman))
      score += 150 -  3 * distance_to_pacman

    else:
      closest_enemy = min(self.getMazeDistance(position, enemy) for enemy in enemies)
      score += 50 - closest_enemy
    
    if gameState.getAgentState(self.index).isPacman:
      score -= 10
      
    closest_food = min(food_list, key=lambda food: self.getMazeDistance(position, food))
    closest_enemy_to_closest_food = min(self.getMazeDistance(closest_food, enemy) for enemy in enemies)
    distance_to_closest_food = self.getMazeDistance(position, closest_food)
    if 2 * distance_to_closest_food < closest_enemy_to_closest_food:
      print("CLOSE TO FOOD")
      score -= 20 * distance_to_closest_food
    
    # capsules = self.getCapsules(gameState)
    # if capsules:
    #     capsule = capsules[0]
    #     closest_enemy_to_capsule = min(self.getMazeDistance(capsule, enemy) for enemy in enemies)
    #     distance_to_capsule = self.getMazeDistance(position, capsule)
    #     if distance_to_capsule < closest_enemy_to_capsule:
    #         score -= 30 * distance_to_capsule
    
    return score
