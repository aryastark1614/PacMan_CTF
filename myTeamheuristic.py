from captureAgents import CaptureAgent
import random, time, util, math
from game import Directions

class HeuristicAgent(CaptureAgent):
    def registerInitialState(self, gameState):
        CaptureAgent.registerInitialState(self, gameState)
        self.last_positions = []  # Track previous positions to avoid oscillation
    
    def chooseAction(self, gameState):
        """ Chooses the best action based on heuristic evaluation """
        actions = gameState.getLegalActions(self.index)
        best_action = max(actions, key=lambda a: self.evaluate(gameState, a))
        
        self.last_positions.append(gameState.getAgentPosition(self.index))
        if len(self.last_positions) > 5:
            self.last_positions.pop(0)  # Keep track of last 5 positions to prevent loops
        
        return best_action
    
    def evaluate(self, gameState, action):
        """ State evaluation function for heuristic decision making """
        successor = gameState.generateSuccessor(self.index, action)
        position = successor.getAgentPosition(self.index)
        food_list = self.getFood(successor).asList()
        capsules = self.getCapsules(successor)
        enemies = [successor.getAgentPosition(i) for i in self.getOpponents(successor) if successor.getAgentPosition(i) is not None]
        
        score = successor.getScore()
        
        # Encourage eating food
        if food_list:
            closest_food = min(food_list, key=lambda f: util.manhattanDistance(position, f))
            score += 10 / (1 + util.manhattanDistance(position, closest_food))
        
        # Encourage eating capsules
        if capsules:
            closest_capsule = min(capsules, key=lambda c: util.manhattanDistance(position, c))
            score += 20 / (1 + util.manhattanDistance(position, closest_capsule))
        
        # Avoid enemies
        for enemy in enemies:
            if util.manhattanDistance(position, enemy) < 3:
                score -= 50  # Strong penalty for being near an enemy
        
        # Avoid looping
        if position in self.last_positions:
            score -= 5  # Small penalty for repeating positions
        
        return score
