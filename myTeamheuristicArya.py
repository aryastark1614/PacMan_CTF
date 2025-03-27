from captureAgents import CaptureAgent
import random, util
from game import Directions

def createTeam(firstIndex, secondIndex, isRed,
               first = 'HunterAgent', second = 'DefenderAgent'):
    return [eval(first)(firstIndex), eval(second)(secondIndex)]

class BaseAgent(CaptureAgent):
    def registerInitialState(self, gameState):
        CaptureAgent.registerInitialState(self, gameState)
        self.start = gameState.getAgentPosition(self.index)

class HunterAgent(BaseAgent):
    def chooseAction(self, gameState):
        # Analyze enemy positions
        enemies = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]
        visible_enemies = [e for e in enemies if e.isPacman and e.getPosition() is not None]
        
        # High priority: hunt enemy Pacmen
        if visible_enemies:
            return self.hunt_enemies(gameState, visible_enemies)
        
        # Collect food if no enemies visible
        return self.collect_food(gameState)
    
    def hunt_enemies(self, gameState, enemies):
        """Aggressive hunting strategy for enemy Pacmen"""
        my_pos = gameState.getAgentPosition(self.index)
        
        # Find closest enemy Pacman
        target_enemy = min(enemies, key=lambda e: self.getMazeDistance(my_pos, e.getPosition()))
        enemy_pos = target_enemy.getPosition()
        
        # Get all legal actions
        actions = gameState.getLegalActions(self.index)
        
        # Prioritize actions that move towards enemy
        hunt_actions = [
            action for action in actions 
            if self.get_next_position(my_pos, action) != my_pos and
            self.getMazeDistance(self.get_next_position(my_pos, action), enemy_pos) < 
            self.getMazeDistance(my_pos, enemy_pos)
        ]
        
        return random.choice(hunt_actions) if hunt_actions else random.choice(actions)
    
    def collect_food(self, gameState):
        """Systematic food collection strategy"""
        my_pos = gameState.getAgentPosition(self.index)
        food_list = self.getFood(gameState).asList()
        
        # If no food, return to start
        if not food_list:
            return self.go_home(gameState)
        
        # Find closest food
        target_food = min(food_list, key=lambda f: self.getMazeDistance(my_pos, f))
        
        # Plan move towards food
        actions = gameState.getLegalActions(self.index)
        food_actions = [
            action for action in actions 
            if self.get_next_position(my_pos, action) != my_pos and
            self.getMazeDistance(self.get_next_position(my_pos, action), target_food) < 
            self.getMazeDistance(my_pos, target_food)
        ]
        
        return random.choice(food_actions) if food_actions else random.choice(actions)
    
    def go_home(self, gameState):
        """Return to starting position if lost"""
        my_pos = gameState.getAgentPosition(self.index)
        actions = gameState.getLegalActions(self.index)
        
        home_actions = [
            action for action in actions 
            if self.get_next_position(my_pos, action) != my_pos and
            self.getMazeDistance(self.get_next_position(my_pos, action), self.start) < 
            self.getMazeDistance(my_pos, self.start)
        ]
        
        return random.choice(home_actions) if home_actions else random.choice(actions)
    
    def get_next_position(self, position, action):
        """Compute next position after an action"""
        x, y = position
        if action == Directions.NORTH:
            return (x, y+1)
        elif action == Directions.SOUTH:
            return (x, y-1)
        elif action == Directions.EAST:
            return (x+1, y)
        elif action == Directions.WEST:
            return (x-1, y)
        return position

class DefenderAgent(BaseAgent):
    def chooseAction(self, gameState):
        # Analyze enemy Pacmen
        enemies = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]
        invading_enemies = [e for e in enemies if e.isPacman and e.getPosition() is not None]
        
        # Defend home territory
        if invading_enemies:
            return self.intercept_enemies(gameState, invading_enemies)
        
        # Patrol near home base
        return self.patrol_home(gameState)
    
    def intercept_enemies(self, gameState, enemies):
        """Intercept strategy for enemy Pacmen in our territory"""
        my_pos = gameState.getAgentPosition(self.index)
        
        # Find closest invading enemy
        target_enemy = min(enemies, key=lambda e: self.getMazeDistance(my_pos, e.getPosition()))
        enemy_pos = target_enemy.getPosition()
        
        actions = gameState.getLegalActions(self.index)
        
        # Prioritize actions that move towards enemy
        intercept_actions = [
            action for action in actions 
            if self.get_next_position(my_pos, action) != my_pos and
            self.getMazeDistance(self.get_next_position(my_pos, action), enemy_pos) < 
            self.getMazeDistance(my_pos, enemy_pos)
        ]
        
        return random.choice(intercept_actions) if intercept_actions else random.choice(actions)
    
    def patrol_home(self, gameState):
        """Patrol near home base to prevent enemy intrusion"""
        my_pos = gameState.getAgentPosition(self.index)
        actions = gameState.getLegalActions(self.index)
        
        # Prefer actions that keep agent close to home
        home_patrol_actions = [
            action for action in actions 
            if self.getMazeDistance(self.get_next_position(my_pos, action), self.start) < 10
        ]
        
        return random.choice(home_patrol_actions) if home_patrol_actions else random.choice(actions)
    
    def get_next_position(self, position, action):
        """Compute next position after an action"""
        x, y = position
        if action == Directions.NORTH:
            return (x, y+1)
        elif action == Directions.SOUTH:
            return (x, y-1)
        elif action == Directions.EAST:
            return (x+1, y)
        elif action == Directions.WEST:
            return (x-1, y)
        return position







