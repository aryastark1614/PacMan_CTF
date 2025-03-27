from captureAgents import CaptureAgent
import random, time, util, math
from game import Directions
import game

#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed,
               first='MCTSAgent', second='MCTSAgent'):
    """
    This function returns a list of two agents forming the team.
    """
    return [eval(first)(firstIndex), eval(second)(secondIndex)]

##########
# Agents #
##########

class MCTSNode:
    def __init__(self, state, parent=None, action=None):
        self.state = state
        self.parent = parent
        self.children = []
        self.visits = 0
        self.value = 0
        self.action = action

    def is_fully_expanded(self):
        return len(self.children) > 0 and len(self.children) == len(self.state.getLegalActions(0))
    
    def best_child(self, exploration_weight=1.0):
        return max(self.children, key=lambda child: child.value / (child.visits + 1e-6) + exploration_weight * math.sqrt(math.log(self.visits + 1) / (child.visits + 1e-6)))
    
    def expand(self):
        legal_actions = self.state.getLegalActions(0)
        for action in legal_actions:
            if not any(child.action == action for child in self.children):
                new_state = self.state.generateSuccessor(0, action)
                new_node = MCTSNode(new_state, parent=self, action=action)
                self.children.append(new_node)
                return new_node
        return random.choice(self.children)  # If fully expanded, return a random child
    
    def update(self, reward):
        self.visits += 1
        self.value += reward
        if self.parent:
            self.parent.update(reward)

class MCTSAgent(CaptureAgent):
    """
    A Capture the Flag agent that uses Monte Carlo Tree Search (MCTS) for decision making.
    """
    def registerInitialState(self, gameState):
        CaptureAgent.registerInitialState(self, gameState)
    
    def chooseAction(self, gameState):
        """
        Uses Monte Carlo Tree Search to choose the best action.
        """
        root = MCTSNode(gameState)
        time_limit = time.time() + 0.9  # Ensure computation fits within time constraints

        while time.time() < time_limit:
            node = self.tree_policy(root)
            reward = self.simulate(node.state)
            node.update(reward)

        return root.best_child(exploration_weight=0.0).action  # Choose best move based on value
    
    def tree_policy(self, node):
        """
        Expands the tree by selecting unexplored nodes or best UCT nodes.
        """
        while not node.state.isOver():
            if not node.is_fully_expanded():
                return node.expand()
            node = node.best_child()
        return node
    
    def simulate(self, state, depth=10):
        """
        Simulates a game using a heuristic rollout policy up to a given depth.
        """
        for _ in range(depth):
            if state.isOver():
                break
            action = self.heuristic_rollout_policy(state)
            state = state.generateSuccessor(0, action)
        return self.evaluate(state)
    
    def heuristic_rollout_policy(self, state):
        """
        A simple heuristic for rollouts favoring food collection.
        """
        actions = state.getLegalActions(0)
        best_action = max(actions, key=lambda a: self.evaluate(state.generateSuccessor(0, a)))
        return best_action
    
    def evaluate(self, state):
        """
        Returns a heuristic value of the state.
        """
        return len(self.getFood(state).asList())  # Favor states with more food collected
