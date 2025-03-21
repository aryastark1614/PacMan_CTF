import math
import random
import time
from capture import AgentRules
from distanceCalculator import Distancer


class MCTSNode:
    def __init__(self, gameState, parent=None, move=None):
        """
        MCTS Node
        :param gameState: The gameState at this node.
        :param parent: Parent node in the tree.
        :param move: The move that was applied to get here (None for the root).
        """
        self.gameState = gameState
        self.parent = parent
        self.move = move
        self.children = []  # List of child nodes.
        self.visits = 0  # Number of times this node has been visited.
        self.reward = 0.0  # Cumulative reward collected from simulations passing through this node.
        self.untriedMoves = []  # Moves that have not yet been tried from this node.

    def uct_value(self, exploration_constant=1.41):
        if self.visits == 0:
            return float('inf')
        if self.parent.visits == 0:
            return 0
        return (self.reward / self.visits) + exploration_constant * math.sqrt(
            math.log(self.parent.visits) / self.visits)


class MCTS:
    def __init__(self, gameState, agent_index, max_depth=50, exploration_constant=1.41, time_limit=15, start_time=time.time()):
        """
        Initialize the MCTS agent.
        :param root_state: The current gameState (should support deepCopy() and generateSuccessor()).
        :param agent_index: The index of the agent for which we are planning.
        :param max_depth: Maximum depth (number of moves) in a simulation rollout.
        :param exploration_constant: Constant used in UCT calculation.
        """
        self.root = MCTSNode(gameState)
        self.gameState = gameState
        self.pacman = agent_index == 0
        self.ghost = agent_index == 1
        self.max_depth = max_depth
        self.exploration_constant = exploration_constant
        self.time_limit = time_limit
        self.start_time = start_time

    def select(self, node):
        # If the node is terminal or has untried moves, return it
        if self.is_terminal(node.gameState) or node.untriedMoves:
            return node
        # Otherwise, recursively select the child with the highest UCT value
        best_child = max(node.children, key=lambda child: child.uct_value(self.exploration_constant))
        return self.select(best_child)

    def expand(self, node):
        move = random.choice(node.untriedMoves)
        node.untriedMoves.remove(move)
        next_state = node.gameState.generateSuccessor(self.agent_index, move)
        child_node = MCTSNode(next_state, parent=node, move=move)
        child_node.untriedMoves = AgentRules.getLegalActions(child_node.gameState, self.agent_index)
        node.children.append(child_node)
        return child_node

    def backpropagate(self, node, reward):
        while node is not None:
            node.visits += 1
            node.reward += reward
            node = node.parent

    def search(self):
        """
        Perform MCTS search and return the best move for our agent.
        Implements full multi-level tree search with recursive selection, expansion, simulation, and backpropagation.
        Uses a timer-based loop instead of a fixed number of iterations.
        """
        # Get legal moves for our agent from the root gameState.
        legal_moves = AgentRules.getLegalActions(self.root.gameState, self.agent_index)
        if not legal_moves:
            return None

        # Initialize untried moves at the root
        self.root.untriedMoves = legal_moves[:]

        # Optionally, initialize the root's children (one per legal move) if desired
        for move in legal_moves:
            next_state = self.root.gameState.generateSuccessor(self.agent_index, move)
            child_node = MCTSNode(next_state, parent=self.root, move=move)
            child_node.untriedMoves = AgentRules.getLegalActions(child_node.gameState, self.agent_index)
            self.root.children.append(child_node)

        while time.time() - self.start_time < self.time_limit:
            # Selection: recursively select a node
            node = self.select(self.root)

            # Expansion: if the node is not terminal and has untried moves, expand it
            if node.untriedMoves:
                node = self.expand(node)

            # Simulation: run a rollout from the selected (or expanded) node
            reward = self.simulate(node.gameState, self.max_depth)

            # Backpropagation: update the node and its ancestors with the simulation result
            self.backpropagate(node, reward)

        # After the time limit, choose the move corresponding to the child of the root with the highest average reward
        best_child = max(self.root.children, key=lambda child: (child.reward / child.visits) if child.visits > 0 else float('-inf'))
        return best_child.move

    def simulate_rollout(self, gameState, depth):
        """
        Simulate a rollout from the given gameState using random moves until a terminal state is reached or the depth limit is hit.
        :param gameState: The starting gameState for the simulation.
        :param depth: Maximum number of moves to simulate.
        :return: The evaluated reward of the resulting gameState.
        """

        current_state = gameState.deepCopy()
        current_depth = 0

        num_agents = len(current_state.data.agentStates)

        while current_depth < depth and not self.is_terminal(current_state):
            # HINT: The agent turn is determined in a round-robin fashion.
            current_agent = current_depth % num_agents
            legal_moves = AgentRules.getLegalActions(current_state, current_agent)
            if not legal_moves:
                break
            move = random.choice(legal_moves)
            current_state = current_state.generateSuccessor(current_agent, move)
            current_depth += 1

        # Return the evaluation of the final state reached in the simulation.
        return self.evaluate_state(current_state)

    def is_terminal(self, gameState):
        if hasattr(gameState.data, '_win') and gameState.data._win:
            return True
        if hasattr(gameState.data, '_lose') and gameState.data._lose:
            return True
        return False

    def evaluate_state(self, gameState):
        """
        Evaluate the given gameState based on multiple factors:
          - Collected food (using the gameState's score)
          - Distance to nearest food
          - Distance to nearest enemy defender
          - Distance to nearest enemy attacker
          - Distance to home side
        """
        distcalc = Distancer(gameState.data.layout)
        score = gameState.data.score  # Base score reflecting collected food.

        # Get our agent's state and position.
        my_state = gameState.getAgentState(self.agent_index)
        my_pos = gameState.getAgentPosition(self.agent_index)

        enemy_defenders = []
        enemy_attackers = []
        num_agents = len(gameState.data.agentStates)
        for i in range(num_agents):
            if i == self.agent_index:
                continue
            if gameState.isOnRedTeam(self.agent_index) != gameState.isOnRedTeam(i):
                opp_state = gameState.getAgentState(i)
                opp_pos = gameState.getAgentPosition(i)
                if opp_state.isPacman:
                    enemy_attackers.append(opp_pos)
                else:
                    enemy_defenders.append(opp_pos)

        if self.pacman == True:
            food_list = gameState.data.food.asList()
            if food_list:
                nearest_food_dist = min(distcalc.getDistance(my_pos, food) for food in food_list if food is not None)
            else:
                nearest_food_dist = 0
            score -= 10 * nearest_food_dist

    if self.agent_index == 1:

        # Distance to the nearest enemy defender.
        if enemy_defenders:
            nearest_defender_dist = min(distcalc.getDistance(my_pos, pos) for pos in enemy_defenders)
        else:
            nearest_defender_dist = 0
        score += 5 * nearest_defender_dist  # Reward being far from enemy defenders (if you're an attacker).

        # Distance to the nearest enemy attacker.
        if enemy_attackers:
            nearest_attacker_dist = min(distcalc.getDistance(my_pos, pos) for pos in enemy_attackers)
        else:
            nearest_attacker_dist = 0
        score -= 5 * nearest_attacker_dist  # Penalize being close to enemy attackers (if you're a defender).

        # Distance to home side.
        layout = gameState.data.layout
        if gameState.isOnRedTeam(self.agent_index):
            home_pos = (layout.width // 4, layout.height // 2)
        else:
            home_pos = (3 * layout.width // 4, layout.height // 2)
        home_dist = distcalc.getDistance(my_pos, home_pos)
        if my_state.isPacman:
            score -= 3 * home_dist  # For attackers, being closer to home is desirable.
        else:
            score += 3 * home_dist  # For defenders, staying away from home might be beneficial.

        return score