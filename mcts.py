import math
import random
from capture import AgentRules


class MCTSNode:
    def __init__(self, state, parent=None, action=None, visits=0, agentIndex=None):
        """
        A node in the MCTS tree.
        :param state: The game state at this node.
        :param parent: Parent node in the tree.
        :param action: The action that was applied to get here (None for the root).
        """
        self.state = state
        self.parent = parent
        self.action = action
        self.agentIndex = agentIndex
        self.children = []
        self.visits = visits
        self.reward = 0.0

    def uct_value(self, exploration_constant=1.41):
        # Calculate the UCT (Upper Confidence Bound for Trees) value for a node
        if self.visits == 0:
            return float('inf')
        return (self.reward / self.visits) + exploration_constant * math.sqrt(math.log(self.parent.visits) / self.visits)


class MCTS:
    def __init__(self, root_state, agent_index, iterations=1000, max_depth=50, exploration_constant=1.41):
        """
        Initialize the MCTS search.
        :param root_state: The current game state (should support deepCopy() and generateSuccessor()).
        :param agent_index: The x of the agent for which we are planning (our agent).
        :param iterations: Number of simulations to run.
        :param max_depth: Maximum depth (number of moves) in a simulation rollout.
        :param exploration_constant: Constant used in UCT calculation.
        """
        self.root = MCTSNode(root_state)
        self.agent_index = agent_index
        self.iterations = iterations
        self.max_depth = max_depth
        self.exploration_constant = exploration_constant

    def search(self):
        """
        Perform MCTS search and return the best action for our agent.
        This simple implementation expands only the first level (our agent's moves) and then runs rollouts for each.
        """
        # Get legal moves for our agent from the current state
        legal_moves = AgentRules.getLegalActions(self.root.state, self.agent_index)
        if not legal_moves:
            return None

        # Expand the root with one child per legal action
        self.root.children = []
        for move in legal_moves:
            next_state = self.root.state.generateSuccessor(self.agent_index, move)
            child_node = MCTSNode(next_state, parent=self.root, action=move)
            self.root.children.append(child_node)

        # Run simulations for a number of iterations
        for _ in range(self.iterations):
            # For simplicity, choose a random child to simulate from
            node = random.choice(self.root.children)
            reward = self.simulate(node.state, self.max_depth)
            node.visits += 1
            node.reward += reward

        # Choose the action with the highest average reward
        best_child = max(self.root.children, key=lambda child: (child.reward / child.visits) if child.visits > 0 else float('-inf'))
        return best_child.action

    def simulate(self, state, depth):
        """
        Simulate a rollout from the given state using random moves until a terminal state is reached or a depth limit is hit.
        :param state: The starting game state for the simulation.
        :param depth: Maximum number of moves to simulate.
        :return: The evaluated reward of the resulting state.
        """
        # Work on a copy of the state to avoid modifying the original
        current_state = state.deepCopy()
        current_depth = 0

        # Determine the total number of agents from the state
        num_agents = len(current_state.data.agentStates)

        # Run simulation until terminal state or depth limit
        while current_depth < depth and not self.is_terminal(current_state):
            # Determine which agent's turn it is; here we use a simple round-robin
            current_agent = current_depth % num_agents
            legal_moves = AgentRules.getLegalActions(current_state, current_agent)
            if not legal_moves:
                break
            move = random.choice(legal_moves)
            current_state = current_state.generateSuccessor(current_agent, move)
            current_depth += 1

        return self.evaluate_state(current_state)

    def is_terminal(self, state):
        """
        Check if the given state is terminal (win or lose).
        This can be extended with additional conditions (e.g., action limits) as needed.
        """
        if hasattr(state.data, '_win') and state.data._win:
            return True
        if hasattr(state.data, '_lose') and state.data._lose:
            return True
        return False

    def evaluate_state(self, state):
        """
        Evaluate the given state.
        This implementation uses the game score as the reward.
        It can be extended to a more nuanced heuristic.
        """
        return state.data.score


# Example usage within an agent's chooseAction method:
# def chooseAction(self, gameState):
#     mcts = MCTS(root_state=gameState, agent_index=MY_AGENT_INDEX, iterations=1000, max_depth=50)
#     best_move = mcts.search()
#     return best_move
