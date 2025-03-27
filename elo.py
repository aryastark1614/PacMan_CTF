import random
import math

class Agent:
    def __init__(self, name, rating=1000):
        self.name = name
        self.rating = rating

    def play(self, opponent):
        """Simulate a match between this agent and an opponent."""
        # A simple random win/loss result for demonstration, replace with actual agent logic
        return random.choice([self, opponent])

    def update_elo(self, opponent, result):
        """Update the agent's ELO rating after a match."""
        # Calculate the expected score based on ELO ratings
        expected_score = 1 / (1 + 10 ** ((opponent.rating - self.rating) / 400))
        
        # The outcome of the match: 1 for a win, 0 for a loss, 0.5 for a draw
        if result == self:
            actual_score = 1
        elif result == opponent:
            actual_score = 0
        else:
            actual_score = 0.5
        
        # K-factor determines how much the rating can change
        K = 32
        # Update the ratings
        self.rating += K * (actual_score - expected_score)
        opponent.rating += K * ((1 - actual_score) - (1 - expected_score))

    def __repr__(self):
        return f"{self.name}: {self.rating:.2f}"

def run_tournament(agents, num_matches=100):
    """Run a tournament with a set of agents."""
    results = {agent.name: 0 for agent in agents}
    
    for i in range(num_matches):
        agent1, agent2 = random.sample(agents, 2)  # Select two random agents
        
        winner = agent1.play(agent2)
        results[winner.name] += 1
        agent1.update_elo(agent2, winner)
    
    return results

# Define agents: add heuristic and MCTS agents
heuristic_agent = Agent("myTeamheuristicArya")
mcts_random_rollouts = Agent("myTeamArya")
mcts_heuristic_rollouts = Agent("baseline")

agents = [heuristic_agent, mcts_random_rollouts, mcts_heuristic_rollouts]

# Run a tournament with 100 matches
results = run_tournament(agents, num_matches=1000)

# Print the final results and ELO ratings
for agent in agents:
    print(agent)

# Print the tournament results (number of wins)
print("\nTournament results (number of wins):")
for name, wins in results.items():
    print(f"{name}: {wins} wins")
