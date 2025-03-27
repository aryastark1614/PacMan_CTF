import math
import os
import subprocess


# Elo rating update function
def update_elo(rating1, rating2, score1, k=32):
    expected1 = 1 / (1 + 10 ** ((rating2 - rating1) / 400))
    expected2 = 1 - expected1  # Because sum of probabilities = 1

    new_rating1 = rating1 + k * (score1 - expected1)
    new_rating2 = rating2 + k * ((1 - score1) - expected2)

    return new_rating1, new_rating2

# Run Pac-Man games and evaluate
def evaluate_agents(agent1_file, agent2_file, num_games=20):
    agent1_name = os.path.splitext(os.path.basename(agent1_file))[0]
    agent2_name = os.path.splitext(os.path.basename(agent2_file))[0]

    # Initial Elo ratings
    ratings = {agent1_name: 1500, agent2_name: 1500}

    wins = {agent1_name: 0, agent2_name: 0}

    for i in range(num_games):
        print(f"Running game {i+1}/{num_games}...")
        
        # Run Pac-Man game and capture result
        result = subprocess.run(
            ["python", "capture.py", "--red", agent1_name, "--blue", agent2_name],
            capture_output=True, text=True
        )

        # Determine winner based on output (you may need to modify this based on your game output)
        output = result.stdout.lower()
        if "red team" in output:
            winner = agent1_name
            loser = agent2_name
            score = 1
        elif "blue team" in output:
            winner = agent2_name
            loser = agent1_name
            score = 0
        else:
            continue

        wins[winner] += 1

        # Update Elo ratings
        ratings[winner], ratings[loser] = update_elo(ratings[winner], ratings[loser], score)

    print(f"Final Elo Ratings after {num_games} games:")
    print(f"{agent1_name}: {ratings[agent1_name]:.2f}")
    print(f"{agent2_name}: {ratings[agent2_name]:.2f}")

    return ratings

# Example Usage
if __name__ == "__main__":
    agent1 = "baselineTeam.py"
    agent2 = "teamJelkeHeuristic.py"
    evaluate_agents(agent1, agent2, num_games=1)
