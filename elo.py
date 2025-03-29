import itertools
import math
import os
import subprocess


def update_elo(rating1, rating2, score1, num_matches):
    k = max(10, 32 - num_matches // 10)
    expected1 = 1 / (1 + 10 ** ((rating2 - rating1) / 400))
    expected2 = 1 - expected1
    
    new_rating1 = rating1 + k * (score1 - expected1)
    new_rating2 = rating2 + k * ((1 - score1) - expected2)
    
    return new_rating1, new_rating2

def evaluate_agents(agents, num_rounds=100, log_file="elo_log.txt"):
    ratings = {agent: 1500 for agent in agents}
    match_counts = {agent: 0 for agent in agents}
    
    with open(log_file, "w") as log:
        log.write("Starting Elo Evaluation for Pac-Man Agents\n\n")
        
        for round_num in range(1, num_rounds + 1):
            log.write(f"Round {round_num}\n")
            print(f"Starting Round {round_num}...")
            
            for agent1, agent2 in itertools.combinations(agents, 2):
                print(f"Running {agent1} vs {agent2}...")
                
                result = subprocess.run(
                    ["python", "capture.py", "--layout", "RANDOM", "--red", os.path.splitext(agent1)[0], "--blue", os.path.splitext(agent2)[0]],
                    capture_output=True, text=True
                )
                
                output = result.stdout.lower()
                #print last line of output
                output = output.splitlines()[-1]
                
                if "red team" in output:
                    winner, loser, score = agent1, agent2, 1
                elif "blue team" in output:
                    winner, loser, score = agent2, agent1, 0
                else:
                    continue
                
                match_counts[winner] += 1
                match_counts[loser] += 1
                
                ratings[winner], ratings[loser] = update_elo(ratings[winner], ratings[loser], score, match_counts[winner] + match_counts[loser])
                
                log.write(f"{winner} won against {loser}. New Ratings: {winner}: {ratings[winner]:.2f}, {loser}: {ratings[loser]:.2f}\n")
                print(f"{winner} won against {loser}. New Ratings: {winner}: {ratings[winner]:.2f}, {loser}: {ratings[loser]:.2f}\n")
            log.write("\n")
            print(f"Round {round_num} completed. Updated Elo ratings saved.\n")
        
        log.write("Final Elo Ratings:\n")
        for agent, rating in sorted(ratings.items(), key=lambda x: x[1], reverse=True):
            log.write(f"{agent}: {rating:.2f}\n")
    
    print("Elo evaluation completed. Results saved in", log_file)

if __name__ == "__main__":
    agents = ["teamHeuristic.py", "teamEvalMCTS.py", "teamEvalMastMCTS.py", "teamEvalRaveMCTS.py", "teamVanillaMCTS.py"]
    evaluate_agents(agents, num_rounds=100)