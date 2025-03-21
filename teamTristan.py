# teamTristan.py
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


from captureAgents import CaptureAgent
import random, time, util
from game import Directions
import game
from mcts import MCTS



#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed,
               first = 'OffensivePacmanAgent', second = 'DefensiveGhostAgent'):
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

class OffensivePacmanAgent(CaptureAgent):

  def registerInitialState(self, gameState):
    self.simulation_time = 0.1
    self.max_depth = 6

    CaptureAgent.registerInitialState(self, gameState)
    self.start = gameState.getAgentPosition(self.index)


  def chooseAction(self, gameState):
    start_time = time.time()
    mcts_instance = MCTS(gameState=gameState, agent_index=self.index, max_depth=self.max_depth,
                         time_limit=self.simulation_time, start_time=start_time)
    best_move = mcts_instance.search()

    return best_move

class DefensiveGhostAgent(CaptureAgent):

  def registerInitialState(self, gameState):
    self.simulation_time = 0.1
    self.max_depth = 6

    CaptureAgent.registerInitialState(self, gameState)
    self.start = gameState.getAgentPosition(self.index)

  def chooseAction(self, gameState):

    start_time = time.time()
    mcts_instance = MCTS(gameState=gameState, agent_index= self.index, max_depth=self.max_depth, time_limit=self.simulation_time, start_time=start_time)
    best_move = mcts_instance.search()

    return best_move


