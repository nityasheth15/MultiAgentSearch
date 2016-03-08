# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent
from pacman import GameState

class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        #newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        
        reward = 0
        distanceFromGhost = 0
        nearestFoodDistance = 100000;
        
        for ghost in newGhostStates:
            distanceFromGhost = util.manhattanDistance(newPos, ghost.getPosition())
            reward = reward + distanceFromGhost*5
        
        if distanceFromGhost < 2:
            reward -= 100000
        
        for aFood in newFood.asList():
            tempDistance = util.manhattanDistance(newPos, aFood)
            if(tempDistance < nearestFoodDistance):
                nearestFoodDistance = tempDistance
                
        reward = reward - nearestFoodDistance*10
        
        if newFood.asList().__contains__(newPos):
            reward += 1000
        
        reward = reward + -1000 * len(newFood.asList())
        
        if(len(newFood.asList()) == 0):
            reward = 1000
        
        return reward

def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game
        """
        "*** YOUR CODE HERE ***"
        
        action = self.minimax(gameState, 0)
        return action[0]
    
    
    def minimax(self, gameState, depth):
        evalMaxValue = (None, 0, 0) #Just like initiating. we need (action, self.evaluation(gameState), depth)
        evalCostForMaximizer  = 0
        evalCostForMinimizer = 10000
        evalMinVal = (None, 10000, 0)
        if depth == self.depth*gameState.getNumAgents() or gameState.isWin() or gameState.isLose(): #if the node is a leaf node
            evaluationValue = (None, self.evaluationFunction(gameState), depth)
            return evaluationValue
        
        #maximizer calculation
        if depth % gameState.getNumAgents() == 0: #if current node is a maximizer
            legalActions = gameState.getLegalActions(0) #for pacman, agentIndex = 0
            
            evalCostForMaximizer = -1000000
            for eachAction in legalActions:
                successor = gameState.generateSuccessor(0, eachAction)
                tempEvalValue = self.minimax(successor, depth+1)
                if tempEvalValue[1] > evalCostForMaximizer: # or gameState.isWin() or gameState.isLose():
                    evalMaxValue = (eachAction, tempEvalValue[1], depth+1)
                    evalCostForMaximizer = tempEvalValue[1]
            return evalMaxValue
        #minimizer calculation
        else:
            ply = depth%gameState.getNumAgents()
            legalActions = gameState.getLegalActions(ply) #nth ply
            
            evalCostForMinimizer = 10000000
            for eachAction in legalActions:
                successor = gameState.generateSuccessor(ply, eachAction)
                tempEvalValue = self.minimax(successor, depth+1)
                if tempEvalValue[1] < evalCostForMinimizer: #or gameState.isWin() or gameState.isLose():
                    evalMinVal = (eachAction, tempEvalValue[1], depth+1)
                    evalCostForMinimizer = tempEvalValue[1] 
            return evalMinVal
                
        return evalMaxValue        

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """
    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        action = self.minimaxalphabeta(gameState, 0, -1000000, 1000000)
        return action[0]
        util.raiseNotDefined()
        
    def minimaxalphabeta(self, gameState, depth, alpha, beta):
        evalMaxValue = (None, 0, 0) #Just like initiating. we need (action, self.evaluation(gameState), depth)
        evalCostForMaximizer  = 0
        evalCostForMinimizer = 10000
        evalMinVal = (None, 10000, 0)
        if depth == self.depth*gameState.getNumAgents() or gameState.isWin() or gameState.isLose(): #if the node is a leaf node
            evaluationValue = (None, self.evaluationFunction(gameState), depth, alpha, beta)
            return evaluationValue
        
        #maximizer calculation
        if depth % gameState.getNumAgents() == 0: #if current node is a maximizer
            legalActions = gameState.getLegalActions(0) #for pacman, agentIndex = 0
            
            evalCostForMaximizer = -1000000
            for eachAction in legalActions:
                successor = gameState.generateSuccessor(0, eachAction)
                tempEvalValue = self.minimaxalphabeta(successor, depth+1, alpha, beta)
                if tempEvalValue[1] > evalCostForMaximizer: # or gameState.isWin() or gameState.isLose():
                    evalCostForMaximizer = tempEvalValue[1]
                    evalMaxValue = (eachAction, tempEvalValue[1], depth+1, alpha, beta)
                if(evalMaxValue[1] > beta):
                    return evalMaxValue
                alpha = max(evalMaxValue[1], alpha)
            return evalMaxValue
        #minimizer calculation
        else:
            ply = depth%gameState.getNumAgents()
            legalActions = gameState.getLegalActions(ply) #nth ply
            
            evalCostForMinimizer = 10000000
            for eachAction in legalActions:
                successor = gameState.generateSuccessor(ply, eachAction)
                tempEvalValue = self.minimaxalphabeta(successor, depth+1, alpha, beta)
                if tempEvalValue[1] < evalCostForMinimizer: #or gameState.isWin() or gameState.isLose():
                    evalMinVal = (eachAction, tempEvalValue[1], depth+1, alpha, beta)
                    evalCostForMinimizer = tempEvalValue[1]
                if(evalMinVal[1] < alpha):
                    return tempEvalValue
                beta = min(evalMinVal[1], beta)
            return evalMinVal
                
        return evalMaxValue 
        
        
class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        action = self.minimax(gameState, 0)
        return action[0]
        
    def minimax(self, gameState, depth):
        evalMaxValue = (None, 0, 0) #Just like initiating. we need (action, self.evaluation(gameState), depth)
        evalCostForMaximizer  = 0
        evalCostForMinimizer = 10000
        evalMinVal = (None, 10000, 0)
        if depth == self.depth*gameState.getNumAgents() or gameState.isWin() or gameState.isLose(): #if the node is a leaf node
            evaluationValue = (None, self.evaluationFunction(gameState), depth)
            return evaluationValue
        
        #maximizer calculation
        if depth % gameState.getNumAgents() == 0: #if current node is a maximizer
            legalActions = gameState.getLegalActions(0) #for pacman, agentIndex = 0
            
            evalCostForMaximizer = -1000000
            for eachAction in legalActions:
                successor = gameState.generateSuccessor(0, eachAction)
                tempEvalValue = self.minimax(successor, depth+1)
                if tempEvalValue[1] > evalCostForMaximizer: # or gameState.isWin() or gameState.isLose():
                    evalMaxValue = (eachAction, tempEvalValue[1], depth+1)
                    evalCostForMaximizer = tempEvalValue[1]
            return evalMaxValue
        #minimizer calculation
        else:
            ply = depth%gameState.getNumAgents()
            legalActions = gameState.getLegalActions(ply) #nth ply
            
            evalCostForMinimizer = 10000000
            for eachAction in legalActions:
                successor = gameState.generateSuccessor(ply, eachAction)
                tempEvalValue = self.minimax(successor, depth+1)
                if tempEvalValue[1] < evalCostForMinimizer: #or gameState.isWin() or gameState.isLose():
                    evalMinVal = (eachAction, tempEvalValue[1], depth+1)
                    evalCostForMinimizer = tempEvalValue[1] 
            return evalMinVal
                
        return evalMaxValue   

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    
    reward = 0
    
    listOfFood = currentGameState.getFood().asList()
    
    tempDist = 0
    dist = 100000
    
    for aFood in listOfFood:
        tempDist = util.manhattanDistance(aFood, currentGameState.getPacmanPosition())
        if tempDist < dist:
            dist = tempDist
            
    reward -= dist*10
    
    reward = reward + -100 * len(listOfFood)
    
    reward +=  -1000*len(currentGameState.getCapsules())
    
    reward -= 100*currentGameState.getNumAgents()
    
    
    if listOfFood.__contains__(currentGameState.getPacmanPosition()):
        reward += 1000
    
    ghosts = currentGameState.getGhostPositions()
    
    for aGhost in ghosts:
        distance = util.manhattanDistance(currentGameState.getPacmanPosition(), aGhost)
        reward += distance*5
    
    if distance  < 2:
            reward -= 100000
    
    if(len(listOfFood) == 0):
            reward = 10000
    
    return reward
    
    
# Abbreviation
better = betterEvaluationFunction

