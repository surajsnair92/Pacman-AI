# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to
# http://inst.eecs.berkeley.edu/~cs188/pacman/pacman.html
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
        oldFood = currentGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        
        "*** YOUR CODE HERE ***"
        evaluationScore = 0
        for ghost in newGhostStates:
            distance = util.manhattanDistance(ghost.getPosition(), newPos)
            if (distance <= 1):
                if (ghost.scaredTimer != 0):
                    evaluationScore += 100.0
                else:
                    evaluationScore -= 10.0
    
        for capsule in currentGameState.getCapsules():
            distance = util.manhattanDistance(capsule, newPos)
            if (distance == 0):
                evaluationScore += 5
            else:
                evaluationScore += 1.0 / distance

        for width in range(oldFood.width):
            for height in range(oldFood.height):
                if (oldFood[width][height]):
                    distance = util.manhattanDistance((width, height), newPos)
                    if (distance == 0):
                        evaluationScore += 5.0
                    else:
                        evaluationScore += 1.0 / (distance * distance)
        return evaluationScore

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
        
        
        def max_value(gameState, pos):
            maxVal = float('-inf')
            pos+=1
            if ((pos == self.depth) or (gameState.isWin()) or (gameState.isLose())):
                return self.evaluationFunction(gameState)
            
            for alpha in gameState.getLegalActions(0):
                maxVal =  max(maxVal, min_value(gameState.generateSuccessor(0,alpha), pos, 1))
            return maxVal
        
        def min_value(gameState, pos, ghost):
            minVal  = float('inf')
            if((gameState.isLose()) or (gameState.isWin())):
                return self.evaluationFunction(gameState)
            for beta in gameState.getLegalActions(ghost):
                totalNumberOfAgents = gameState.getNumAgents()
                if ghost == (totalNumberOfAgents -1):
                    minVal = min(minVal, max_value(gameState.generateSuccessor(ghost, beta), pos))
                else:
                    minVal = min(minVal, min_value(gameState.generateSuccessor(ghost, beta), pos, ghost + 1))
            return minVal
        
        pacman = gameState.getLegalActions(0)
        maxVal  = float('-inf')
        maxAction = ''
        for pmove in pacman:
            pos = 0
            maximumValue = min_value(gameState.generateSuccessor(0,pmove),pos,1)
            if maximumValue > maxVal:
                maxVal = maximumValue
                maxAction = pmove
        return maxAction



class AlphaBetaAgent(MultiAgentSearchAgent):
    """
        Your minimax agent with alpha-beta pruning (question 3)
        """
    
    def getAction(self, gameState):
        """
            Returns the minimax action using self.depth and self.evaluationFunction
            """
        "*** YOUR CODE HERE ***"
        agentIndex = 0
        pacman = gameState.getLegalActions(agentIndex)
        maxScore = float('-inf')
        alpha = float('-inf')
        beta = float('inf')
        retAct = ''
        numOfAgents = gameState.getNumAgents()
        for move in pacman:
            
            maxVal = self.abprune(gameState.generateSuccessor(agentIndex,move),self.depth, alpha,beta, agentIndex+1, numOfAgents)
            if maxVal > maxScore:
                retAct = move
                maxScore = maxVal
            if maxScore >= beta:
                return retAct
            alpha = max(alpha,maxScore)
        
        return retAct
    
    def abprune(self, gameState, pos, alpha, beta, agent, tot_agents):
        if pos <= 0  or gameState.isLose() or gameState.isWin():
            return self.evaluationFunction(gameState)
        val = float('inf')
        if agent == 0:
            val = float('-inf')
        for move in gameState.getLegalActions(agent):
            if agent == 0:
                val = max(val, self.abprune(gameState.generateSuccessor(agent, move), pos, alpha, beta, agent + 1, tot_agents))
                alpha = max(alpha,val)
                if(val > beta):
                    return val
            elif agent == tot_agents-1:
                val = min(val, self.abprune(gameState.generateSuccessor(agent, move), pos - 1, alpha, beta, 0, tot_agents))
                beta = min(beta,val)
                if(val < alpha):
                    return val
            else:
                val = min(val, self.abprune(gameState.generateSuccessor(agent, move), pos, alpha, beta, agent + 1, tot_agents))
                beta = min(beta,val)
                if(val<alpha):
                    return val
        return val

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
        "*** YOUR CODE HERE ***"
        
        def max_value(gameState, pos):
            maxVal = float('-inf')
            pos+=1
            if ((pos == self.depth) or (gameState.isWin()) or (gameState.isLose())):
                return self.evaluationFunction(gameState)
            
            for alpha in gameState.getLegalActions(0):
                maxVal =  max(maxVal, expectFunction(gameState.generateSuccessor(0,alpha), pos, 1))
            return maxVal
        
        def expectFunction(gameState, pos, ghost):
            val = 0
            if(gameState.isWin()) or (gameState.isLose()):
                return self.evaluationFunction(gameState)
            ghostAction = gameState.getLegalActions(ghost)
            for gmove in ghostAction:
                totalNumberOfAgents = gameState.getNumAgents()
                if ghost == (totalNumberOfAgents - 1):
                    val += max_value(gameState.generateSuccessor(ghost, gmove), pos)/len(ghostAction)
                else:
                    val+= expectFunction(gameState.generateSuccessor(ghost, gmove), pos, ghost + 1)/len(ghostAction)
            return val
        
        pacman = gameState.getLegalActions(0)
        maxVal = float('-inf')
        maxAction = ''
        for pmove in pacman:
            pos = 0
            maximumValue = expectFunction(gameState.generateSuccessor(0, pmove), pos, 1)
            if maximumValue > maxVal:
                maxVal = maximumValue
                maxAction = pmove
        return maxAction





def betterEvaluationFunction(currentGameState):
    """
        Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
        evaluation function (question 5).
        
        DESCRIPTION:
        """
    
    "*** YOUR CODE HERE ***"
    food = currentGameState.getFood()
    foodPos = food.asList()
    foodDistances = []
    evaluationScore = 0
    ghostStates = currentGameState.getGhostStates()
    capPos = currentGameState.getCapsules()
    pos = currentGameState.getPacmanPosition()
    
    for food in foodPos:
        distance = manhattanDistance(food, pos)
        foodDistances.append(-1 * distance)
    
    if len(foodDistances) == 0:
        foodDistances.append(0)
    
    return max(foodDistances) + currentGameState.getScore()


# Abbreviation
better = betterEvaluationFunction


