# search.py
# ---------
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
from sre_constants import SUCCESS
from cmath import exp


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print "Start:", problem.getStartState()
    print "Is the start a goal?", problem.isGoalState(problem.getStartState())
    print "Start's successors:", problem.getSuccessors(problem.getStartState())
    """
    "*** YOUR CODE HERE ***"
    frontier = util.Stack()
    explored = []
    frontier.push((problem.getStartState(), [], []))
    
    while not frontier.isEmpty():
        currentNode = frontier.pop()
        currentState = currentNode[0]
        currentAction = currentNode[1]
        currentPath = currentNode[2]
         
        if(currentState not in explored):
            explored.append(currentState)
            if(problem.isGoalState(currentState)):
                return currentPath
        
            successorsList = list(problem.getSuccessors(currentState))
            
            for node in successorsList:
                if node[0] not in explored:
                    frontier.push((node[0], node[1], currentPath + [node[1]])) 
    return []

def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    frontier = util.Queue()
    explored = []
    frontier.push((problem.getStartState(), [], []))
    
    while not frontier.isEmpty():
        currentNode = frontier.pop()
        currentState = currentNode[0]
        currentAction = currentNode[1]
        currentPath = currentNode[2]
       
        if(currentState not in explored):
            explored.append(currentState)
            if(problem.isGoalState(currentState)):
                return currentPath
      
            successorsList = list(problem.getSuccessors(currentState))
            for node in successorsList:
                if node[0] not in explored:
                    frontier.push((node[0], node[1], currentPath + [node[1]]))
    return []

def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    frontier = util.PriorityQueue()
    explored = []
    
    frontier.push(((problem.getStartState(), [], 0), [], 0),[])
    
    while not frontier.isEmpty():
        currentNode = frontier.pop()
        currentState = currentNode[0][0]
        currentAction = currentNode[0][1]
        currentPath = currentNode[1]
        currentCost = currentNode[2]
        
        if currentState not in explored:
            explored.append(currentState)
            if (problem.isGoalState(currentState)):
                return currentPath 
            
            successorList = list(problem.getSuccessors(currentState))
            for node in successorList:
                if node[0] not in explored:
                    frontier.push((node,currentPath+[node[1]],currentCost+node[2]),currentCost+node[2])
                    
    return[]                

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    frontier = util.PriorityQueue()
    explored = []
    costFunction = 0
    heuristicFunction = heuristic(problem.getStartState(), problem)
    combinedFunction = costFunction + heuristicFunction
    frontier.push((problem.getStartState(), [], costFunction, []), combinedFunction)
    while not frontier.isEmpty():
        currentNode = frontier.pop()
        currentState = currentNode[0]
        cuurentAction = currentNode[1]
        currentCost = currentNode[2]
        if currentState not in explored:
            currentPath = currentNode[3]
            explored.append(currentState)
            if(problem.isGoalState(currentState)):
                return currentPath 
                   
            successorsList = list(problem.getSuccessors(currentState))
            for node in successorsList:
                if node[0] not in explored:
                    costFunction = currentCost + node[2]
                    heuristicFunction = heuristic(node[0], problem)
                    combinedFunction = costFunction + heuristicFunction
                    frontier.push((node[0], node[1], costFunction, currentPath+[node[1]]), combinedFunction)
    return []


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
