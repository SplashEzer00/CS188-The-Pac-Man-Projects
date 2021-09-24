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
        您不需要更改此方法，但欢迎您这样做。 getAction根据评估函数在最佳选项中进行选择。
        就像之前的项目一样，getAction接受一个GameState并返回一些方向。对于集合{NORTH, SOUTH, WEST, EAST, STOP}中的某个X
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "如果您愿意，可以在这里添加更多代码"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        在这里设计一个更好的评估函数。
        评估函数接受当前的和建议的后续GameStates (pacman.py)并返回一个数字，其中数字越大越好。
        下面的代码从状态中提取一些有用的信息，比如剩余的食物(newFood)和移动后的吃豆人位置(newPos)。
        newScaredTimes提供了每个幽灵会因为吃豆人吃了能量球而保持恐惧的移动数。
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)#生成指定pacman移动后的后续状态
        newPos = successorGameState.getPacmanPosition()#得到吃豆子的位置
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()#获得鬼的位置
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        #Mycode
        newFood = successorGameState.getFood().asList()#获得食物的位置
        nearestFood = float('inf')#初始化最近食物距离为正无穷
        for food in newFood:
            nearestFood = min(nearestFood, manhattanDistance(newPos, food))#取最近食物和xy1和xy2之间的曼哈顿距离的较小值

        for ghost in successorGameState.getGhostPositions():
            if (manhattanDistance(newPos, ghost) < 3):#如果离鬼近，评估函数则返回负无穷
                return -float('inf')

        return successorGameState.getScore() + 1.0/nearestFood#离食物越近，评估函数越高

def scoreEvaluationFunction(currentGameState):
    """
    这个默认的评估函数只是返回状态的分数。 分数与在吃豆人GUI中显示的分数相同。
    这个评估函数用于对抗搜索代理(而不是反射代理)。
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    这个类提供了一些公共元素
    多智能搜索。这里定义的任何方法都是可用的
    到MinimaxPacmanAgent, AlphaBetaPacmanAgent和ExpectimaxPacmanAgent。
    您*不*需要在这里做任何更改，但如果您想为所有对抗性搜索代理添加功能，则可以。但是，请不要删除任何东西。
    注意:这是一个抽象类:一个不应该实例化的类。它只是部分指定，并被设计为可扩展的。Agent (game.py)是另一个抽象类。
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    你的极小极大代理(question 2)
    """
    """
    使用self.depth和self.evaluationFunction从当前游戏状态返回极大极小动作。
    下面是实现minimax时可能有用的一些方法调用。
    
    gameState.getLegalActions (agentIndex):
    传入的参数 agentIndex = 0 表示吃豆人，>= 1 表示幽灵，表明不止一只幽灵
    gameState。generateSuccessor (agentIndex, action):
    在Agents执行操作后返回后续游戏状态
    gameState.getNumAgents ():
    返回游戏中agents的总数
    gameState.isWin ():
    返回游戏状态是否为获胜状态
    gameState.isLose ():
    返回游戏状态是否为失败状态
    """
    def getAction(self, gameState):
        actions = gameState.getLegalActions(0)
        return max(actions, key=lambda x: self.minimaxSearch(gameState.generateSuccessor(0, x), 1))#选择吃豆人与MINMAX算法的较优操作

    def minimaxSearch(self, gameState, turn):#MINMAX算法
        numOfAgents = gameState.getNumAgents()#游戏Agents总数
        agentIndex = turn % numOfAgents#计算角色轮转
        depth = turn // numOfAgents#计算深度
        if gameState.isWin() or gameState.isLose() or depth == self.depth:
            return self.evaluationFunction(gameState)#进行评估
        actions = gameState.getLegalActions(agentIndex)#吃豆人下一步
        evals = [self.minimaxSearch(gameState.generateSuccessor(agentIndex, action), turn + 1) for action in actions]#递归
        if agentIndex > 0:#一对多MINMAX算法，
            return min(evals)
        else:
            return max(evals)

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    你的极小最大值代理和α - β修剪(question 3)

    Returns 使用self.depth和self.evaluationFunction实现极大极小动作
    """
    def getAction(self, gameState):
        actions = gameState.getLegalActions(0)#合法行动
        alpha, beta = -float('inf'), float('inf')#初始化
        vals = []
        for action in actions:
          val = self.alphabetaSearch(gameState.generateSuccessor(0, action), 1, alpha, beta)
          alpha = max(alpha, val)#记录最优值
          vals.append(val)#合并
        for i in range(len(actions)):
          if alpha == vals[i]:
            return actions[i]#返回最优解

    def alphabetaSearch(self, gameState, turn, alpha, beta):
        numOfAgents = gameState.getNumAgents()#获取agents个数
        agentIndex = turn % numOfAgents#记录角色
        depth = turn // numOfAgents#记录深度
        if gameState.isWin() or gameState.isLose() or depth == self.depth:#跳出条件
          return self.evaluationFunction(gameState)
        actions = gameState.getLegalActions(agentIndex)#记录动作
        if agentIndex == 0: val = -float('inf')#记录alpha/beta（跟当前agent类型判断）
        else: val = float('inf')
        for action in actions:#行动
          successor = gameState.generateSuccessor(agentIndex, action)
          #alphabeta算法
          if agentIndex > 0:
            val = min(val, self.alphabetaSearch(successor, turn + 1, alpha, beta))#记录beta
            if val < alpha: return val#剪枝
            else: beta = min(beta, val)
          else:
            val = max(val, self.alphabetaSearch(successor, turn + 1, alpha, beta))#记录alpha
            if val > beta: return val#剪枝
            else: alpha = max(alpha, val)
        return val

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
    你expectimax代理(question 4)

    Returns 使用self.depth和self.evaluationFunction来实现expectimax动作
    所有的幽灵都应该被建模为从它们的合法移动中一致随机选择。蒙特卡洛树搜索？
    """
    def getAction(self, gameState):
        actions = gameState.getLegalActions(0)
        return max(actions, key=lambda x: self.expectimaxSearch(gameState.generateSuccessor(0, x), 1))

    def expectimaxSearch(self, gameState, turn):
        numOfAgents = gameState.getNumAgents()#获得agent个数
        agentIndex = turn % numOfAgents#记录角色
        depth = turn // numOfAgents#记录深度
        if gameState.isWin() or gameState.isLose() or depth == self.depth:#跳出条件
            return self.evaluationFunction(gameState)#返回评估函数
        actions = gameState.getLegalActions(agentIndex)#记录行动
        evals = [self.expectimaxSearch(gameState.generateSuccessor(agentIndex, action), turn + 1) for action in actions]#递归
        #蒙特卡洛树搜索
        if agentIndex > 0:
            return sum(evals) * 1.0 / len(evals)#返回期望值（平均值）
        return max(evals)#返回最大值

def betterEvaluationFunction(currentGameState):
    """
    你极端的捉鬼，抓丸，吃东西，不可阻挡的评估功能(question 5).
    """
    if currentGameState.isLose(): return - float('inf')
    if currentGameState.isWin():  return float('inf')
    foods = currentGameState.getFood()#获取食物位置
    ghostStates = currentGameState.getGhostStates()#获取鬼位置
    pacmanPostion = currentGameState.getPacmanPosition()#获取吃豆人位置

    nearestFood = min(manhattanDistance(food, pacmanPostion) for food in foods.asList())#获得最近食物的分数（曼哈顿距离越近越高）
    coverMe = sum([(manhattanDistance(ghost.getPosition(), pacmanPostion) < 3) for ghost in ghostStates])#获得鬼的个数（距离<3）
    scareMe = sum([(ghost.scaredTimer == 0) for ghost in ghostStates])#获得惊吓时间分数（越久越大）
    #最近食物的分数越小，更好评估函数返回值越大；鬼的个数越少，更好评估函数返回值越大；鬼的惊吓时间越长，更好评估函数返回值越大；
    return currentGameState.getScore() + 1.0 / nearestFood + 1.0 * coverMe + 1.0 / (scareMe + 0.1)


# Abbreviation
better = betterEvaluationFunction