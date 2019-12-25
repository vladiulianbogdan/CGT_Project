#parameters:
    # - group size
    # - number of generations
    # - population size
    # - number of games
    # - number of rounds
    # - alpha value
    # - lambda value
    # - rich / poor scenario
    # - fitness function
    # - risk function

# allocate random strategies to all players
# allocate initial amount of money

import numpy as np
import random as rand

class Individual:
    """A single individual with the information it has in the system

    Attributes:
        endowment(float): the current wealth of the individual
        strategy(np.array(tau, a, b)): the strategy set of the individual
            tau(float): threshold
            a(float): result if the threshold is smaller or equal then the given value
            b(float): result if the threshold is bigger
        type(boolean): either poor(False) or rich(True) individual
        roundsPlayed(int): how many rounds did this individual play
        cumulatedPayoff(float): payoff for all rounds
    """
    def __init__(self, endowment, strategy, individualType):
        self.startingWealth = endowment # should be immutable
        self.endowment = endowment
        self.strategy = strategy
        self.type = individualType
        self.roundsPlayed = 0
        self.cumulatedPayoff = 0.0

    def getPayoff(self):
        """ Get the averaged payoff of this indivdual """
        return (self.cumulatedPayoff / self.roundsPlayed) if (self.roundsPlayed > 0) else 0

    def __repr__(self):
        return "[Wealth = " + str(self.endowment) + "]\n" + str(self.strategy)

class Population:
    """ represents a popualtion
        TODO add Docstring
    """
    def __init__(self, populationSize, selectionFunction, fitnessFunction):
        self.selectionFunction = selectionFunction
        self.fitnessFunction = fitnessFunction
        self.populationSize = populationSize
        self.population = []

    def addIndividual(self, individual):
        # add option to concatenate array instead with automatical check
        if (len(self.population) > self.populationSize):
            raise Exception("Population exceeded bound, don't add more elements then allowed!")
        self.population.append(individual)

    def selectParent(self):
        return self.selectionFunction ( self.population, self.fitnessFunction )

    def prettyPrintPopulation(self):
        print("Total size of popualtion: " + str(len(self.population)))
        for individual in self.population:
            print(individual)

class Game:
    """
        TODO add Docstring
    """
    def __init__(self, population, groupSize, rounds, riskFunction, selectionFunction ,alphaPoor, alphaRich):
        self.population = population
        self.rounds = rounds
        self.groupSize = groupSize
        self.riskFunction = riskFunction
        self.selectionFunction = selectionFunction
        self.alphaPoor = alphaPoor
        self.alphaRich = alphaRich

    def select(self):
        """ use the given selection function to get an amount of indivduals equal to the groupSize """
        return self.selectionFunction( self.population,  self.groupSize )

    def isContributing(self, currentStrategy, collectivePot):
        """ Checking for the strategy (tau; a, b)

        Select the threshold from the strategy the individual uses this round

        Args:
            currentStrategy(np.array): strategy array as described in the paper page 6/7
            collectivePot(float): how much was contributed so far

        Returns:
                Boolean: Returns True if a should be selected.
                Otherwise b should be selected
        """
        threshold = currentStrategy[0]
        return True if collectivePot <= threshold else False

    def chooseContribution(self, currentStrategy, choice):
        """ returns this indiduals contribution value

        Args:
            currentStrategy(np.array): strategy array as described in the paper page 6/7
            choice(boolean): Depending on the boolean delivered from isContributing

        Result:
            Float: for True we choose currentStrategy[1] which equals a
            otherwise we choose currentStrategy[2] which equals b
        """
        return currentStrategy[1] if choice else currentStrategy[2]

    def getActualContribution(self, individual, plannedContribution):
        """ checks how much the individual can actually contribute """
        return plannedContribution if (individual.endowment >= plannedContribution) else individual.endowment

    def collectiveLoss(self, selection):
        """" All member loss, ONLY SIDE EFFECTS!

        Collective loss happens to all individuals in the seleciton
        depending on their type!
        """
        for individual in selection:
            if individual.individualType:
                individual.endowment *= self.alphaRich
            else:
                individual.endowment *= self.alphaPoor

    def play(self):
        """ Play one game with a certain amount of rounds """
        selection = self.select()
        collectivePot = 0
        for currentRound in range(0,self.rounds):
            contributionThisRound = 0
            for individual in selection:
                currentStrategy = individual.strategy[currentRound]
                choice = self.isContributing(currentStrategy, collectivePot)
                plannedContribution = self.chooseContribution(currentStrategy, choice)
                actualContribution = self.getActualContribution(individual, plannedContribution)
                individual.endowment -= actualContribution
                contributionThisRound += actualContribution
            collectivePot += contributionThisRound
            if self.riskFunction(selection, collectivePot):
                self.collectiveLoss(selection)
        # reset individuals and add payoff for this round
        for individual in selection:
            individual.roundsPlayed += 1
            individual.cumulatedPayoff += indivdual.endowment
            individual.endowment = indivdual.startingWealth

def randomInitialization(wealth, minThreshold, maxThreshold, minA, maxA, minB, maxB, typeInd):
    """ creates a random individual with given boundaries

    Function initializes an Individual with given wealth. It selects uniformly a value
    values between the boundaries for the threshold, a and b.

    Args:
        wealth(Float): initial wealth of the individual
        minThreshold(Float): minimum threshold of the individual (inclusive)
        maxThreshold(Float): maximum threshold of the individual (inclusive)
        minA(Float): minimum a of the individual (inclusive)
        maxA(Float): maximum a of the individual (inclusive)
        minB(Float): minimum b of the individual (inclusive)
        maxB(Float): maximum b of the individual (inclusive)
        type(boolean): poor = False, rich = True

    Result:
        Individual: a individual
    """
    threshold = rand.uniform(minThreshold, maxThreshold)
    a = rand.uniform(minA, maxA)
    b = rand.uniform(minB, maxB)
    strategy = np.array([threshold, a, b])
    return Individual(wealth, strategy, typeInd)

def randomSelection(population, groupSize):
    """ mock-function for random selection

        Selects N (for N = groupSize) random individuals from the population with replacment.

        Args:
            population(Population): Population to select from, using the population fiel
            groupSize(int): number of individuals choosen

        Returns:
            Array: (Individual, ..., Individual)

    """
    return [rand.choice(population.population) for _ in range(0, groupSize)]

def linearRiskCurve(selection, collectivePot, lambdaValue):
    """ Equation (1) page 7 of the paper

    Returns:
        True, Loss is happening
        False, Loss is not happening
    """
    w0 = sum([individual.startingWealth for individual in selection])
    probLoss = (1 - (collectivePot/w0)*lambdaValue)
    if rand.uniform(0,1) > probLoss:
        return False
    else:
        return True

def drawValueFromNormalDistribution(mean, sigma = 0.15):
    """  Draws a random value from a distribution """
    return np.random.normal(mean, sigma)

def simpleMutation(individual, mutationChance = 0.03):
    """ Mutates an indivdual with a certain chance

    Args:
        individual(Individual): individual to mutate if the case
        mutationChance(float): low value, default 3% chance of mutationChance

    Results:
        Individual: either the one give, or a mutated version of that one
    """
    if rand.uniform(0,1) <= mutationChance:
        strategy = individual.strategy
        threshold_new = drawValueFromNormalDistribution(strategy[0])
        a_new = drawValueFromNormalDistribution(strategy[1])
        b_new = drawValueFromNormalDistribution(strategy[2])
        return Individual(individual.endowment, np.array([threshold_new, a_new, b_new]), indivdual.typeInd)
    else:
        return indivdual


#parameters:
    # - group size
    # - number of generations
    # - population size
    # - number of games
    # - number of rounds
    # - alpha value
    # - lambda value
    # - rich / poor scenario
    # - fitness function
    # - risk function
def runSimulation(  generations, numberOfGames,
                    numberOfRounds, groupSize, selectionFunctionGame,
                    popSize, initFunction, mutationFunction, selectionFunctionPopulation, fitnessFunction,
                    alphaPoor, alphaRich, riskFunction):
    """
    Args:
        first-line: Simulation parameter
        second-line: game parameter
        third-line: population parameter
        fourth-line individual parameter
    """
    population = Population(popSize, selectionFunctionPopulation, fitnessFunction)
    for _ in range(0, popSize):
        individual = initFunction()
        population.addIndividual( individual )
    population.prettyPrintPopulation()
    for generation in range(0, generations):
        game = Game(population, groupSize, numberOfRounds, riskFunction, selectionFunctionGame ,alphaPoor, alphaRich)
        for _ in range(0, numberOfGames):
            game.play()
        child_population = Population(popSize, selectionFunctionPopulation, fitnessFunction )
        for _ in range(0, popSize):
            individual = population.selectParent()
            mutated_individual = mutationFunction( individual )
            child_population.addIndividual( mutated_individual )
        population = child_population

        # keep track of contribution


if __name__ == "__main__":
    print("Running as main!")
    generations = 100
    numberOfGames = 1000
    numberOfRounds = 2
    groupSize = 2
    selectionFunctionGame = lambda pop, groupSize: randomSelection(pop, groupSize)
    popSize = 50
    initFunction = lambda: randomInitialization(1, 0, 1, 0, 1, 0, 1, False)
    mutationFunction = lambda indivdual: simpleMutation(individual, 0.04)
    selectionFunctionPopulation = lambda pop, fitnessFunction: randomSelection(pop, 1)
    fitnessFunction = lambda fitness: fitness
    alphaPoor = 0.5
    alphaRich = 0.5
    riskFunction = lambda selection, collectivePot: linearRiskCurve(selection, collectivePot, 0.5)
    runSimulation(  generations, numberOfGames, \
                    numberOfRounds, groupSize, selectionFunctionGame, \
                    popSize, initFunction, mutationFunction, selectionFunctionPopulation, fitnessFunction, \
                    alphaPoor, alphaRich, riskFunction)
