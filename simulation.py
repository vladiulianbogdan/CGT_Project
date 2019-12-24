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
        return "[Wealth = " + str(self.endowment) + "]"

class Population:
    """ represents a popualtion
        TODO
    """
    def __init__(self, populationSize, initializationFunction):
        self.initializationFunction = initializationFunction
        self.populationSize = populationSize
        self.population = []

    def createPopulation(self):
        for newIndividual in range(0, self.populationSize):
            self.population.append(self.initializationFunction())

    def prettyPrintPopulation(self):
        print("Total size of popualtion: " + str(len(self.population)))
        for individual in self.population:
            print(individual)


class Game:
    """

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
        return (plannedContribution if individual.endowment) >= plannedContribution else individual.endowment

    def collectiveLoss(self, selection):
        """" All member loss, ONLY SIDE EFFECT!

        Collective loss happens to all individuals in the seleciton
        depending on their type!
        """
        for individual in selection:
            if individual.individualType:
                individual.endowment *= self.alphaRich
            else:
                individual.endowment *= self.alphaPoor

    def play(self):
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


def randomInitialization(wealth, minThreshold, maxThreshold, minA, maxA, minB, maxB, typeInd):
    """" creates a random individual with given boundaries

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
    """" mock-function for random selection

        Selects N (for N = groupSize) random individuals from the population with replacment.

        Args:
            population(Population): Population to select from, using the population fiel
            groupSize(int): number of individuals choosen

        Returns:
            Array: (Individual, ..., Individual)

    """
    return [rand.choice(population.population) for _ in range(0, groupSize)]

def linearRiskCurve(selection, collectivePot, lambdaValue):
    """""" Equation (1) page 7 of the paper

    Returns:
        True, Loss is happening
        False, Loss is not happening
    """
    w0 = sum()[individual.startingWealth for individual in selection])
    probLoss = (1 - (collectivePot/w0)*lambdaValue)
    if rand.uniform(0,1) > probLoss:
        return False
    else:
        return True


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
def runSimulation(generations, groupSize, popSize, numberOfGames, numberOfRounds, alphaPoor, alphaRich, lambdaValue, fitnessFunction, riskFunction):
    # initialization
    for generation in range(0, number_of_generations):
        for game in range(o, n):
            # select group of K random players
            game = Game()

            game.play()

            # get money
            for round in range(0, n):
                # invest

                # check for possibility of loss

        # mutate the generation

        # keep track of contribution


if __name__ == "__main__":
    print("Running as main!")
    myInitFunction = lambda: randomInitialization(1, 0, 1, 0, 1, 0, 1, False)
    myLinearRiskCurve = lambda selection, collectivePot: linearRiskCurve(selection, collectivePot, 0.5)
    pop = Population(100, myInitFunction)
    pop.createPopulation()
    pop.prettyPrintPopulation()
