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
"""
TODO-list is above runSimulation! :)
"""

import numpy as np
import random as rand
import enum

# Using enum class create enumerations
class RiskInRound(enum.Enum):
   EveryRound = 1
   FirstRound = 2
   LastRound = 3
   RandomRound = 4

class RiskCurve(enum.Enum):
   Linear = 1
   PieceWiseLinear = 2
   PowerFunction = 3
   Curves = 4

class Individual:
    """A single individual with the information it has in the system

    Attributes:
        startingWealth(float): needed to reset the wealth and also to calculate the risk curve
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
        self.individualType = individualType
        self.gamesPlayed = 0
        self.cumulatedPayoff = 0.0

    # TODO add selection intensity
    def getFitness(self):
        """ Get the averaged payoff of this indivdual """
        return (self.cumulatedPayoff / self.gamesPlayed) if (self.gamesPlayed > 0) else 0

    def __repr__(self):
        return "[Wealth = " + str(self.endowment) + "]\n" + str(self.strategy)

class Population:
    """ represents a popualtion

        Attributes:
            - selectionFunction(function(population, fitnessFunction) => [Individual, ... Individual] ): Returns a number of individuals given a population
            - fitnessFunction(function => fitnessvalue): A function that can access all members of the population class, should be used to evaluate fitnessvalue
                                                        together with the selectionFunction
            - populationSize(Int): number of individuals in the population
            - population([Individual, ...., Individual]): all individuals
    """
    def __init__(self, populationSize):
        self.populationSize = populationSize
        self.population = []

    def addIndividual(self, individual):
        # add option to concatenate array instead with automatical check
        if (len(self.population) > self.populationSize):
            raise Exception("Population exceeded bound, don't add more elements then allowed!")
        self.population.append(individual)

    def prettyPrintPopulation(self):
        print("Total size of popualtion: " + str(len(self.population)))
        for individual in self.population:
            print(individual)

class Game:
    """
        TODO add Docstring
        risk in round: 
    """
    def __init__(self, population, groupSize, rounds, riskFunction, riskInRound, selectionFunction ,alphaPoor, alphaRich):
        self.population = population
        self.rounds = rounds
        self.groupSize = groupSize
        self.riskInRound = riskInRound
        self.riskFunction = riskFunction
        self.selectionFunction = selectionFunction
        self.alphaPoor = alphaPoor
        self.alphaRich = alphaRich
        self.contributionsPerRound = np.zeros(rounds)

    def select(self):
        """ use the given selection function to get an amount of indivduals equal to the groupSize """
        return self.selectionFunction( self.population,  self.groupSize )

    def contribution(self, individual, currentRound, collectivePot):
        """ Checking for the strategy (tau; a, b)

        Select the threshold from the strategy the individual uses this round

        Args:
            currentStrategy(np.array): strategy array as described in the paper page 6/7
            collectivePot(float): how much was contributed so far

        Returns:
                Boolean: Returns True if a should be selected.
                Otherwise b should be selected
        """
        currentStrategy = individual.strategy[currentRound]
        threshold = currentStrategy[0]
        contribution = currentStrategy[1] if collectivePot <= threshold else currentStrategy[2]

        return contribution if (individual.endowment >= contribution) else individual.endowment

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
        """ Play one game with a certain amount of rounds

        Outline:
            - select a number of individuals specified in self.groupSize => selection
            - play a number of rounds with this group
            - at the end of each round we call the risk function, if a loss happens
        """
        selection = self.select()
        collectivePot = 0

        randomRound = rand.uniform(0, self.rounds)
        for currentRound in range(0,self.rounds):
            contributionThisRound = 0
            for individual in selection:
                contribution = self.contribution(individual, currentRound, collectivePot)
                individual.endowment -= contribution
                contributionThisRound += contribution

                self.contributionsPerRound[currentRound] += contribution
            collectivePot += contributionThisRound

            if (
                (
                (self.riskInRound == RiskInRound.FirstRound and currentRound == 0) or
                (self.riskInRound == RiskInRound.LastRound and currentRound == self.rounds) or
                (self.riskInRound == RiskInRound.EveryRound) or
                (self.riskInRound == RiskInRound.RandomRound and currentRound == randomRound)
                ) and
                self.riskFunction(selection, collectivePot)
               ):
                self.collectiveLoss(selection)

        # reset individuals and add payoff for this round
        for individual in selection:
            individual.gamesPlayed += 1
            individual.cumulatedPayoff += individual.endowment
            individual.endowment = individual.startingWealth

def randomInitialization(wealth, typeInd, numberOfRounds, minThreshold=0, maxThreshold=0.5, minA=0, maxA=0.5, minB=0, maxB=0.5):
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
        Individual: a Individual
    """
    threshold = rand.uniform(minThreshold, maxThreshold)
    a = rand.uniform(minA, maxA)
    b = rand.uniform(minB, maxB)
    strategy = np.array([[threshold, a, b]])
    for round in range(1, numberOfRounds):
        threshold = rand.uniform(minThreshold, maxThreshold)
        a = rand.uniform(minA, maxA)
        b = rand.uniform(minB, maxB)
        strategy = np.concatenate((strategy, np.array([[threshold, a, b]])), axis=0)
    return Individual(wealth, strategy, typeInd)

def randomSelection(population, groupSize):
    """ mock-function for random selection

        Selects N (for N = groupSize) random individuals from the population with replacment.

        Args:
            population(Population or list[Individual.... Individual]): Population to select from, using the population fiel
            groupSize(int): number of individuals choosen

        Returns:
            Array: (Individual, ..., Individual)

    """
    if isinstance(population, Population):
        return [rand.choice(population.population) for _ in range(0, groupSize)]
    else:
        return [rand.choice(population) for _ in range(0, groupSize)]

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

def simpleMutation(individual, mutationChance = 0.05):
    """ Mutates an indivdual with a certain chance

    Args:
        individual(Individual): individual to mutate if the case
        mutationChance(float): low value, default 3% chance of mutationChance

    Results:
        Individual: either the one give, or a mutated version of that one
    """
    if rand.uniform(0,1) <= mutationChance:
        strategy = individual.strategy
        threshold_new = drawValueFromNormalDistribution(strategy[:,0])
        a_new = drawValueFromNormalDistribution(strategy[:,1])
        b_new = drawValueFromNormalDistribution(strategy[:,2])

        a_new[a_new < 0] = 0
        b_new[b_new < 0] = 0
        threshold_new[threshold_new < 0] = 0

        new_strategy = np.array([threshold_new, a_new, b_new]).transpose()
        return Individual(individual.startingWealth, new_strategy, individual.individualType)
    else:
        return individual

"""

New functions below here:
    TODO
    - Wrigth-Fisher selection step: selectionFunctionPopulation(Population, fitnessFunction) => [Individual, ..., Individual]
            Now I am using just a random selection
    - FitnessFunction: somehow this on is tied with the wright-fisher process
            Now no fitness function is used for breeding the next generation
    - Inititalization (lambda popSize, numberOfRounds => Individual): give the simulation a executable lambda to initialize this one
            How are my strategies set in the beginning?
    - add other risk-curves
    - testing of the implementation
    - how to save our simulation / save the results?
    - split project into several files?
"""


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
                    popSize, alphaPoor, alphaRich, riskFunction, riskInRound, file, heterogeneous, wealthPoor, wealthRich, typeOfRiskCurve):
    """
    Args:
        first-line: simulation parameter
        second-line: game parameter
        third-line: population parameter
        fourth-line individual parameter

    Result:
        ToBeAdded
    """
    # Initialization
    population = Population(popSize)
    for _ in range(0, popSize):
        individual = randomInitialization(wealthRich, True, numberOfRounds)
        population.addIndividual( individual )
    population.prettyPrintPopulation()

    file.write("%d\n" % heterogeneous)
    file.write("%d\n" % popSize) # population size
    if heterogeneous == False:
        file.write("%d\n" % wealthRich) # initial endowment
    else:
        file.write("%d\n" % wealthRich) # initial endowment
        file.write("%d\n" % wealthPoor) # initial endowment
    file.write("%d\n" % numberOfRounds) # number of rounds
    file.write("%d\n" % typeOfRiskCurve.value) # type of risk curve

    if heterogeneous == False:
        file.write("%d\n" % alphaRich)
    else:
        file.write("%d\n" % alphaRich)
        file.write("%d\n" % alphaPoor)
    file.write("%f\n" % 0.5) # lambda value

    # Outline of the process
    for _ in range(0, generations):
        game = Game(population, groupSize, numberOfRounds, riskFunction, riskInRound, selectionFunctionGame ,alphaPoor, alphaRich)
        for _ in range(0, numberOfGames):
            game.play()

        averagedContributionsPerRound = game.contributionsPerRound / (groupSize * numberOfGames)

        for contribution in averagedContributionsPerRound:
            file.write("%f " % contribution)
        file.write("\n")

def wrightFisher(population):
    total = 0
    newPopulation = Population(population.populationSize)

    for individual in population.population:
        total += individual.getFitness()

    frequencies = [population.population[0].getFitness() / total]

    for i in range(1, population.populationSize):
        frequency = population.population[i].getFitness() / total
        frequencies.append(frequencies[-1] + frequency)

    for i in range(0, population.populationSize):
        number = rand.uniform(0, 1)

        for j in range(0, population.populationSize):
            if (number < frequencies[j]):
                ind = Individual(
                    population.population[j].endowment,
                    population.population[j].strategy,
                    population.population[j].individualType
                )
                newPopulation.addIndividual(ind)
                break

    return newPopulation

def mutation(population):
    individualIndex = int(rand.uniform(0, population.populationSize - 1))
    population.population[individualIndex] = simpleMutation(population.population[individualIndex])

    return population


if __name__ == "__main__":
    print("Running as main!")
    generations = 100
    numberOfGames = 1000
    numberOfRounds = 2
    groupSize = 2
    selectionFunctionGame = lambda pop, groupSize: randomSelection(pop, groupSize)
    popSize = 50
    # initFunction = lambda rounds: randomInitialization(1, 0, 0.5, 0, 0.5, 0, 0.5, False, rounds)
    alphaPoor = 0.5
    alphaRich = 0.5
    wealthPoor = 1
    wealthRich = 1
    typeOfRiskCurve = RiskCurve.Linear

    heterogeneous = True

    file = open("simulation.dat", "w+")

    riskFunction = lambda selection, collectivePot: linearRiskCurve(selection, collectivePot, 0.5)
    runSimulation(  generations, numberOfGames, \
                    numberOfRounds, groupSize, selectionFunctionGame, \
                    popSize, 
                    alphaPoor, alphaRich, riskFunction, RiskInRound.FirstRound, file, heterogeneous, wealthPoor, wealthRich, typeOfRiskCurve)
