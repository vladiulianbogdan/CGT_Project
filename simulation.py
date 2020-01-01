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
import math
import time
import sys
from datetime import datetime

# Using enum class create enumerations
class RiskInRound(enum.Enum):
   EveryRound = 1
   FirstRound = 2
   LastRound = 3
   RandomRound = 4

class RiskCurve(enum.Enum):
   Linear = 1
   PowerFunction = 2
   StepWiseLinear = 3

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
        maxThreshold(float): The maximum value that the threshold of a strategy can take.
    """
    def __init__(self, endowment, strategy, individualType, maxThreshold):
        self.maxThreshold = maxThreshold
        self.startingWealth = endowment # should be immutable
        self.endowment = endowment
        self.strategy = strategy
        self.individualType = individualType
        self.gamesPlayed = 0
        self.totalContribution = 0
        self.cumulatedPayoff = 0.0

    def isRich(self):
        return self.individualType == True

    # TODO add selection intensity
    def getFitness(self):
        """ Get the averaged payoff of this indivdual """
        return math.exp(self.cumulatedPayoff / self.gamesPlayed) if (self.gamesPlayed > 0) else 0

    def __repr__(self):
        return "[fitness = " + str(self.getFitness()) + "]\n" + "[endowment = " + str(self.endowment) + "]\n" + str(self.strategy)

class Population:
    """ represents a population

        There are two ways to create a population:

        1. by calling this constructor with populationSize and then adding individuals
        2. by calling the constructor with populationSize and the rich and poor populations.

        Attributes:
            - populationSize(Int): number of individuals in the population
            - richPopulation([Individual, ...., Individual]): all rich individuals or empty
            - poorPopulation([Individual, ...., Individual]): all poor individuals or empty
    """
    def __init__(self, populationSize, richPopulation=[], poorPopulation=[]):
        if len(richPopulation) != 0 or len(poorPopulation) !=0:
            self.populationSize = len(richPopulation) + len(poorPopulation)
        else:
            self.populationSize = populationSize
        self.population = richPopulation + poorPopulation
        self.richPopulation = richPopulation
        self.poorPopulation = poorPopulation

    def addIndividual(self, individual):
        # add option to concatenate array instead with automatical check
        if (len(self.population) > self.populationSize):
            raise Exception("Population exceeded bound, don't add more elements then allowed!")
        self.population.append(individual)

        if (individual.individualType == True):
            self.richPopulation.append(individual)
        else:
            self.poorPopulation.append(individual)

    def modifyIndividual(self, index, newIndividual):
        self.population[index] = newIndividual
        if index >= len(self.richPopulation):
            self.poorPopulation[index - len(self.richPopulation)] = newIndividual
        else:
            self.richPopulation[index] = newIndividual


    def prettyPrintPopulation(self):
        print("Total size of popualtion: " + str(len(self.population)))
        for individual in self.population:
            print(individual)

class Game:
    """ Initialization of the Game of a population with given functions, ready to play multiple times

    Attributes:
        population(Population): that the population that will play the game
        groupSize(Int): in the paper this is the m value, how many Indviduals are playing a game against each other
        rounds(Int): how many rounds are played against each other?
        riskFunction(Function): chance of risk in the multi-loss risk game
        riskInRound(Int as Enum): In which round does the risk happen/ or are there multiple loses
        alphaPoor(Float): loss fraction for poor individuals
        alphaRich(Float): loss fraction for rich individuals
        heterogeneous(Boolean): False for only one population and True for rich and poor
    """
    def __init__(self, population, groupSize, rounds, riskFunction, riskInRound ,alphaPoor, alphaRich, heterogeneous):
        self.population = population
        self.rounds = rounds
        self.groupSize = groupSize
        self.riskInRound = riskInRound
        self.riskFunction = riskFunction
        self.alphaPoor = alphaPoor
        self.alphaRich = alphaRich
        self.contributionsPerRoundRich = np.zeros(rounds)
        self.contributionsPerRoundPoor = np.zeros(rounds)
        self.heterogeneous = heterogeneous

    def select(self):
        """ use the given selection function to get an amount of indivduals equal to the groupSize """
        return randomSelection(self.population,  self.groupSize, self.heterogeneous)

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
                individual.endowment *= (1 - self.alphaRich)
            else:
                individual.endowment *= (1 - self.alphaPoor)

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
                individual.totalContribution += contribution

                if individual.isRich() == True:
                    self.contributionsPerRoundRich[currentRound] += contribution
                else:
                    self.contributionsPerRoundPoor[currentRound] += contribution
            collectivePot += contributionThisRound

            if (
                (
                (self.riskInRound == RiskInRound.FirstRound and currentRound == 0) or
                (self.riskInRound == RiskInRound.LastRound and currentRound == self.rounds - 1) or
                (self.riskInRound == RiskInRound.EveryRound) or
                (self.riskInRound == RiskInRound.RandomRound and currentRound == int(randomRound))
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
    return Individual(wealth, strategy, typeInd, maxThreshold)

def randomSelection(population, groupSize, heterogeneous):
    """ mock-function for random selection

        Selects N (for N = groupSize) random individuals from the population with replacment.

        Args:
            population(Population or list[Individual.... Individual]): Population to select from, using the population fiel
            groupSize(int): number of individuals choosen
            groupSize(boolean): true if the simulation is heterogeneous, false otherwise

        Returns:
            Array: (Individual, ..., Individual)

    """
    if heterogeneous == False:
        return [rand.choice(population.population) for _ in range(0, groupSize)]
    else:
        richPopulation = [rand.choice(population.richPopulation) for _ in range(0, groupSize)]
        poorPopulation = [rand.choice(population.poorPopulation) for _ in range(0, groupSize)]
        return richPopulation + poorPopulation

def linearRiskCurve(selection, collectivePot, lambdaValue):
    """ Equation (1) page 7 of the paper

    Returns:
        True, Loss is happening
        False, Loss is not happening
    """
    w0 = sum([individual.startingWealth for individual in selection])
    probLoss = (1.0 - (collectivePot/w0)*lambdaValue)
    if rand.uniform(0,1) > probLoss:
        return False
    else:
        return True

def powerRiskCurve(selection, collectivePot, lambdaValue):
    """ Equation (2) page 7 of the paper"""
    w0 = sum([individual.startingWealth for individual in selection])
    probLoss = (1.0 - (collectivePot/w0) ** lambdaValue)
    if rand.uniform(0,1) > probLoss:
        return False
    else:
        return True

def stepWiseRiskCurve(selection, collectivePot, lambdaValue):
    """Equation (3) page 7 of the paper"""
    w0 = sum([individual.startingWealth for individual in selection])
    probLoss = 1.0 / ( math.exp(lambdaValue*(collectivePot/w0 - 0.5)) + 1.0 )
    if rand.uniform(0,1) > probLoss:
        return False
    else:
        return True

def drawValueFromNormalDistribution(mean, high, low = 0, sigma = 0.15):
    """  Draws a random value from a distribution """

    value = np.random.normal(mean, sigma)
    while ( value < low or value > high):
        value = np.random.normal(mean, sigma)
    return value

def simpleMutation(individual, mutationChance = 0.01):
    """ Mutates an indivdual with a certain chance

    Args:
        individual(Individual): individual to mutate if the case
        mutationChance(float): low value, default 1% chance of mutationChance

    Results:
        Individual: either the one give, or a mutated version of that one
    """
    if rand.uniform(0,1) <= mutationChance:
        strategyArray = individual.strategy
        newStrategy = []

        for strategy in strategyArray:
            threshold_new = drawValueFromNormalDistribution(strategy[0], individual.maxThreshold)
            a_new = drawValueFromNormalDistribution(strategy[1], individual.startingWealth)
            b_new = drawValueFromNormalDistribution(strategy[2], individual.startingWealth)

            newStrategy.append([threshold_new, a_new, b_new])
        return Individual(individual.startingWealth, newStrategy, individual.individualType, individual.maxThreshold)
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

def writeContributionDataToFile(file, heterogeneous, averagedContributionsPerRoundRich, averagedContributionsPerRoundPoor):
    if heterogeneous == True:
        file.write("r ")
        for contribution in averagedContributionsPerRoundRich:
            file.write("%f " % contribution)
        file.write("\n")
        file.write("p ")
        for contribution in averagedContributionsPerRoundPoor:
            file.write("%f " % contribution)
        file.write("\n")
    else:
        for contribution in averagedContributionsPerRoundRich:
            file.write("%f " % contribution)
        file.write("\n")
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
                    numberOfRounds, groupSize,
                    popSize, alphaPoor, alphaRich, riskFunction, riskInRound, file, heterogeneous, wealthPoor, wealthRich, typeOfRiskCurve, globalLambdaValue):


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

    if (heterogeneous == False):
        population = Population(popSize)
    else:
        population = Population(2*popSize)

    if (heterogeneous == True):
        for _ in range(0, popSize):
            individual = randomInitialization(wealthRich, True, numberOfRounds, 0, wealthRich * groupSize + wealthPoor * groupSize, 0, wealthRich, 0, wealthRich)
            population.addIndividual(individual)

        for _ in range(0, popSize):
            individual = randomInitialization(wealthPoor, False, numberOfRounds, 0, wealthRich * groupSize + wealthPoor * groupSize, 0, wealthPoor, 0, wealthPoor)
            population.addIndividual(individual)
    else:
        for _ in range(0, popSize):
            individual = randomInitialization(wealthRich, True, numberOfRounds, 0, wealthRich * groupSize, 0, wealthRich, 0, wealthRich)
            population.addIndividual(individual)

    # Save the total average contribution over all rounds, games and generations
    totalAveragedContribution = 0

    totalAveragedContributionsPerRoundRich = np.zeros(numberOfRounds)
    totalAveragedContributionsPerRoundPoor = np.zeros(numberOfRounds)

    # Outline of the process
    for _ in range(0, generations):
        game = Game(population, groupSize, numberOfRounds, riskFunction, riskInRound ,alphaPoor, alphaRich, heterogeneous)
        for _ in range(0, numberOfGames):
            game.play()

        # Compute total average contribution over all rounds, games and generations
        totalAverageThisGeneration = 0
        individualsThatPlayedThisGeneration = 0
        for individual in population.population:
            if individual.gamesPlayed != 0:
                totalAverageThisGeneration += individual.totalContribution / individual.gamesPlayed
                individualsThatPlayedThisGeneration += 1

        totalAveragedContribution += (totalAverageThisGeneration/individualsThatPlayedThisGeneration)

        richPopulation = wrightFisher(population.richPopulation)
        poorPopulation = wrightFisher(population.poorPopulation)
        population = Population(population.populationSize, richPopulation, poorPopulation)
        population = mutation(population)

        # Add all contributions in order to make the average contribution over all generations for each round
        for i in range(0, numberOfRounds):
            totalAveragedContributionsPerRoundRich[i] += game.contributionsPerRoundRich[i]
            totalAveragedContributionsPerRoundPoor[i] += game.contributionsPerRoundPoor[i]

        averagedContributionsPerRoundRich = game.contributionsPerRoundRich / (groupSize * numberOfGames)
        averagedContributionsPerRoundPoor = game.contributionsPerRoundPoor / (groupSize * numberOfGames)

        writeContributionDataToFile(file, heterogeneous, averagedContributionsPerRoundRich, averagedContributionsPerRoundPoor)

    totalAveragedContributionsPerRoundRich /= (groupSize * numberOfGames * generations)
    totalAveragedContributionsPerRoundPoor /= (groupSize * numberOfGames * generations)

    file.write("%f\n" % (totalAveragedContribution / generations))

    for c in totalAveragedContributionsPerRoundRich:
        file.write("%f " % c)
    file.write("\n")

    if heterogeneous == True:
        for c in totalAveragedContributionsPerRoundPoor:
            file.write("%f " % c)
        file.write("\n")

def wrightFisher(populationArray):
    total = 0
    newPopulation = []

    if (len(populationArray) == 0):
        return []

    for individual in populationArray:
        total += individual.getFitness()

    frequencies = [populationArray[0].getFitness() / total]

    for i in range(1, len(populationArray)):
        frequency = populationArray[i].getFitness() / total
        frequencies.append(frequencies[-1] + frequency)

    for i in range(0, len(populationArray)):
        number = rand.uniform(0, 1)

        for j in range(0, len(populationArray)):
            if (number <= frequencies[j]):
                ind = Individual(
                    populationArray[j].endowment,
                    populationArray[j].strategy,
                    populationArray[j].individualType,
                    populationArray[j].maxThreshold
                )
                newPopulation.append(ind)
                break

    return newPopulation

def mutation(population):
    #individualIndex = int(rand.uniform(0, population.populationSize))
    for individualIndex in range(0, population.populationSize):
        population.modifyIndividual(individualIndex, simpleMutation(population.population[individualIndex]))

    return population

if __name__ == "__main__":
    generations = int(sys.argv[1])
    numberOfRounds = int(sys.argv[2])
    groupSize = int(sys.argv[3])
    popSize = int(sys.argv[4])

    riskInRound = RiskInRound(int(sys.argv[5]))

    alphaPoor = float(sys.argv[6])
    alphaRich = float(sys.argv[7])
    print(f"alphaPoor={alphaPoor}")
    print(f"alphaRich={alphaRich}")
    numberOfGames = int(sys.argv[8])
    wealthPoor = float(sys.argv[9])
    wealthRich = float(sys.argv[10])
    typeOfRiskCurve = RiskCurve(int(sys.argv[11]))

    heterogeneous = True if (int(sys.argv[12]) == 1) else False
    globalLambdaValue = float(sys.argv[13])

    filename = sys.argv[14]

    file = open("%s_%d_%d_%d_%d_%s_%0.2f_%0.2f_%d_%0.2f_%0.2f_%s_%d_%0.2f_%s.dat" % (filename, generations, numberOfRounds, groupSize, popSize, riskInRound.name, alphaPoor, alphaRich, numberOfGames, wealthPoor, wealthRich, typeOfRiskCurve.name, heterogeneous, globalLambdaValue, datetime.now().strftime("%d-%m-%Y_%H:%M:%S")), "w+")

    doc = """nr_generations: %d
number_of_rounds: %d
group_size: %d
population_size: %d
risk_in_round: %s
alpha_poor: %0.2f
alpha_rich: %0.2f
number_of_games: %d
wealth_poor: %0.2f
wealth_rich: %0.2f
type_of_risk_curve: %s
heterogenous: %d
lambda_value: %0.2f
""" % (generations, numberOfRounds, groupSize, popSize, riskInRound.name, alphaPoor, alphaRich, numberOfGames, wealthPoor, wealthRich, typeOfRiskCurve.name, heterogeneous, globalLambdaValue)
    print(doc)
    file.write(doc)

    # correspondce to orange curve figure 1
    # correspondce also to red curve figure 1 but with high lambda
    if (RiskCurve.Linear == typeOfRiskCurve):
       riskFunction = lambda selection, collectivePot: linearRiskCurve(selection, collectivePot, globalLambdaValue)
    # correspondce to blue curve figure 1
    elif RiskCurve.StepWiseLinear == typeOfRiskCurve:
       riskFunction = lambda selection, collectivePot: stepWiseRiskCurve(selection, collectivePot, globalLambdaValue)
    # correspondce to black curve figure 1
    elif RiskCurve.PowerFunction == typeOfRiskCurve:
       riskFunction = lambda selection, collectivePot: powerRiskCurve(selection, collectivePot, globalLambdaValue)
    else:
       print("Invalid risk curve.")

    runSimulation(  generations, numberOfGames, \
                    numberOfRounds, groupSize, \
                    popSize,
                    alphaPoor, alphaRich, riskFunction, riskInRound, file, heterogeneous, wealthPoor, wealthRich, typeOfRiskCurve, globalLambdaValue)
