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
from datetime import datetime

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
    """ represents a popualtion

        Attributes:
            - selectionFunction(function(population, fitnessFunction) => [Individual, ... Individual] ): Returns a number of individuals given a population
            - fitnessFunction(function => fitnessvalue): A function that can access all members of the population class, should be used to evaluate fitnessvalue
                                                        together with the selectionFunction
            - populationSize(Int): number of individuals in the population
            - population([Individual, ...., Individual]): all individuals
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
    """
        TODO add Docstring
        risk in round:
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

def randomSelection(population, groupSize, heterogeneous):
    """ mock-function for random selection

        Selects N (for N = groupSize) random individuals from the population with replacment.

        Args:
            population(Population or list[Individual.... Individual]): Population to select from, using the population fiel
            groupSize(int): number of individuals choosen

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

def drawValueFromNormalDistribution(mean, sigma = 0.15):
    """  Draws a random value from a distribution """

    return np.random.normal(mean, sigma)

def simpleMutation(individual, mutationChance = 0.01):
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

        # if there is only one strategy, the a_new will not be a list
        if len(strategy) > 1:
            a_new[a_new < 0] = 0.0
            b_new[b_new < 0] = 0.0
            threshold_new[threshold_new < 0] = 0
            new_strategy = np.array([threshold_new, a_new, b_new]).transpose()
        else:
            if a_new < 0:
                a_new = 0
            if b_new < 0:
                b_new = 0
            if threshold_new < 0:
                threshold_new = 0

            new_strategy = np.array([[threshold_new, a_new, b_new]])
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

def writeHeaderDataToFile(file, heterogeneous, popSize, wealthRich, wealthPoor, numberOfRounds, typeOfRiskCurve, alphaRich, alphaPoor):
    file.write("%d\n" % heterogeneous)
    file.write("%d\n" % popSize) # population size
    if heterogeneous == False:
        file.write("%f\n" % wealthRich) # initial endowment
    else:
        file.write("%f\n" % wealthRich) # initial endowment
        file.write("%f\n" % wealthPoor) # initial endowment
    file.write("%d\n" % numberOfRounds) # number of rounds
    file.write("%d\n" % typeOfRiskCurve.value) # type of risk curve

    if heterogeneous == False:
        file.write("%f\n" % alphaRich)
    else:
        file.write("%f\n" % alphaRich)
        file.write("%f\n" % alphaPoor)
    file.write("%f\n" % globalLambdaValue) # lambda value

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

    if (heterogeneous == False):
        population = Population(popSize)
    else:
        population = Population(2*popSize)

    for _ in range(0, popSize):
        individual = randomInitialization(wealthRich, True, numberOfRounds, 0, 1, 0, wealthRich, 0, wealthRich)
        population.addIndividual(individual)
        population.prettyPrintPopulation()

    if (heterogeneous == True):
        for _ in range(0, popSize):
            individual = randomInitialization(wealthPoor, False, numberOfRounds, 0, 1, 0, wealthPoor, 0, wealthPoor)
            population.addIndividual(individual)
            population.prettyPrintPopulation()

    writeHeaderDataToFile(file, heterogeneous, popSize, wealthRich, wealthPoor, numberOfRounds, typeOfRiskCurve, alphaRich, alphaPoor)

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
        for individual in population.population:
            if individual.gamesPlayed != 0:
                totalAveragedContribution += individual.totalContribution / individual.gamesPlayed
            else:
                totalAveragedContribution += 1

        richPopulation = wrightFisher(population.richPopulation)
        poorPopulation = wrightFisher(population.poorPopulation)
        population = Population(population.populationSize, richPopulation, poorPopulation)
        population = mutation(population)

        # Add all contributions in order to make the average contribution over all generations for each round
        for i in range(0, numberOfRounds-1):
            totalAveragedContributionsPerRoundRich[i] += game.contributionsPerRoundRich[i]
            totalAveragedContributionsPerRoundPoor[i] += game.contributionsPerRoundPoor[i]

        averagedContributionsPerRoundRich = game.contributionsPerRoundRich / (groupSize * numberOfGames)
        averagedContributionsPerRoundPoor = game.contributionsPerRoundPoor / (groupSize * numberOfGames)

        writeContributionDataToFile(file, heterogeneous, averagedContributionsPerRoundRich, averagedContributionsPerRoundPoor)

    totalAveragedContributionsPerRoundRich /= (groupSize * numberOfGames * generations)
    totalAveragedContributionsPerRoundPoor /= (groupSize * numberOfGames * generations)

    file.write("%f\n" % (totalAveragedContribution / (popSize * generations)))

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

    for individual in populationArray:
        total += individual.getFitness()

    frequencies = [populationArray[0].getFitness() / total]

    for i in range(1, len(populationArray)):
        frequency = populationArray[i].getFitness() / total
        frequencies.append(frequencies[-1] + frequency)

    for i in range(0, len(populationArray)):
        number = rand.uniform(0, 1)

        for j in range(0, len(populationArray)):
            if (number < frequencies[j]):
                ind = Individual(
                    populationArray[j].endowment,
                    populationArray[j].strategy,
                    populationArray[j].individualType
                )
                newPopulation.append(ind)
                break

    return newPopulation

def mutation(population):
    #individualIndex = int(rand.uniform(0, population.populationSize))
    for individualIndex in range(0, population.populationSize):
        population.modifyIndividual(individualIndex, simpleMutation(population.population[individualIndex]))

    return population

globalLambdaValue = 10

if __name__ == "__main__":
    print("Running as main!")
    generations = 10000
    numberOfGames = 1000
    numberOfRounds = 4
    groupSize = 2
    popSize = 100

    alphaPoor = 1
    alphaRich = 1
    wealthPoor = 1
    wealthRich = 4
    typeOfRiskCurve = RiskCurve.Linear

    heterogeneous = True

    file = open("simulation_" + datetime.now().strftime("%d-%m-%Y_%H:%M:%S") + ".dat", "w+")

    # The three different risk curves with given lambda values
    # riskFunction = lambda selection, collectivePot: linearRiskCurve(selection, collectivePot, globalLambdaValue)
    #riskFunction = lambda selection, collectivePot: powerRiskCurve(selection, collectivePot, globalLambdaValue)
    riskFunction = lambda selection, collectivePot: stepWiseRiskCurve(selection, collectivePot, globalLambdaValue)
    runSimulation(  generations, numberOfGames, \
                    numberOfRounds, groupSize, \
                    popSize,
                    alphaPoor, alphaRich, riskFunction, RiskInRound.LastRound, file, heterogeneous, wealthPoor, wealthRich, typeOfRiskCurve)
