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
    # -

# allocate random strategies to all players

# allocate initial amount of money

import numpy as np
import random as rand

class Individual:
    """A single individual with the information it has in the system

    Attributes:
        endowment(float): the current wealth of the individual
        strategy(np.array(tau, a, b)): the strategy set of the individual
            tau: threshold
            a: result if the threshold is smaller or equal then the given value
            b: result if the threshold is bigger
        type(boolean): either poor(False) or rich(True) individual
    """
    def __init__(self, endowment, strategy, individualType):

        self.endowment = endowment
        self.strategy = strategy
        self.type = individualType

    def __repr__(self):
        return "[Wealth = " + str(self.endowment) + "]\nStrategy:\n"+ str(self.strategy)

class Population:
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
    def select(self, population, selectionFunction):
        # returns two elements
        pass

    def play():
        select()

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

def randomSelection(population):
    """ mock-function for random selection

        Selects to random individuals from the population with replacment

        Args:
            population(Population): Population to select from, using the population fiel

        Returns:
            Tuple: (Individual, Individual)

    """
    return (rand.choice(population.population), rand.choice(population.population))

"""
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
"""

if __name__ == "__main__":
    print("Running as main!")
    myInitFunction = lambda: randomInitialization(1, 0, 1, 0, 1, 0, 1, False)
    pop = Population(100, myInitFunction)
    pop.createPopulation()
    pop.prettyPrintPopulation()
