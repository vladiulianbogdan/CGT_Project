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

class Individual:
    """A single individual with the information it has in the system

    Attributes:
        endowment(float): the current wealth of the individual
        strategy(np.array(tau, a, b)): the strategy set of the individual
            tau: threshold
            a: result if the threshold is smaller or equal then the given value
            b: result if the threshold is bigger
        type(boolean): either rich or poor individual
    """
    def __init__(self, endowment, strategy, individualType):

        self.endowment = endowment
        self.strategy = strategy
        self.type = individualType

    def __repr__(self):
        return "[Wealth = " + str(self.endowment) + "]\nStrategy:\n"+ str(self.strategy)

class Population:
    def __init__(self, populationSize, initializationFunction):
        self.populationSize = populationSize
        self.population = []

    def createPopulation(self):
        for newIndividual in range(0, self.populationSize):
            population.append(initializationFunction())

class Game:
    def select():
        # returns two elements
        pass

    def play():
        select()

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
