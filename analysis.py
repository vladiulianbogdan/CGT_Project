import numpy as np 
import matplotlib.pyplot as plt 

# Here are some functions for reading the .dat file that is generated in simulation.py
# These functions were developed in TestBook notebook where I construct them interactively.
# I also intended that these functions are used interactivaly in the notebook and not from a script
# Because doing analysis is way easier in a notebook
# - lennart (2019)

def file_len(fname):
    """ Returns number of lines in the file
    """
    with open(fname) as f:
        for i, l in enumerate(f):
            pass
    return i + 1

def readContributionFileToData(fileName, heterogeneous, rounds, linesHeader=7, linesSummary=2, numGenerations=None):
    """ Read the contributions per generation of a file
        
        Attributes:
            - fileName(str): String of the file name/path
            - heterogeneous(bool): Tells if the data is heterogeneous
            - rounds(int): Number of rounds
            - linesHeader(int): Number of header lines of the file
            - linesSummary(int): Number of summary lines at the end of the file.
            - numGenerations(int): Number of generations. Default is None, then it is manually calculated based on the file length
        
        Returns a multidimensional numpy array with the average contribution level:
            - (NumGen,rounds) in the case of homogeneous
            - (NumGen,2,rounds) in the heterogeneous where dimension 1 has is the health status.
    """
    lenfile = file_len(fileName)
    
    if heterogeneous:
        # Not implemented yet because the format of the heterogenous case might change.
        print("heterogeneous not implemented yet")
        
    else: # Is homogeneous. In this case the contributions are just individual 
        
        if numGenerations is None: # If number of generations not specified construct it from the file length
            numGenerations = lenfile - linesHeader - linesSummary
            
        # init contribution
        contribution = np.empty(shape=(numGenerations,rounds))
        
        with open(fileName) as file:
            for i, line in enumerate(file):
                if i < linesHeader:
                    continue
                if i >= lenfile - linesSummary:
                    continue
                contribution[i-linesHeader] = np.fromstring(line,sep=" ")
                
        return contribution

def plotContributionVsGeneration(contributionArray):
    rounds = contributionArray.shape[-1]
    fig = plt.figure(figsize=(10,5),dpi=100)
    for r in range(0,rounds):
        plt.plot(contributionArray[:,r],label=f"round={r+1}")
    
    plt.xlabel("iteration",fontsize=15)
    plt.ylabel("Contribution",fontsize=15)
    plt.title("Average Contribution over time",fontsize=18)
    plt.legend()
    #plt.savefig("Contribution")
    return fig


if __name__ == "__main__":

	filename= "TestSimulation.dat"
	contributionArray=readContributionFileToData(filename,False,4)
	fig = plotContributionVsGeneration(contributionArray)

