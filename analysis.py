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

def readContributionFileToData(fileName, heterogeneous, rounds, numGenerations=None):
    """ Read the contributions per generation of a file
        
        Attributes:
            - fileName(str): String of the file name/path
            - heterogeneous(bool): Tells if the data is heterogeneous
            - rounds(int): Number of rounds
            - numGenerations(int): Number of generations. Default is None, then it is manually calculated based on the file length
        
        Returns a multidimensional numpy array with the average contribution level:
            - (NumGen,rounds) in the case of homogeneous
            - (NumGen,2,rounds) in the heterogeneous where dimension 1 has is the health status.
    """
    lenfile = file_len(fileName)
    
    if heterogeneous:
        
        linesHeader = 9
        linesSummary = 3
        
        if numGenerations is None:
            linesGenerations = lenfile - linesHeader -linesSummary
            numGenerations = int(linesGenerations/linesPerGeneration)
            
        contribution = np.empty(shape = (numGenerations,2,rounds))
        
        with open(fileName) as file:
            for i, line in enumerate(file):
                if i < linesHeader:
                    continue
                if i >= lenfile - linesSummary:
                    continue
                
                lineWealthType = (i-linesHeader)%linesPerGeneration # 0 = Rich, 1=Poor
                generation = (i-linesHeader)//linesPerGeneration
                
                if lineWealthType==0:# Rich
                    contribution[generation,0] = np.fromstring(line[2:],sep=" ") # remove the first two characters
                else: # Poor
                    contribution[generation,1] = np.fromstring(line[2:],sep=" ")
        
        
    else: # Is homogeneous. In this case the contributions are just individual
        
        linesHeader = 7
        linesSummary = 2
        
        if numGenerations is None:
            numGenerations = lenfile - linesHeader -linesSummary
        
        contribution = np.empty(shape=(numGenerations,rounds))
        
        with open(fileName) as file:
            for i, line in enumerate(file):
                if i < linesHeader:
                    continue
                if i >= lenfile - linesSummary:
                    continue
                contribution[i-linesHeader] = np.fromstring(line,sep=" ")
                
    return contribution

def readHeader(filename,heterogeneous):
    """ Extract header information return this in a dictionary"""
    # Start with empty dictionary
    Header = {}
    with open(filename) as file: # Go over each line one by one and add it to the dictionary
        Header["heterogeneous"] = bool(file.readline())
        Header["popSize"] = int(file.readline())
        if heterogeneous==False:
            Header["wealthRich"] = float(file.readline())
        else:
            Header["wealthRich"] = float(file.readline())
            Header["wealthPoor"] = float(file.readline())
        Header["numberOfRounds"] = int(file.readline())
        Header["numberOfRounds"] = int(file.readline())
        if heterogeneous==False:
            Header["alphaRich"] = float(file.readline())
        else: 
            Header["alphaRich"] = float(file.readline())
            Header["alphaPoor"] = float(file.readline())
        Header["lambda"] = float(file.readline())
    return Header

def readSummary(filename, heterogeneous):
    """ Read the summary of a file and return it in a dictionary"""
    lenFile = file_len(filename)
    if heterogeneous:
        lenSummary = 3
    else: # homogeneous
        lenSummary = 2
    
    summary = {}
    # Sadly I don't know an easy way to just start at the end of a file.
    countSummaryLines = 0 # Have a counter that keeps track which summary line we are evaluating
    with open(filename) as file:
        for i, line in enumerate(file):
            if i<(lenFile-lenSummary):
                continue
                
            if i==lenFile-1: # Remove last newline character from the list line else the np.fromstring doesn't work
                line=line[:-2]
            
            if countSummaryLines==0:
                summary["AverageContribution"] = float(line)
            elif countSummaryLines==1:
                summary["AverageContributionPerRoundRich"] = np.fromstring(line,sep=" ")
            elif heterogeneous and countSummaryLines==2:
                summary["AverageContributionPerRoundPoor"] = np.fromstring(line,sep=" ")
            
            countSummaryLines+=1
    
    return summary

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

