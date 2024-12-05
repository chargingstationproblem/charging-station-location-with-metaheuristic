#Function to carry out the crossover
import numpy as np
from ypstruct import structure


def crossover(sol1,sol2):
    
    
    
    alpha=np.random.rand()

    new_sol1=sol1*alpha+sol2*(1-alpha)
    new_sol2=sol2*alpha+sol1*(1-alpha)
    

    
    return new_sol1, new_sol2  
    


