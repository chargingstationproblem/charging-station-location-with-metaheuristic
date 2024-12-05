


def sort_pop(non_dominated_sorted_solution,crowding_distance_values,pop,function1_values,function2_values):
    
    rank=list(range(len(non_dominated_sorted_solution)))
  

# Flatten the list while preserving the order
    newlist = []
    for sublist in non_dominated_sorted_solution:
       newlist.extend(sublist)

    
   
    
    pop2=[]
    function1_values2=[]
    function2_values2=[]
    for i in range(len(newlist)):
       pop2.append(pop[newlist[i]])
       function1_values2.append(function1_values[newlist[i]])
       function2_values2.append(function2_values[newlist[i]])
    
    
    return pop2, function1_values2, function2_values2



