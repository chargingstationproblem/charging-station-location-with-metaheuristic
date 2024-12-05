
import random
import copy

import numpy as np



def decoding_func(vector, vector2,Budget):
    decoded_vector=copy.deepcopy(vector)
    decoded_vector2=copy.deepcopy(vector2)
    N = len(decoded_vector)
    r = max(10,int(vector[0] * N))
    decoded_vector[r + 1:] = 0
    decoded_vector[0] = r
    decoded_vector[1:] = (decoded_vector[1:] >= 0.5).astype(int)
    if sum(decoded_vector[1:r])==0:
        decoded_vector[np.random.randint(r-1)+1]=1
    total_B = (np.count_nonzero(decoded_vector) - 1)*100000
    count_of_ones= (np.count_nonzero(decoded_vector) - 1)
    max_allowable=int(np.floor(Budget/100000))
    if count_of_ones > max_allowable:
        count_of_ones= (np.count_nonzero(decoded_vector) - 1)
        indices_to_set_zero = np.random.choice(np.where(decoded_vector == 1)[0], size=count_of_ones - max_allowable, replace=False)
        decoded_vector[indices_to_set_zero] = 0
        decoded_vector2 = (decoded_vector == 1).astype(int)
    else:
        decoded_vector2 = (decoded_vector == 1).astype(int)
        extera_no_pile=np.floor((Budget-total_B)/10000)
        decoded_vector2 = generate_vector(extera_no_pile, vector2,decoded_vector2,decoded_vector,vector)    
        
    return decoded_vector, decoded_vector2


def generate_vector(extera_no_pile, vector2,decoded_vector2,decoded_vector,vector)  :

    aa=np.sum(vector2[decoded_vector2==1])
    bb=vector2[decoded_vector2==1]/aa
    cc=np.round(bb*extera_no_pile)
    cc[-1]=extera_no_pile-sum(cc[:-1])
    # Step 1: Create a 1*N vector filled with zeros
    vector3=[]
    vector4=copy.deepcopy(decoded_vector2)
    P=extera_no_pile
    while sum(vector3)<extera_no_pile:
        b=random.randint(1,P)
        vector3.append(b)
        P=P-b
    
    index_of_one =np.where(vector4 == 1)[0]
    bb=np.random.choice(index_of_one, size=len(cc), replace=False)
    vector4[index_of_one]=cc
    
    return vector4
