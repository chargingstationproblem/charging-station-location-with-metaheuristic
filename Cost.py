import numpy as np
from ypstruct import structure




def cost_function(decoded_vector, decoded_vector2,distances,q_params,final_preference):

# Queueing System Information

    row=np.shape(distances)[0]
    
    final_preference = np.tile(final_preference, (row, 1))

    decoded_vector=decoded_vector[1:row+1].reshape((row, 1))
    utility=np.sum(np.exp(-(distances*decoded_vector))*final_preference)
    landa = q_params.landa
    mio = q_params.mio
    servers = q_params.servers
    # ca = q_params.ca
    # cs = q_params.cs

    ro = landa/(mio*servers)
    mio2=0.1

    # Probability of No EV with at CS
    P0 = 0
    for m in range(int(servers)):
    
            
        P0 = P0+(((servers*ro)**servers)/((np.math.factorial(servers))*(1-ro)))


    # Waitting time for an M/M/C_(k,m) queueing system
        P0 = 1/P0

    WT_M_M_C=(P0*ro*((landa/mio)**servers))/((np.math.factorial(servers))*landa*((1-ro)**2))
    time=((mio2*utility))
  




    return utility,time