
from ypstruct import structure
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import random
import copy
import pandas as pd
import time
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd  # to load cities dataset
from geopy import distance  # to calculate distance on the surface
from timeit import default_timer as timer
from ypstruct import structure
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import Birch, KMeans
from sklearn.datasets import make_blobs
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.datasets import make_blobs
import numpy as np
import matplotlib.pyplot as plt
import osmnx as ox
#from pyclara.cluster import CLARA
from sklearn.mixture import GaussianMixture
import seaborn as sns
from sklearn.cluster import AffinityPropagation
from Graph import graph_func
from Decoding import decoding_func
from Clusturing import ml_clusture_func
from Cost import cost_function

from Solution_generation import gen_solution



from my_fast_sorting import fast_non_dominated_sort
from my_crowding_distance import crowding_distance
from my_crossover import crossover
from my_mutation import mutation
from my_sort_pop import sort_pop
# import more_itertools

G = ox.load_graphml(filepath="my_graph.graphml")
node_coordinates = np.load('node_coordinates.npy')



excel_data = pd.read_excel('last_mine.xlsx')



preference=np.random.uniform(low=0, high=0, size=(len(excel_data),len(G.nodes)))
from geopy.distance import geodesic
excel_data = pd.read_excel('last_mine.xlsx')
def calculate_distance(point1, point2):
    return geodesic(point1, point2).kilometers
# Iterate through all nodes in OSMnx graph
for osmnx_node, osmnx_data in G.nodes(data=True):
    osmnx_coordinates = (osmnx_data['y'], osmnx_data['x'])

    for index, excel_row in excel_data.iterrows():
        excel_coordinates = (excel_row['y'], excel_row['x'])
        distance2 = calculate_distance(osmnx_coordinates, excel_coordinates)

        if distance2 <= 3:
            G.nodes[osmnx_node]['preference'] = excel_row['preference']
            preference[index,osmnx_node]=excel_row['preference']+np.random.randint(excel_row['young_pop'])

final_preference=sum(preference)/(np.max(sum(preference)))
#############################################################################            

N=50

Budget=3000000






# Sphere Test Function

q_params = structure()
q_params.landa=15
q_params.mio=9
q_params.servers=2

# Problem Definition

N=50
# GA Parameters
params = structure()
params.maxit = 2
params.npop = 50
params.beta = 1
params.pc = 1
params.gamma = 0.1
params.mu = 0.01
params.sigma = 0.1
max_gen=2
pop_size=3
pc = 0.6
pm = 0.4
nc = int(np.round(pc*pop_size/2)*2)
nm = int(np.round(pm*pop_size/2)*2)

   # Empty Individual Template
empty_individual = structure()
empty_individual.vector = None
empty_individual.vector2 = None
empty_individual.decoded_vector = None
empty_individual.decoded_vector2 = None

empty_individual.cost = None

pop = empty_individual.repeat(pop_size)
function1_values=[]
function2_values=[]

for i in range(pop_size):
    pop[i].vector,pop[i].vector2 = gen_solution(N)
    pop[i].decoded_vector,pop[i].decoded_vector2=decoding_func(pop[i].vector,pop[i].vector2,Budget)

    cluster_centers,distances=ml_clusture_func(pop[i].decoded_vector,node_coordinates)
    
    f1,f2=cost_function(pop[i].decoded_vector,pop[i].decoded_vector2,distances,q_params,final_preference)
    function1_values.append(f1)      
    function2_values.append(f2)

    non_dominated_sorted_solution = fast_non_dominated_sort(function1_values[:],function2_values[:])       
    # function_values = [cost_function(graph_parameters,pop[i].sol) for i in range(0,pop_size)]
    crowding_distance_values=[]
    for i in range(0,len(non_dominated_sorted_solution)):
        crowding_distance_values.append(crowding_distance(function1_values[:],function2_values[:],non_dominated_sorted_solution[i][:]))
    
    

def sort_by_values(list1, values):
    sorted_list = []
    while(len(sorted_list)!=len(list1)):
        if index_of(min(values),values) in list1:
            sorted_list.append(index_of(min(values),values))
        values[index_of(min(values),values)] = math.inf
    return sorted_list

def index_of(a,list):
    for i in range(0,len(list)):
        if list[i] == a:
            return i
    return -1



for gen_no in range(max_gen):
    
    
# %%
    popc =empty_individual.repeat(nc)

    for i in range(0,nc,2):
        a1 = np.min(np.random.randint(pop_size-1,size=[1,5]))
        b1 = np.min(np.random.randint(pop_size-1,size=[1,5]))

        popc[i].vector,popc[i+1].vector  = crossover(pop[a1].vector,pop[b1].vector)
        popc[i].vector2,popc[i+1].vector2  = crossover(pop[a1].vector2,pop[b1].vector2)
        popc[i].decoded_vector,popc[i].decoded_vector2=decoding_func(popc[i].vector,popc[i].vector2,Budget)
        popc[i+1].decoded_vector,popc[i+1].decoded_vector2=decoding_func(popc[i+1].vector,popc[i+1].vector2,Budget)

        cluster_centers,distances=ml_clusture_func(popc[i].decoded_vector,node_coordinates)
    
        
    function1_values_popc=[]
    function2_values_popc=[]
    for i in range(0,nc):
        f1,f2=cost_function(popc[i].decoded_vector,popc[i].decoded_vector2,distances,q_params,final_preference)

        function1_values_popc.append(f1)
        function2_values_popc.append(f2)
    
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    
    popm =empty_individual.repeat(nm)

    for i in range(0,nm):
        a1 = np.min(np.random.randint(pop_size-1,size=[1,5]))


        popm[i].vector = mutation(pop[a1].vector)
        popm[i].vector2 = mutation(pop[a1].vector2)

        popm[i].decoded_vector,popm[i].decoded_vector2=decoding_func(popm[i].vector,popm[i].vector2,Budget)
        cluster_centers,distances=ml_clusture_func(popm[i].decoded_vector,node_coordinates)

        
    function1_values_popm=[]
    function2_values_popm=[]
    for i in range(0,nm):
        f1,f2=cost_function(popm[i].decoded_vector,popm[i].decoded_vector2,distances,q_params,final_preference)

        function1_values_popm.append(f1)
        function2_values_popm.append(f2)
  
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    

  
  
  
    
    
    for i in range(len(popc)):
      pop.append(popc[i])
      function1_values.append(function1_values_popc[i])
      function2_values.append(function2_values_popc[i])
    for i in range(len(popm)):
      pop.append(popm[i])
      function1_values.append(function1_values_popm[i])
      function2_values.append(function2_values_popm[i])
        
    
    non_dominated_sorted_solution = fast_non_dominated_sort(function1_values[:],function2_values[:])
   
    crowding_distance_values=[]
    for i in range(0,len(non_dominated_sorted_solution)):
        crowding_distance_values.append(crowding_distance(function1_values[:],function2_values[:],non_dominated_sorted_solution[i][:]))
    
    
    
    pop, function1_values, function2_values=sort_pop(non_dominated_sorted_solution,crowding_distance_values,pop,function1_values,function2_values) 
    pop=pop[0:pop_size]
    function1_values=function1_values[0:pop_size]
    function2_values=function2_values[0:pop_size]
    
    
    non_dominated_sorted_solution = fast_non_dominated_sort(function1_values[:],function2_values[:])
    function1 = [function1_values[i] for i in non_dominated_sorted_solution[0]]
    function2 = [function2_values[j] for j in non_dominated_sorted_solution[0]]
    plt.xlabel('Function 1', fontsize=15)
    plt.ylabel('Function 2', fontsize=15)
    plt.scatter(function1, function2,color='red')
    
    print("The best front for Generation number ",len(non_dominated_sorted_solution[0]), " is")
    
    


non_dominated_sorted_solution = fast_non_dominated_sort(function1_values[:],function2_values[:])
function1 = [function1_values[i] for i in non_dominated_sorted_solution[0]]
function2 = [function2_values[j] for j in non_dominated_sorted_solution[0]]
plt.xlabel('Function 1', fontsize=15)
plt.ylabel('Function 2', fontsize=15)
plt.scatter(function1, function2,color='red')

plt.show()    
    
  




# Assuming you already have G, colortype, unique_colortypes, colortype_colors, colortype_color_dict, node_colors defined


# Generate sample data
n_samples = len(node_coordinates)
n_features = 2
X=(node_coordinates)
#X, y = make_blobs(n_samples=n_samples, n_features=n_features, centers=3, random_state=42)

# K-Means clustering
n_clusters = 100
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
kmeans.fit(X)
kmeans_labels = kmeans.labels_
cluster_centers = kmeans.cluster_centers_

for node, label in zip(G.nodes(), kmeans_labels):
    G.nodes[node]['colortype'] = label
    
# Plot the results

fig, ax = plt.subplots(figsize=(30, 30))
fig.patch.set_facecolor('white')
ax.set_facecolor('white')
# Plot each cluster and its center





# Create a dictionary to map 'colortype' to unique colors
colortype_colors = {}
unique_colortypes = set()
for node in G.nodes():
    colortype = G.nodes[node]['colortype']
    unique_colortypes.add(colortype)

colortype_colors = sns.color_palette('Set3', n_colors=len(unique_colortypes))

# Create a dictionary to map 'colortype' to its corresponding color
colortype_color_dict = dict(zip(unique_colortypes, colortype_colors))
node_colors = [colortype_color_dict[G.nodes[node]['colortype']] for node in G.nodes()]

# Plot the graph using ox.plot_graph
fig, ax = ox.plot_graph(G, node_color=node_colors, node_size=30, edge_linewidth=0.5, figsize=(100, 100), show=False)


fig, ax = plt.subplots(figsize=(30, 30))
fig.patch.set_facecolor('white')
ax.set_facecolor('white')

# Plot the graph using ox.plot_graph
ox.plot_graph(G, node_color=node_colors, node_size=30, edge_linewidth=0.5, ax=ax, show=False)

# Plot each cluster center with a different symbol and larger size
for i in range(n_clusters):
    center_x, center_y = cluster_centers[i]
    ax.scatter(center_x, center_y, marker='*', s=300, color='red')

    # Plot a smaller circle around each cluster center
    circle_radius = 333  # Adjust the radius as needed
    circle = plt.Circle((center_x, center_y), circle_radius, color='red', fill=False)
    ax.add_patch(circle)

# Save or display the plot
plt.show()

