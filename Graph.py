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





def graph_func():    
    # ax = ox.project_gdf(state).plot(fc='gray', ec='none')
    north, east, south, west = 44.711211, -63.40275, 44.391167, -63.722657

    # Downloading the map as a graph object
    # G = ox.graph_from_bbox(north, south, east, west,custom_filter=["power"])
    G = ox.graph_from_bbox(north, south, east, west,network_type='drive',simplify=True)
    G = ox.graph_from_place("Halifax, Nova Scotia, Canada",network_type='drive',simplify=True)


    # ox.plot_graph(G,node_size=4,dpi=150,figsize=(13,13))
    k = 0
    mapping = {}
    for i in G.nodes:
        mapping[i] = k
        k = k+1
    G = nx.relabel_nodes(G, mapping)

    node_coordinates=[]
    for node in G.nodes():
        x = G.nodes[node]['x']
        y = G.nodes[node]['y']
        node_coordinates.append((x, y))

    node_coordinates=np.array(node_coordinates)    

    return G, node_coordinates
