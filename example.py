'''
This repository is for the many-to-one problem.
'''

import numpy as np
import pandas as pd

from linkage import single_linkage, complete_linkage, average_linkage, centroid_linkage


#Generate random data
# np.random.seed(0) #Set random seed for reproducability
# data = pd.DataFrame(np.random.rand(10, 5))
# data['cluster'] = [1, 1, 1, 2, 2, 3, 3, 3, 3, 3]

# data = pd.DataFrame(np.random.rand(5, 2))
data = pd.DataFrame(np.array([[1, 2], [1, 4], [1, 0], [4, 3], [4, 4]]))
data['cluster'] = [1, 1, 3, 2, 4]

# #print distances (rounded to 2 decimal places) between all points
# print("Distances between all points:")
# print(np.round(np.sqrt(np.square(data.values[:, :-1].reshape(data.shape[0], 1, data.shape[1] - 1) - data.values[:, :-1].reshape(1, data.shape[0], data.shape[1] - 1)).sum(axis = 2)), 2))

print("Initial Cluster Assignments:")
print(data)

#Perform combining through linkage strategy
data = single_linkage(data = data, final_clusters=3)

#Print the data
print("Final Cluster Assignments:")
print(data)