import pandas as pd
import numpy as np

def _perform_checks_(current_no_of_clusters, final_no_of_clusters):
    '''
    Perform checks on the number of clusters.
    
    Parameters:
    current_no_of_clusters (int): The current number of clusters.
    final_no_of_clusters (int): The final number of clusters.

    Returns:
    bool: True if the checks are passed, False otherwise.
    '''


    #number or clusters
    if (current_no_of_clusters < final_no_of_clusters):
        print("Error: The current number of clusters is less than the final number of clusters.")
        print("Returning the data as it is.")
        return False
    elif (current_no_of_clusters == final_no_of_clusters):
        print("Error: The current number of clusters is equal to the final number of clusters.")
        print("No further clustering required. Returning the data as it is.")
        return False
    else:
        pass
        
    
    if (final_no_of_clusters < 1):
        print("Error: The final number of clusters is less than 1.")
        print("Returning the data as it is.")
        return False
    


    return True


def single_linkage(data, final_clusters):
    '''
    Single linkage clustering algorithm.
    The function will combine the nearest data clusters based on the minimum euclidian distance between the clusters, 
    calculated as the distance between the closest points.
    The function will return the updated clustered data.
    
    Parameters:
    data (pandas dataframe): The data to be clustered. Last column is the cluster labels.
    final_clusters (int): The number of clusters to be formed.

    Returns:
    data (pandas dataframe): The updated clustered data.
    '''

    #CHECK: if the last column does not contain int values as cluster labels, then return the data as it is and raise an error
    if (data.iloc[:, -1].dtype != 'int64'):
        print("Error: The last column does not contain int values as cluster labels.")
        print("Returning the data as it is.")
        return data
    
    #CHECK: if the cluster labels are not in the range that starts from 1 then return the data as it is and raise an error
    if (np.min(data.iloc[:, -1]) < 1):
        print("Error: The cluster labels must start from 1.")
        print("Returning the data as it is.")
        return data


    # Calculate the current number of clusters
    clusters = np.unique(data.iloc[:, -1])
    n_clusters = len(clusters)

    #CHECK: For the validity of the number of clusters
    if not(_perform_checks_(n_clusters, final_clusters)):
        return data
    

    """Combine the two clusters having the closest nearest points (in terms of Euclidian distance).
    Continue combining clusters until the number of clusters is equal to the final number of clusters"""
    while (n_clusters > final_clusters):
        # Find the closest points between the clusters
        min_dist = np.inf #initialize the minimum distance to infinity
        for i in range(n_clusters):
            for j in range(i+1, n_clusters):
                # Find the minimum distance between the clusters
                dist = np.min(np.linalg.norm(data[data.iloc[:, -1] == clusters[i]].iloc[:, :-1].values[:, np.newaxis] - data[data.iloc[:, -1] == clusters[j]].iloc[:, :-1].values, axis=2))
                if (dist < min_dist):
                    min_dist = dist
                    cluster1 = clusters[i]
                    cluster2 = clusters[j]

        # Combine the two clusters having the closest nearest points
        data.iloc[data.iloc[:, -1] == cluster2, -1] = cluster1

        # Remove the second cluster from the list of clusters
        clusters = np.unique(data.iloc[:, -1])
        n_clusters = len(clusters)

    
    #Renaming the clusters from 1 to n_clusters
    for i in range(n_clusters):
        data.iloc[data.iloc[:, -1] == clusters[i], -1] = i+1

    
    return data


def complete_linkage(data, final_clusters):
    '''
    Complete linkage clustering algorithm.
    The function will combine the nearest data clusters based on the maximum euclidian distance between the clusters, 
    calculated as the distance between the farthest points.
    The function will return the updated clustered data.
    
    Parameters:
    data (pandas dataframe): The data to be clustered. Last column is the cluster labels.
    final_clusters (int): The number of clusters to be formed.

    Returns:
    data (pandas dataframe): The updated clustered data.
    '''

    #CHECK: if the last column does not contain int values as cluster labels, then return the data as it is and raise an error
    if (data.iloc[:, -1].dtype != 'int64'):
        print("Error: The last column does not contain int values as cluster labels.")
        print("Returning the data as it is.")
        return data
    
    #CHECK: if the cluster labels are not in the range that starts from 1 then return the data as it is and raise an error
    if (np.min(data.iloc[:, -1]) < 1):
        print("Error: The cluster labels must start from 1.")
        print("Returning the data as it is.")
        return data


    # Calculate the current number of clusters
    clusters = np.unique(data.iloc[:, -1])
    n_clusters = len(clusters)

    #CHECK: For the validity of the number of clusters
    if not(_perform_checks_(n_clusters, final_clusters)):
        return data
    

    """Combine the two clusters having the farthest nearest points (in terms of Euclidian distance).
    Continue combining clusters until the number of clusters is equal to the final number of clusters"""
    while (n_clusters > final_clusters):
        # Find the farthest points between the clusters
        max_dist = -np.inf #initialize the maximum distance to -infinity
        for i in range(n_clusters):
            for j in range(i+1, n_clusters):
                # Find the maximum distance between the clusters
                dist = np.max(np.linalg.norm(data[data.iloc[:, -1] == clusters[i]].iloc[:, :-1].values[:, np.newaxis] - data[data.iloc[:, -1] == clusters[j]].iloc[:, :-1].values, axis=2))
                if (dist > max_dist):
                    max_dist = dist
                    cluster1 = clusters[i]
                    cluster2 = clusters[j]

        # Combine the two clusters having the farthest nearest points
        data.iloc[data.iloc[:, -1] == cluster2, -1] = cluster1
        
        # Remove the second cluster from the list of clusters
        clusters = np.unique(data.iloc[:, -1])
        n_clusters = len(clusters)


    #Renaming the clusters from 1 to n_clusters
    for i in range(n_clusters):
        data.iloc[data.iloc[:, -1] == clusters[i], -1] = i+1


    return data


def average_linkage(data, final_clusters):
    '''
    Average linkage clustering algorithm.
    The function will combine the nearest data clusters based on the average euclidian distance between the clusters, 
    calculated as the average distance between all the points.
    The function will return the updated clustered data.
    
    Parameters:
    data (pandas dataframe): The data to be clustered. Last column is the cluster labels.
    final_clusters (int): The number of clusters to be formed.

    Returns:
    data (pandas dataframe): The updated clustered data.
    '''

    #CHECK: if the last column does not contain int values as cluster labels, then return the data as it is and raise an error
    if (data.iloc[:, -1].dtype != 'int64'):
        print("Error: The last column does not contain int values as cluster labels.")
        print("Returning the data as it is.")
        return data
    
    #CHECK: if the cluster labels are not in the range that starts from 1 then return the data as it is and raise an error
    if (np.min(data.iloc[:, -1]) < 1):
        print("Error: The cluster labels must start from 1.")
        print("Returning the data as it is.")
        return data


    # Calculate the current number of clusters
    clusters = np.unique(data.iloc[:, -1])
    n_clusters = len(clusters)

    #CHECK: For the validity of the number of clusters
    if not(_perform_checks_(n_clusters, final_clusters)):
        return data
    

    """Combine the two clusters having the smallest average distance between the points.
    Continue combining clusters until the number of clusters is equal to the final number of clusters"""
    while (n_clusters > final_clusters):
        # Find the average distance between the points of the clusters
        min_dist = np.inf
        for i in range(n_clusters):
            for j in range(i+1, n_clusters):
                # Find the average distance between the points of the clusters
                dist = np.mean(np.linalg.norm(data[data.iloc[:, -1] == clusters[i]].iloc[:, :-1].values[:, np.newaxis] - data[data.iloc[:, -1] == clusters[j]].iloc[:, :-1].values, axis=2))
                if (dist < min_dist):
                    min_dist = dist
                    cluster1 = clusters[i]
                    cluster2 = clusters[j]

        # Combine the two clusters having the smallest average distance between the points
        data.iloc[data.iloc[:, -1] == cluster2, -1] = cluster1

        # Remove the second cluster from the list of clusters
        clusters = np.unique(data.iloc[:, -1])
        n_clusters = len(clusters)


    #Renaming the clusters from 1 to n_clusters
    for i in range(n_clusters):
        data.iloc[data.iloc[:, -1] == clusters[i], -1] = i+1


    return data


def centroid_linkage(data, final_clusters):
    '''
    Centroid linkage clustering algorithm.
    The function will combine the nearest data clusters based on the average euclidian distance between the clusters, 
    calculated as the distance between the centroids of the clusters.
    The function will return the updated clustered data.
    
    Parameters:
    data (pandas dataframe): The data to be clustered. Last column is the cluster labels.
    final_clusters (int): The number of clusters to be formed.

    Returns:
    data (pandas dataframe): The updated clustered data.
    '''

    #CHECK: if the last column does not contain int values as cluster labels, then return the data as it is and raise an error
    if (data.iloc[:, -1].dtype != 'int64'):
        print("Error: The last column does not contain int values as cluster labels.")
        print("Returning the data as it is.")
        return data
    
    #CHECK: if the cluster labels are not in the range that starts from 1 then return the data as it is and raise an error
    if (np.min(data.iloc[:, -1]) < 1):
        print("Error: The cluster labels must start from 1.")
        print("Returning the data as it is.")
        return data


    # Calculate the current number of clusters
    clusters = np.unique(data.iloc[:, -1])
    n_clusters = len(clusters)

    #CHECK: For the validity of the number of clusters
    if not(_perform_checks_(n_clusters, final_clusters)):
        return data
    

    """Combine the two clusters having the closest nearest points (in terms of Euclidian distance).
    Continue combining clusters until the number of clusters is equal to the final number of clusters"""
    while (n_clusters > final_clusters):
        # Find the closest points between the clusters
        min_dist = np.inf #initialize the minimum distance to infinity
        for i in range(n_clusters):
            for j in range(i+1, n_clusters):
                # Find the distance between the centroids of the clusters
                dist = np.linalg.norm(np.mean(data[data.iloc[:, -1] == clusters[i]].iloc[:, :-1].values, axis=0) - np.mean(data[data.iloc[:, -1] == clusters[j]].iloc[:, :-1].values, axis=0))
                if (dist < min_dist):
                    min_dist = dist
                    cluster1 = clusters[i]
                    cluster2 = clusters[j]

        # Combine the two clusters having
        data.iloc[data.iloc[:, -1] == cluster2, -1] = cluster1
        
        # Remove the second cluster from the list of clusters
        clusters = np.unique(data.iloc[:, -1])
        n_clusters = len(clusters)


    #Renaming the clusters from 1 to n_clusters
    for i in range(n_clusters):
        data.iloc[data.iloc[:, -1] == clusters[i], -1] = i+1


    return data
