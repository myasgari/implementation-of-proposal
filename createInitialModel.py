import numpy as np
from sklearn.cluster import KMeans


class createInitialModel:
    def __init__(self, data_set , labels , NUMBER_OF_CLUSTER=12):
        self.NUMBER_OF_CLUSTER = NUMBER_OF_CLUSTER
        self.divided_sets = self.divide_dataset_by_labels(data_set , labels)

    def divide_dataset_by_labels(self, dataset, labels):
        unique_labels = np.unique(labels)
        divided_sets = {}

        # Initialize an empty list for each unique label
        for label in unique_labels:
            divided_sets[label] = []

        # Assign data points to respective label sets
        for i in range(len(dataset)):
            label = labels[i]
            divided_sets[label].append(dataset[i])

        return divided_sets

    def k_means_cluster(self, k=None):
        if k is None:
            k = self.NUMBER_OF_CLUSTER
        
        mc = {}
        for i in range(len(self.divided_sets)):
            mc[list(self.divided_sets.keys())[i]] = {}

        for label, data_points in self.divided_sets.items():
            kmeans = KMeans(init='k-means++', n_clusters=k, n_init=20)
            kmeans.fit(data_points)
            labels = kmeans.labels_
            data_points = np.array(data_points)

            for i in range(kmeans.n_clusters):
                mc[label][i] = {}
                cluster_points = data_points[labels == i]
                linear_sum = np.sum(cluster_points, axis=0)
                mc[label][i]['LS'] = linear_sum

                squared_sum = np.sum(np.square(cluster_points), axis=0)
                mc[label][i]['SS'] = squared_sum

                mc[label][i]['N'] = cluster_points.__len__()

                mc[label][i]['R'] = 1

                mc[label][i]['T'] = 0

                deviations = cluster_points - np.mean(cluster_points, axis=0)
                covariance_matrix = np.cov(deviations.T)
                mc[label][i]['PI'] = covariance_matrix

        return mc