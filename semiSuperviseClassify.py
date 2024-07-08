import numpy as np
from collections import Counter

class semiSuperviseClassify:
    def __init__(self, NUMBER_OF_CLUSTER = 12):
        self.NUMBER_OF_CLUSTER = NUMBER_OF_CLUSTER
        self.numberOfInstance = 0
        self.A = []
        self.B = []
        self.weights = [0.02,0.03,0.04,0.05]

    def calculate_distance(self, instance, centroid):
        return np.linalg.norm(instance - centroid)

    def predict(self, mc, instance):
        distances = []
        labels = []
        mcs =[]

        for key, clusters in mc.items():
            for i in range(self.NUMBER_OF_CLUSTER):
                distances.append(self.calculate_distance(instance, clusters[i]['LS'] / clusters[i]['N']))
                labels.append(key)
                mcs.append([key , i])
        nearest_cluster_indices = np.argsort(distances)[:9]
        nearest_cluster_labels = [labels[i] for i in nearest_cluster_indices]
        use_mcs = [mcs[i] for i in nearest_cluster_indices]
        use_mcs.append(distances[nearest_cluster_indices[0]])
        return nearest_cluster_labels , use_mcs
    
    def updateWeights(self, predictionLabel, actualLabel):
        self.weights = [0.05] * len(predictionLabel[0])
        k = 0
        m = len(predictionLabel)

        for i in range(len(predictionLabel[0])):
            for j in range(len(predictionLabel)):
                if predictionLabel[j][i] == actualLabel[i]:
                    self.weights[k] += 1 / m
            k += 1
        return self.weights

    def classify(self, mc , instance , label) :
        nearestNeighbors = []
        nearest_cluster_labels , use_mcs = self.predict(mc,instance)
        for j in (3,5,7,9) :
            counterKNN = Counter(nearest_cluster_labels[0:j])
            most_common = counterKNN.most_common(1)
            most_repeated_element, count = most_common[0]
            nearestNeighbors.append(most_repeated_element)

        self.numberOfInstance +=1
        self.A.append(nearestNeighbors)
        self.B.append(label)

        if (self.numberOfInstance == 10) :
            self.numberOfInstance = 0
            self.weights = self.updateWeights(self.A , self.B)
            self.A = []
            self.B = []
        
    #print(weights)

        return nearestNeighbors[self.weights.index(max(self.weights))] , use_mcs

        