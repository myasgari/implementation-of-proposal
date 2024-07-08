import numpy as np
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score
from datetime import datetime
from createInitialModel import createInitialModel
from semiSuperviseClassify import semiSuperviseClassify

dataset, labels = make_classification(
    n_samples=1000,
    n_features=5,
    n_classes=2,
    random_state=42
)

model = createInitialModel(dataset, labels)
classify = semiSuperviseClassify()

mc = model.k_means_cluster()
#print(mc)
buffer_size = 500
buffer = []
novel_list = []
p_labels = []

dataset, labels = make_classification(
    n_samples=1000,
    n_features=5,
    n_classes=2,
    random_state=45
)




for k in range(len(labels)) :

    p_label , use_mcs  = classify.classify(mc , dataset[k] , labels[k])
    p_labels.append(p_label)
    for i in range (len(use_mcs)-1) :
        key = use_mcs[i][0]
        label = use_mcs[i][1]

        if key == labels[k] :
            mc[key][label]['R'] += 1
            now = datetime.now()
            current_time = now.strftime("%H:%M:%S")
            mc[key][label]['T'] = current_time
        else :
            mc[key][label]['R'] -= 1
    key = use_mcs[0][0]
    label = use_mcs[0][1]
    if use_mcs[9] > classify.calculate_distance(mc[key][label]['LS']/mc[key][label]['N'],np.sqrt((((mc[key][label]['N'] * mc[key][label]['SS']) - np.square(mc[key][label]['LS'])) / np.square(mc[key][label]['N'])))) :
        if len(buffer) < buffer_size :
            buffer.append(dataset[k])
    


print(len(buffer))
print(len(buffer[0]))
accuracy = accuracy_score(labels , p_labels)


print(f"Accuracy of classifier: {accuracy:.2f}")