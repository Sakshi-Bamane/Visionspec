import numpy as np

def compute_class_weights(generator):
    labels = generator.classes
    class_counts = np.bincount(labels)
    total = sum(class_counts)

    class_weights = {
        0: total / (2 * class_counts[0]),
        1: total / (2 * class_counts[1])
    }
    return class_weights
