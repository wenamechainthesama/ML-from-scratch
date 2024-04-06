import numpy as np


class HMM:
    def __init__(self, transition, emission):
        self.transition = transition
        self.emission = emission


transition = np.ndarray([[0.5, 0.3, 0.2], [0.4, 0.2, 0.4], [0.0, 0.3, 0.7]])
emission = np.ndarray([[0.9, 0.1], [0.6, 0.4], [0.2, 0.8]])
hmm = HMM(transition, emission)
hmm.predict([0, 0, 1])
