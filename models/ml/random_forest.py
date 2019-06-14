import numpy as np
from models.ml.decision_tree import DecisionTree
#https://www.youtube.com/watch?v=y6DmpG_PtN0&list=PLPOTBrypY74xS3WD0G_uzqPjCQfU6IRK-
'''
How to use?
x -> features
y -> labels
n_trees -> number of uncorrelated trees we ensemble to create the random forest
n_features -> the number of features to sample and pass onto each tree
sample_sz -> the number of rows randomly selected and passed onto each tree. This is usually equal to total number of rows
depth -> depth of each decision tree. Higher depth means more number of splits which increases the over fitting tendency of each tree
min_leaf -> minimum number of rows required in a node to cause further split. Lower the min_leaf, higher the depth of the tree.
'''

class RandomForest():
    def __init__(self, x, y, n_trees, n_features, sample_size=None, depth=10, min_leaf=5):
        np.random.seed(12)
        if not sample_size:
            sample_size = x.shape[0]
        if n_features == 'sqrt':
            self.n_features = int(np.sqrt(x.shape[1]))
        elif n_features == 'log2':
            self.n_features = int(np.log2(x.shape[1]))
        else:
            self.n_features = n_features
        print(self.n_features, "sha: ", x.shape[1])
        self.x, self.y, self.sample_sz, self.depth, self.min_leaf = x, y, sample_size, depth, min_leaf
        self.trees = [self.create_tree() for i in range(n_trees)]

    def create_tree(self):
        idxs = np.random.permutation(len(self.y))[:self.sample_sz]
        f_idxs = np.random.permutation(self.x.shape[1])[:self.n_features]
        return DecisionTree(self.x.iloc[idxs], self.y[idxs], self.n_features, f_idxs,
                            idxs=np.array(range(self.sample_sz)), depth=self.depth, min_leaf=self.min_leaf)

    def predict(self, x):
        return np.mean([t.predict(x) for t in self.trees], axis=0)