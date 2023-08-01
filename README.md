# Classification-DecisionTree_RandomForest-Explained

#### This is for informational purposes only as these algorithms lack features. Implementing these algorithms with real life problems can be complex and this model would not suffice. I recommend using libraries such as Sci-Kit Learn, PyTorch, Keras, or Tensorflow for more complete algorithms with more features. This is just an explanation of how these algorithms generally work.

#### This Classification decision tree is built in a similar way that my regression decision tree was built, the purpose of this explanation is to provide others with an explanation through code. 
Because this code is like the regression tree model, I will only be going over changes from the decision tree but not explaining the complete code again. A complete explanation of the code can be found here.

### Classification Tree
```
class DecisionNode:
    def __init__(self, feature_index=None, threshold=None, left=None, right=None, labels=None):
        self.feature_index = feature_index
        self.threshold = threshold
        self.labels = labels
        self.left = left
        self.right = right
```
#### We start with the first change being the change of variable names of leaf nodes from value to labels. 
```
    def _gini_index(self,y):
        # Calculate the Gini Impurity
        _, counts = np.unique(y, return_counts=True)
        probabilities = counts / np.sum(counts)
        gini = 1 - np.sum(probabilities ** 2)
        return gini
```
#### Our biggest change would be how we calculate our splitting criterion, instead of using variance/MSE as a measure we use the Gini impurity. Other splitting criteria are information gain, entropy, averaging adjacent values for Gini, etc. but these will not be explained here.

<p align="center">Gini Impurity</p>

```math
$$  1 - \sum_{i=1}^n \left( p_i \right)^2 $$
```

#### Gini is a measure of impurity of data, 
#### •	A Gini impurity of 0 is no impurity, all values belong to the same label.
#### •	A Gini impurity of 1 is perfect impurity, all values belong to a different label.
#### •	A Gini impurity of 0.5 means half the labels belong to one label and the others to another.


```
def _best_split(self, X, y):
        # Find the best split
        num_samples, num_features = X.shape
        best_feature, best_threshold = None, None
        best_gini = 1  # Best Gini impurity
```

#### We have a new variable “best_gini” and it is set to 1. Because the Gini impurity index is a measure of impurity in the data, we start by setting this to 1 to represent completely impure data as we search for the least impure data split at or near 0.

#### When we have a Gini impurity at or near 0 for a split, means that we have almost pure same labeled data and if it is a leaf node, we would have an accurate prediction.
```
for feature_index in range(num_features):
            thresholds = np.unique(X[:, feature_index])
            for threshold in thresholds:
                left_indices = np.where(X[:, feature_index] < threshold)
                right_indices = np.where(X[:, feature_index] >= threshold)

                left_gini = self._gini_index(y[left_indices])
                right_gini = self._gini_index(y[right_indices])
```

#### More changes made to the _best_split method is the calculation of left and right Gini using the new _gini_index method. However, this part of the code still functions in a similar way to the regression tree.
```
gini = (len(left_indices[0]) / num_samples) * left_gini + (len(right_indices[0]) / num_samples) * right_gini

                if gini < best_gini:
                    best_gini = gini
                    best_feature = feature_index
                    best_threshold = threshold

        return best_feature, best_threshold
```
#### After calculating the Gini index for both left and right splits we then calculate the weighted average Gini impurity as our actual Gini impurity of our split. We count the number of features in each split both left and right, divide each by the total number of total data points and multiply each by their respective Gini impurity, and now we have our weighted Gini impurity of the split.

### Random Forest
```
    def predict(self, X):
        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()
            
        tree_preds = np.array([[tree.predict(sample) for tree in self.trees] for sample in X])

        most_common = np.array([np.bincount(preds).argmax() for preds in tree_preds])

        return most_common
```
#### Our changes to our random forest algorithm are the new variable “tree_preds” which creates a list for each data sample input within a list. Now we have also added the variable “most_common” which counts the most repeated values (mode) and returns it.
