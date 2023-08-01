import numpy as np

class DecisionNode:
    def __init__(self, feature_index=None, threshold=None, left=None, right=None, value=None):
        self.feature_index = feature_index
        self.threshold = threshold
        self.value = value  # Change "labels" to "value"
        self.left = left
        self.right = right


class DecisionTreeClassifier:
    def __init__(self, min_samples_split=2, max_depth=2):
        self.root = None
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth

    def fit(self, X, y, node=None, depth=0):
        if node is None:
            node = DecisionNode()
            self.root = node

        if len(y) < self.min_samples_split or depth == self.max_depth:
            node.value = max(y, key=list(y).count) if y.size else None
            return

        feature_index, threshold = self._best_split(X, y)

        if feature_index is None or threshold is None:

            node.feature_index = feature_index
            node.threshold = threshold

            node.left = DecisionNode()
            node.right = DecisionNode()

            left_indices = np.where(X[:, feature_index] < threshold)
            right_indices = np.where(X[:, feature_index] >= threshold)

            self.fit(X[left_indices], y[left_indices], node.left, depth + 1)
            self.fit(X[right_indices], y[right_indices], node.right, depth + 1)
        else:
            node.value = max(y, key=list(y).count)

    def _best_split(self, X, y):
        num_samples, num_features = X.shape
        best_feature, best_threshold = None, None
        best_gini = 1


        for feature_index in range(num_features):
            thresholds = np.unique(X[:, feature_index])
            for threshold in thresholds:
                left_indices = np.where(X[:, feature_index] < threshold)
                right_indices = np.where(X[:, feature_index] >= threshold)

                left_gini = self._gini_index(y[left_indices])
                right_gini = self._gini_index(y[right_indices])

                gini = (len(left_indices[0]) / num_samples) * left_gini + (len(right_indices[0]) / num_samples) * right_gini

                if gini < best_gini:
                    best_gini = gini
                    best_feature = feature_index
                    best_threshold = threshold

        return best_feature, best_threshold

    def _gini_index(self, y):
        _, counts = np.unique(y, return_counts=True)
        probabilities = counts / np.sum(counts)
        gini = 1 - np.sum(probabilities ** 2)
        return gini

    def predict(self, X, node=None):
        if node is None:
            node = self.root

        if node.value is not None:
            return node.value

        if X[node.feature_index] < node.threshold:
            return self.predict(X, node.left)
        else:
            return self.predict(X, node.right)


import numpy as np

class DictVectorizer:
    def __init__(self):
        self.feature_indices_ = {}
        self.num_features_ = 0
        self.categories_ = {}

    def fit(self, data):
        for dictionary in data:
            for feature, value in dictionary.items():
                if feature not in self.feature_indices_:
                    self.feature_indices_[feature] = self.num_features_
                    self.num_features_ += 1
                    
                if isinstance(value, str):
                    if feature not in self.categories_:
                        self.categories_[feature] = {}
                    if value not in self.categories_[feature]:
                        self.categories_[feature][value] = len(self.categories_[feature])
                    
    def transform(self, data):
        num_samples = len(data)
        X = np.zeros((num_samples, self.num_features_))
        
        for i, dictionary in enumerate(data):
            for feature, value in dictionary.items():
                if feature in self.feature_indices_:
                    if isinstance(value, str):
                        value = self.categories_[feature].get(value, -1)
                    X[i, self.feature_indices_[feature]] = value
                    
        return X

# Usage
data = [
        {'Weather': 'Cloudy', 'Temperature': 21, 'Go Outside': 'Yes'},
        {'Weather': 'Sunny', 'Temperature': 24, 'Go Outside': 'Yes'},
        {'Weather': 'Sunny', 'Temperature': 26, 'Go Outside': 'Yes'},
        {'Weather': 'Rainy', 'Temperature': 19, 'Go Outside': 'No'},
        {'Weather': 'Cloudy', 'Temperature': 22, 'Go Outside': 'Yes'},
        {'Weather': 'Rainy', 'Temperature': 18, 'Go Outside': 'No'},
        {'Weather': 'Sunny', 'Temperature': 27, 'Go Outside': 'Yes'},
        {'Weather': 'Cloudy', 'Temperature': 21, 'Go Outside': 'Yes'},
        {'Weather': 'Rainy', 'Temperature': 20, 'Go Outside': 'No'},
        {'Weather': 'Sunny', 'Temperature': 25, 'Go Outside': 'Yes'}
        ]


vectorizer = DictVectorizer()
vectorizer.fit(data)
X = vectorizer.transform(data)

import numpy as np

class RandomForestClassifier:
    def __init__(self, n_estimators=100, min_samples_split=2, max_depth=2):
        self.n_estimators = n_estimators
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.trees = []

    def bootstrap_sample(self, X, y):
        n_samples = X.shape[0]
        idxs = np.random.choice(n_samples, size=n_samples, replace=True)
        return X[idxs], y[idxs]

    def fit(self, X, y):
        # Convert pandas DataFrame to numpy array
        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()
        if isinstance(y, pd.Series):
            y = y.to_numpy()

        self.trees = []
        for _ in range(self.n_estimators):
            tree = DecisionTreeClassifier(min_samples_split=self.min_samples_split, max_depth=self.max_depth)
            X_sample, y_sample = self.bootstrap_sample(X, y)
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)

    def predict(self, X):
        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()
            
        tree_preds = np.array([[tree.predict(sample) for tree in self.trees] for sample in X])
        print ('tree preds',tree_preds)
        most_common = np.array([np.bincount(preds).argmax() for preds in tree_preds])
        print ('most common', most_common)

        return most_common


import numpy as np

# We'll use a dictionary to map the categorical weather data to numbers
weather_mapping = {'Cloudy': 0, 'Sunny': 1, 'Rainy': 2}

# Now, let's create our data arrays
X = np.array([
    [weather_mapping['Cloudy'], 21],
    [weather_mapping['Sunny'], 24],
    [weather_mapping['Sunny'], 26],
    [weather_mapping['Rainy'], 19],
    [weather_mapping['Cloudy'], 22],
    [weather_mapping['Rainy'], 18],
    [weather_mapping['Sunny'], 27],
    [weather_mapping['Cloudy'], 21],
    [weather_mapping['Rainy'], 20],
    [weather_mapping['Sunny'], 25]
])

# We'll encode 'No' as 0 and 'Yes' as 1
y = np.array([1, 1, 1, 0, 1, 0, 1, 1, 0, 1])

# Let's split the data into training and test sets
train_ratio = 0.8
train_size = int(train_ratio * X.shape[0])

X_train = X[:train_size]
y_train = y[:train_size]

X_test = X[train_size:]
y_test = y[train_size:]

# Now, we can use our data with the RandomForestClassifier
clf = RandomForestClassifier(n_estimators=100, min_samples_split=2, max_depth=2)
clf.fit(X_train, y_train)

# We can then make predictions on new data
X_new = np.array([[weather_mapping['Rainy'], 19], [weather_mapping['Rainy'], 20]])
predictions = clf.predict(X_new)

print(predictions)
