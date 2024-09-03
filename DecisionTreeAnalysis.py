import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

class DecisionTreeAnalysis:
    def __init__(self, features, target, max_depth=None, test_size=0.3, random_state=42):
        self.features = features
        self.target = target
        self.max_depth = max_depth
        self.test_size = test_size
        self.random_state = random_state
        self.model = DecisionTreeRegressor(max_depth=self.max_depth)
        self.feature_importances_ = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def train(self, df):
        X = df[self.features]
        y = df[self.target]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state
        )
        self.model.fit(self.X_train, self.y_train)
        self.feature_importances_ = self.model.feature_importances_

    def plot_tree(self, max_depth=None, feature_names=None, figsize=(20, 10)):
        y_pred_test = self.model.predict(self.X_test)
        r_squared_test = r2_score(self.y_test, y_pred_test)
        test_error = mean_squared_error(self.y_test, y_pred_test)

        plt.figure(figsize=figsize)
        plot_tree(self.model, 
                  max_depth=max_depth, 
                  feature_names=feature_names if feature_names else self.features, 
                  filled=True)
        plt.title(f"Decision Tree for {self.target} (Max Depth: {max_depth})\nTest R-squared: {r_squared_test:.4f}, Test Error: {test_error:.4f}")
        plt.show()

    def print_most_important_features(self, top_x=5):
        if self.feature_importances_ is None:
            print("Model has not been trained yet.")
            return
        
        sorted_indices = self.feature_importances_.argsort()[::-1]
        for i in range(min(top_x, len(self.features))):
            print(f"{i + 1}: {self.features[sorted_indices[i]]} (Importance: {self.feature_importances_[sorted_indices[i]]:.4f})")

    def save_top_important_features(self, top_x=11):
        if self.feature_importances_ is None:
            print("Model has not been trained yet.")
            return

        sorted_indices = self.feature_importances_.argsort()[::-1]
        top_features = [(self.features[sorted_indices[i]], self.feature_importances_[sorted_indices[i]]) 
                        for i in range(min(top_x, len(self.features)))]

        filename = f"top_{top_x}_features_max_depth_{self.max_depth}.txt"
        
        with open(filename, 'w') as f:
            for i, (feature, importance) in enumerate(top_features, start=1):
                f.write(f"{i}: {feature} (Importance: {importance:.4f})\n")
        
        print(f"Top {top_x} important features saved to {filename}")

    def train_and_plot_metrics(self, df, max_depth_x):
        X = df[self.features]
        y = df[self.target]

        r_squared_scores = []
        num_leaves = []
        depths = list(range(1, max_depth_x + 1))

        for depth in depths:
            self.max_depth = depth
            self.model = DecisionTreeRegressor(max_depth=self.max_depth)
            self.model.fit(self.X_train, self.y_train)
            self.feature_importances_ = self.model.feature_importances_

            y_pred_test = self.model.predict(self.X_test)
            r_squared_test = r2_score(self.y_test, y_pred_test)
            r_squared_scores.append(r_squared_test)
            num_leaves.append(self.model.get_n_leaves())

            print(f"Plotting tree for max_depth={depth}")
            self.plot_tree(max_depth=depth)

            self.save_top_important_features(top_x=11)

        # Plot R-squared scores and number of leaves
        plt.figure(figsize=(14, 7))
        plt.subplot(1, 2, 1)
        plt.plot(depths, r_squared_scores, marker='o')
        plt.title(f"Test R-squared vs. Max Depth for {self.target}")
        plt.xlabel("Max Depth")
        plt.ylabel("Test R-squared")

        # Plot number of leaves
        plt.subplot(1, 2, 2)
        plt.plot(depths, num_leaves, marker='o', color='red')
        plt.title(f"Number of Leaves vs. Max Depth for {self.target}")
        plt.xlabel("Max Depth")
        plt.ylabel("Number of Leaves")

        plt.tight_layout()
        plt.show()


class RandomForestAnalysis:
    def __init__(self, features, target, max_depth=None, n_estimators=100, test_size=0.3, random_state=42):
        self.features = features
        self.target = target
        self.max_depth = max_depth
        self.n_estimators = n_estimators
        self.test_size = test_size
        self.random_state = random_state
        self.model = RandomForestRegressor(max_depth=self.max_depth, n_estimators=self.n_estimators, random_state=self.random_state)
        self.feature_importances_ = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def train(self, df):
        X = df[self.features]
        y = df[self.target]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state
        )
        self.model.fit(self.X_train, self.y_train)
        self.feature_importances_ = self.model.feature_importances_

    def plot_tree(self, max_depth=None, feature_names=None, figsize=(20, 10)):
        y_pred_test = self.model.predict(self.X_test)
        r_squared_test = r2_score(self.y_test, y_pred_test)
        test_error = mean_squared_error(self.y_test, y_pred_test)

        # Plotting feature importances as the "tree"
        indices = np.argsort(self.feature_importances_)[::-1]
        plt.figure(figsize=figsize)
        plt.title(f"Random Forest for {self.target} (Max Depth: {max_depth})\nTest R-squared: {r_squared_test:.4f}, Test Error: {test_error:.4f}")
        plt.bar(range(len(self.feature_importances_)), self.feature_importances_[indices], align="center")
        plt.xticks(range(len(self.feature_importances_)), np.array(self.features)[indices], rotation=90)
        plt.xlim([-1, len(self.feature_importances_)])
        plt.show()

    def print_most_important_features(self, top_x=5):
        if self.feature_importances_ is None:
            print("Model has not been trained yet.")
            return
        
        sorted_indices = self.feature_importances_.argsort()[::-1]
        for i in range(min(top_x, len(self.features))):
            print(f"{i + 1}: {self.features[sorted_indices[i]]} (Importance: {self.feature_importances_[sorted_indices[i]]:.4f})")

    def save_top_important_features(self, top_x=11):
        if self.feature_importances_ is None:
            print("Model has not been trained yet.")
            return

        sorted_indices = self.feature_importances_.argsort()[::-1]
        top_features = [(self.features[sorted_indices[i]], self.feature_importances_[sorted_indices[i]]) 
                        for i in range(min(top_x, len(self.features)))]

        filename = f"top_{top_x}_features_max_depth_{self.max_depth}.txt"
        
        with open(filename, 'w') as f:
            for i, (feature, importance) in enumerate(top_features, start=1):
                f.write(f"{i}: {feature} (Importance: {importance:.4f})\n")
        
        print(f"Top {top_x} important features saved to {filename}")

    def train_and_plot_metrics(self, df, max_depth_x):
        X = df[self.features]
        y = df[self.target]

        r_squared_scores = []
        depths = list(range(1, max_depth_x + 1))

        for depth in depths:
            self.max_depth = depth
            self.model = RandomForestRegressor(max_depth=self.max_depth, n_estimators=self.n_estimators, random_state=self.random_state)
            self.model.fit(self.X_train, self.y_train)
            self.feature_importances_ = self.model.feature_importances_

            y_pred_test = self.model.predict(self.X_test)
            r_squared_test = r2_score(self.y_test, y_pred_test)
            r_squared_scores.append(r_squared_test)

            print(f"Plotting random forest for max_depth={depth}")
            self.plot_tree(max_depth=depth)

            self.save_top_important_features(top_x=11)

        # Plot R-squared scores
        plt.figure(figsize=(14, 7))
        plt.plot(depths, r_squared_scores, marker='o')
        plt.title(f"Test R-squared vs. Max Depth for {self.target}")
        plt.xlabel("Max Depth")
        plt.ylabel("Test R-squared")
        plt.tight_layout()
        plt.show()