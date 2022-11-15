import pandas as pd
import numpy as np
import math
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier

class Tree:
    def __init__(self, val, output, depth):
        self.val = val
        self.children = {}
        self.output = output
        self.depth = depth

    def add_child(self, selected_feature, node):
        self.children[selected_feature] = node


class Decision_tree_classification:
    def __init__(self):
        self.__root = None

    def get_freq(self, n):
        data_split = {}
        for i in n:
            if i in data_split:
                data_split[i] += 1
            else:
                data_split[i] = 1
        return data_split

    def get_entropy(self, n):
        get_items_freq = self.get_freq(n)
        entropy = 0
        total = len(n)
        for i in get_items_freq:
            p = get_items_freq[i] / total
            entropy += p * math.log2(p)
        return -1 * entropy

    def get_split_info(self, x, feature_index):
        values = set(x[:, feature_index])
        total_size = np.shape(x)[0]
        d = {}
        split_info = 0
        for i in range(x.shape[0]):
            if x[i][feature_index] not in d:
                d[x[i][feature_index]] = 1
            else:
                d[x[i][feature_index]] += 1
        for i in values:
            split_info += (d[i] / total_size) * math.log2(d[i] / total_size)
        return (-1) * split_info

    def get_information_gain(self, x, y, feature_index):
        tot_info = self.get_entropy(y)
        values = set(x[:, feature_index])
        total_size = np.shape(x)[0]
        data_split = {}
        curr_info = 0
        df = pd.DataFrame(x)
        df[df.shape[1]] = y
        for i in range(x.shape[0]):
            if x[i][feature_index] not in data_split:
                data_split[x[i][feature_index]] = 1
            else:
                data_split[x[i][feature_index]] += 1
        for i in values:
            df1 = df[df[feature_index] == i]
            curr_info += (data_split[i] / total_size) * self.get_entropy(df1[df1.shape[1] - 1])
        return tot_info - curr_info

    def get_gain_ratio(self, x, y, feature_index):
        info_gain = self.get_information_gain(x, y, feature_index)
        split_ratio = self.get_split_info(x, feature_index)
        if split_ratio == 0:
            return math.inf
        else:
            return float(info_gain / split_ratio)

    def get_gini_index(self, n):
        get_items_freq = self.get_freq(n)
        gini = 0
        total = len(n)
        for i in get_items_freq:
            p = get_items_freq[i] / total
            gini += p ** 2
        return 1 - gini

    def get_gini_max(self, x, y, feature_index):
        total_gini_value = self.get_gini_index(y)
        values = set(x[:, feature_index])
        total_size = np.shape(x)[0]
        data_split = {}
        curr_gini = 0
        df = pd.DataFrame(x)
        df[df.shape[1]] = y
        for i in range(x.shape[0]):
            if x[i][feature_index] not in data_split:
                data_split[x[i][feature_index]] = 1
            else:
                data_split[x[i][feature_index]] += 1
        for i in values:
            df1 = df[df[feature_index] == i]
            curr_gini += (data_split[i] / total_size) * self.get_gini_index(df1[df1.shape[1] - 1])
        return total_gini_value - curr_gini

    def get_best_selection_attribute(self, d, y, type, feature_list):
        max_value = -math.inf
        best_feature = None
        if type == "gini":
            for i in feature_list:
                curr_gain = self.get_gini_max(d, y, i)
                if curr_gain > max_value:
                    best_feature = i
                    max_value = curr_gain
        elif type == "gain":
            for i in feature_list:
                curr_gain = self.get_gain_ratio(d, y, i)
                if curr_gain > max_value:
                    best_feature = i
                    max_value = curr_gain
        else:
            for i in feature_list:
                curr_gain = self.get_information_gain(d, y, i)
                if curr_gain > max_value:
                    best_feature = i
                    max_value = curr_gain
        return best_feature

    def decision_tree(self, d, out_list, metric_type, features_list, tree_depth, classes):
        # If the node consists of only one class.
        if len(set(out_list)) == 1:
            # print("Reached Leaf node with decision Tree depth = ", tree_depth)
            output = out_list[0]
            # print("Count of ", out_list[0], "=", len(out_list))
            # if metric_type == "gain":
                # print("Current Entropy = 0")
            # else:
                # print("Current gini index = 0")
            # print()
            return Tree(None, output, tree_depth)

        # If there are no more features left to classify
        elif len(features_list) == 0:
            # print("Reached Leaf node with decision Tree depth = ", tree_depth)
            get_items_freq = self.get_freq(out_list)
            curr_count = -math.inf
            output = None
            for i in classes:
                if i in get_items_freq:
                    frequency = get_items_freq[i]
                    if frequency > curr_count:
                        output = i
                        curr_count = frequency
                    # print("Count of ", i, "=", frequency)
            # if metric_type == "gain":
            #     print("Current Entropy = ", self.get_entropy(out_list))
            # else:
            #     print("Current gini index = 0", self.get_gini_index(out_list))
            # print()
            return Tree(None, output, tree_depth)

        # Find the best selection metric to split data.
        best_feature = self.get_best_selection_attribute(d, out_list, metric_type, features_list)

        # need to update this part of the code
        # print("Decision Tree depth = ", tree_depth)
        freq_map = self.get_freq(out_list)
        output = None
        max_count = -math.inf

        for i in classes:
            if i in freq_map:
            #     print("Count of", i, "=", 0)
            # else:
                if freq_map[i] > max_count:
                    output = i
                    max_count = freq_map[i]
                # print("Count of", i, "=", freq_map[i])

        # if metric_type == "gain":
        #     print("Current Entropy = ", self.get_entropy(out_list))
        #     print("Data is split using the feature: ", best_feature,
        #           " with Gain ratio= ", self.get_gain_ratio(d, out_list, best_feature))
        # elif metric_type == "gini":
        #     print("Current gini index = ", self.get_gini_index(out_list))
        #     print("Data is split using the feature: ", best_feature,
        #           " with Gini gain = ", self.get_gini_max(d, out_list, best_feature))
        # else:
        #     print("Current Entropy = ", self.get_entropy(out_list))
        #     print("Data is split using the feature: ", best_feature,
        #           " with information Gain = ", self.get_information_gain(d, out_list, best_feature))
        # print()
        values = set(d[:, best_feature])
        df_complete = pd.DataFrame(d)
        df_x = pd.DataFrame(d)
        df_complete[df_complete.shape[1]] = out_list
        curr_node = Tree(best_feature, output, tree_depth)
        index = features_list.index(best_feature)
        features_list.remove(best_feature)

        for i in values:
            dfx = df_x[df_x[best_feature] == i]
            dfy = df_complete[df_complete[best_feature] == i]
            node = self.decision_tree(dfx.to_numpy(), (dfy.to_numpy()[:, -1:]).flatten(), metric_type, features_list,
                                      tree_depth + 1, classes)
            curr_node.add_child(i, node)

        features_list.insert(index, best_feature)
        return curr_node

    def preprocessing(self, d, y, metric_type="gain"):
        features = [i for i in range(len(d[0]))]
        classes = set(y)
        initial_depth = 0
        self.__root = self.decision_tree(d, y, metric_type, features, initial_depth, classes)

    def __predict_for(self, data, node):
        # predicts the class for a given testing point and returns the answer

        # We have reached a leaf node
        # print("In Predict For: ")
        # print("Node value : ", node.val, " Node children: ", len(node.children), " Node output: ", type(node.output))
        if len(node.children) == 0:
            # print("Inside the if condition len(node.children) == 0:")
            return node.output
        # print("In Predict For after : len(node.children) == 0")

        val = data[node.val]  # represents the value of feature on which the split was made
        # print("In Predict For after val = data[node.val]")
        if val not in node.children:
            return node.output

        # print("In Predict For after if val not in node.children:")
        # Recursively call on the splits
        return self.__predict_for(data, node.children[val])

    def predict(self, d):
        Y = [0 for i in range(len(d))]
        for i in range(len(d)):
            # print("Predict: Row number = ", i, "| Data = ", d[i])
            Y[i] = self.__predict_for(d[i], self.__root)
        return np.array(Y)

    def score(self, X, Y):
        # returns the mean accuracy
        # Y_pred = self.predict(X)
        count = 0
        for i in range(len(Y)):
            if X[i] == Y[i]:
                count += 1
        return count / len(Y)

    def print_tree_only(self, node, spacing=""):
        # Base case: we've reached a leaf
        if len(node.children) == 0:
            print(spacing + "Leaf Node: Attribute_Split: " , str(node.val) , " Tree Depth = ",node.depth , " Label Class: " , str(node.output))
            return

        # Print the question at this node
        print(spacing + "Regular Node: with ", len(node.children) , " Children and Attribute_Split: " , str(node.val),  " Tree Depth = ",node.depth)

        for i in node.children:
            # Call this function recursively on the true branch
            print(spacing + '-->Child')
            self.print_tree_only(node.children[i], spacing + "  ")

    def print_tree(self):
        self.print_tree_only(self.__root, "")
# x = np.array([[0, 0],
#               [0, 1],
#               [1, 0]])
#
# y = np.array([0,
#               1,
#               1])
# x1 =([[1,1]])
# y1=np.array([1])
# print(x1)
# print(y1)
# clf1 = Decision_tree_classification()
# clf1.preprocessing(x, y)
# Y_pred = clf1.predict(x1)
# # print("Predictions :", Y_pred)
# print()
# print("Score :", clf1.score(y1, Y_pred)) # Score on training data
# print()
# clf1.print_tree()

from sklearn import model_selection

# data = pd.read_csv("abalone-Multipleclasses.csv.data", skiprows=1, header=None)
# data = pd.read_csv("DataSet\glass-identification-7-classes.csv", skiprows=1, header=None)
# Generating a random dataset
# X, Y = datasets.make_classification(n_samples=100, n_features=5, n_classes=3, n_informative=3, random_state=0)
# To reduce the values a feature can take ,converting floats to int
# df = pd.read_csv('DataSet\\tic-tac-toe-2classes.csv')
# df = pd.read_csv('DataSet\car-evaluation-4classes.csv')
# df = pd.read_csv('DataSet\seisimic-bumps-2classes.csv')
df = pd.read_csv('DataSet\\breast-cancer-2classes.csv')
# df = pd.read_csv('DataSet\poker-hand-training-true-9classes.csv')
# df = pd.read_csv('DataSet\data_banknote_authentication-2classes.csv')
lst = df.values.tolist()
trainDF_x, testDF_x = model_selection.train_test_split(lst, test_size=0.2)
trainDF_y = []
for l in trainDF_x:
    trainDF_y.append(l[-1])
    del l[-1]

testDF_y = []
for lst in testDF_x:
    testDF_y.append(lst[-1])
    del lst[-1]

print('X train: ', np.array(lst))
print('Y train: ', np.array(trainDF_y))
print('X test: ', np.array(testDF_x))
print('Y test: ', np.array(testDF_y))

clf2 = Decision_tree_classification()
tree = clf2.preprocessing(np.array(trainDF_x), np.array(trainDF_y).flatten(), "info_gain")
Y_pred2 = clf2.predict(np.array(testDF_x))
print("Predictions : ", Y_pred2)
print("Predictions Type: ", type(Y_pred2[0]))
print(clf2.score(np.array(testDF_y), Y_pred2))

clf2.print_tree()
print()
