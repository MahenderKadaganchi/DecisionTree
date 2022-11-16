import pandas as pd
import numpy as np
import math
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier



class Tree:
    """ The Tree class contains the all the values that are present on each node and the child nodes data."""
    def __init__(self, val, output, depth):
        self.val = val
        self.children = {}
        self.output = output
        self.depth = depth

    def add_child(self, selected_feature, node):
        self.children[selected_feature] = node


class Decision_tree_classification:
    """This Class contains all the required methods to implement the decision tree"""

    # Initialize the root of the tree to None
    def __init__(self):
        self.__root = None

    # This method takes 1d array as an input and
    # returns a dictionary with keys as unique values and their frequencies as values.
    def get_unique_freq(self, n):
        data_split = {}
        for i in n:
            if i in data_split:
                data_split[i] += 1
            else:
                data_split[i] = 1
        return data_split

    # This method takes the output 1d array and calculates the degree of uncertainty, i.e. entropy
    def get_entropy(self, n):
        get_items_freq = self.get_unique_freq(n)
        entropy = 0
        total = len(n)
        for i in get_items_freq:
            p = get_items_freq[i] / total
            entropy += p * math.log2(p)
        return -1 * entropy

    # This method returns the split ratio value, used to calculate the gain ratio
    def get_split_ratio(self, x, feature_index):
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

    # This method is used to calculate the information gain for the given attribute
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

    # This method uss information gain and split ratio in order to calculate the gain ratio.
    def get_gain_ratio(self, x, y, feature_index):
        info_gain = self.get_information_gain(x, y, feature_index)
        split_ratio = self.get_split_ratio(x, feature_index)
        if split_ratio == 0:
            return math.inf
        else:
            return float(info_gain / split_ratio)

    # This method is used to calculate the gini index, which is then used to calculate the gini max value
    def get_gini_index(self, n):
        get_items_freq = self.get_unique_freq(n)
        gini = 0
        total = len(n)
        for i in get_items_freq:
            p = get_items_freq[i] / total
            gini += p ** 2
        return 1 - gini

    # This method is used to calculate the gini max value for the given attribute
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

    # This method loops through all the features present for the data set passed and
    # calculates the metrics value for the given metric and returns the best attribute.
    # We use this best metric to split the data set into the child nodes.
    def get_best_selection_attribute(self, d, y, type, feature_list):
        max_value = -math.inf
        best_feature = None
        for i in feature_list:
            # We check if the feature has only 2 unique values and
            # assign the selection metric to max gini, as it is the best in case of multivalued attributes.
            if len(self.get_unique_freq(d[:, i])) == 2 or type == "gini":
                curr_gain = self.get_gini_max(d, y, i)
                if curr_gain > max_value:
                    best_feature = i
                    max_value = curr_gain
            elif type == "gain":
                curr_gain = self.get_gain_ratio(d, y, i)
                if curr_gain > max_value:
                    best_feature = i
                    max_value = curr_gain
            else:
                curr_gain = self.get_information_gain(d, y, i)
                if curr_gain > max_value:
                    best_feature = i
                    max_value = curr_gain
        return best_feature

    # Decision tree is implemented in this method.
    # If data set belongs to the same output class, we mark the node as leaf and return the node with output class.
    # If there are no more attributes present to divide the data further,
    # mark the node as leaf and output the class that has the highest frequency among the data sets.
    # We then check for the best attribute and split the data using that attribute.
    # Recursively call the decision_tree method for the child nodes and return the current node in the end.
    def decision_tree(self, d, out_list, metric_type, features_list, tree_depth, classes):
        # If the node consists of only one class.
        if len(set(out_list)) == 1:
            output = out_list[0]
            return Tree(None, output, tree_depth)

        # If there are no more features left to classify
        elif len(features_list) == 0:
            # print("Reached Leaf node with decision Tree depth = ", tree_depth)
            get_items_freq = self.get_unique_freq(out_list)
            curr_count = -math.inf
            output = None
            for i in classes:
                if i in get_items_freq:
                    frequency = get_items_freq[i]
                    if frequency > curr_count:
                        output = i
                        curr_count = frequency
            return Tree(None, output, tree_depth)

        best_feature = self.get_best_selection_attribute(d, out_list, metric_type, features_list)
        freq_map = self.get_unique_freq(out_list)
        output = None
        max_count = -math.inf
        for i in classes:
            if i in freq_map:
                if freq_map[i] > max_count:
                    output = i
                    max_count = freq_map[i]
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

    # Preprocessing method is used to append all the attributes, output classes present in the data set
    def preprocessing(self, d, y, metric_type="gain"):
        features = [i for i in range(len(d[0]))]
        classes = set(y)
        initial_depth = 0
        self.__root = self.decision_tree(d, y, metric_type, features, initial_depth, classes)

    # This method is used to predict the output values for the given input
    def __predict_for(self, data, node):
        if len(node.children) == 0:
            return node.output
        val = data[node.val]
        if val not in node.children:
            return node.output
        return self.__predict_for(data, node.children[val])

    # This method is used for preprocessing to calculate the predicted output
    def predict(self, d):
        Y = [0 for i in range(len(d))]
        for i in range(len(d)):
            Y[i] = self.__predict_for(d[i], self.__root)
        return np.array(Y)

    # This method is used to calculate the precision of the model. It is scaled to 1.
    def score(self, X, Y):
        count = 0
        for i in range(len(Y)):
            if X[i] == Y[i]:
                count += 1
        return count / len(Y)

    # Print the tree in preorder traversal way.
    def print_tree_only(self, node, spacing=""):
        # Base case: we've reached a leaf
        if len(node.children) == 0:
            print(spacing + "Leaf Node: Attribute_Split: " , str(node.val) , " Tree Depth = ",node.depth , " Label Class: " , str(node.output))
            return

        # Print the Node with the number of children and attribute used to split
        print(spacing + "Regular Node: with ", len(node.children) , " Children and Attribute_Split: " , str(node.val),  " Tree Depth = ",node.depth)

        for i in node.children:
            # Call this function recursively on all the child branches
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

print('X train: ', np.array(trainDF_x))
print('Y train: ', np.array(trainDF_y))
print('X test: ', np.array(testDF_x))
print('Y test: ', np.array(testDF_y))

clf2 = Decision_tree_classification()
tree = clf2.preprocessing(np.array(trainDF_x), np.array(trainDF_y).flatten(), "gain")
Y_pred2 = clf2.predict(np.array(testDF_x))
print("Predictions of our model: ", Y_pred2)
# clf2.print_tree()
print()
print("Score of our model: ", clf2.score(np.array(testDF_y), Y_pred2))
print()


# print("Performing inbuilt DT on same dataset::")
# clf4 = DecisionTreeClassifier()
# clf4.fit(np.array(trainDF_x), np.array(trainDF_y))
# inbuiltmodel_pred = clf4.predict(np.array(testDF_x))
# print('Predictions Score of Inbuilt Model: {0:0.4f}'.format(clf2.score(np.array(testDF_y), inbuiltmodel_pred)))
# print()
#
# print("Performing SVM on same dataset::")
# from sklearn.svm import SVC
# from sklearn.metrics import accuracy_score
# svc=SVC()
# svc.fit(np.array(trainDF_x), np.array(trainDF_y).flatten())
# svm_pred=svc.predict(np.array(testDF_x))
# print('SVM Model accuracy score with default hyperparameters: {0:0.4f}'. format(accuracy_score(testDF_y, svm_pred)))
#
#
# #Saving predictions generated by the three models
# save_df = pd.DataFrame({"Original Test Data": testDF_y, "OurModel Predictions" : Y_pred2, "InBuilt DT Predictions:" : inbuiltmodel_pred, "SVM Model Predictions" : svm_pred})
# save_df.to_csv("Predictions.csv", index=False)
# print()
