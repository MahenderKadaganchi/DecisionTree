import pandas as pd
import numpy as np
import math


class Tree:
    def __init__(self, val, output):
        self.val = val
        self.children = {}
        self.output = output

    def add_child(self, selected_feature, node):
        self.children[selected_feature] = node


class Decision_tree_classification:
    def __init__(self):
        self.__root = None

    def new_get_freq(self, n):
        data_split = {}
        for i in n:
            if i in data_split:
                data_split[i] += 1
            else:
                data_split[i] = 1
        return data_split

    def new_get_entropy(self, n):
        get_items_freq = self.new_get_freq(n)
        entropy = 0
        total = len(n)
        for i in get_items_freq:
            p = get_items_freq[i] / total
            entropy += p * math.log2(p)
        return -1 * entropy

    def new_get_split_info(self, x, feature_index):
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

    def new_get_gain_ratio(self, x, y, feature_index):
        tot_info = self.new_get_entropy(y)
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
            curr_info += (data_split[i] / total_size) * self.new_get_entropy(df1[df1.shape[1] - 1])

        split_info = self.new_get_split_info(x, feature_index)
        if split_info == 0:
            return math.inf
        else:
            gain = tot_info - curr_info
            return gain / split_info

    def new_get_gini_value(self, n):
        get_items_freq = self.new_get_freq(n)
        gini = 0
        total = len(n)
        for i in get_items_freq:
            p = get_items_freq[i] / total
            gini += p ** 2
        return 1 - gini

    def new_get_gini_max(self, x, y, feature_index):
        total_gini_value = self.new_get_gini_value(y)
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
            curr_gini += (data_split[i] / total_size) * self.new_get_gini_value(df1[df1.shape[1] - 1])
        return total_gini_value - curr_gini

    def get_best_selection_metric(self, d, y, type, feature_list):
        max_value = -math.inf
        best_feature = None
        if type == "gini":
            for i in feature_list:
                curr_gain = self.new_get_gini_max(d, y, i)
                if curr_gain > max_value:
                    best_feature = i
                    max_value = curr_gain
        else:
            for i in feature_list:
                curr_gain = self.new_get_gain_ratio(d, y, i)
                if curr_gain > max_value:
                    best_feature = i
                    max_value = curr_gain
        return best_feature

    def decision_tree(self, d, out_list, metric_type, features_list, tree_depth, classes):
        # If the node consists of only one class.
        if len(set(out_list)) == 1:
            print("Reached Leaf node with decision Tree depth = ", tree_depth)
            output = out_list[0]
            print("Count of ", out_list[0], "=", len(out_list))
            if metric_type == "gain":
                print("Current Entropy = 0")
            else:
                print("Current gini index = 0")
            print()
            return Tree(None, output)

        # If there are no more features left to classify
        elif len(features_list) == 0:
            print("Reached Leaf node with decision Tree depth = ", tree_depth)
            get_items_freq = self.new_get_freq(out_list)
            curr_count = -math.inf
            output = None
            for i in classes:
                if i in get_items_freq:
                    frequency = get_items_freq[i]
                    if frequency > curr_count:
                        output = i
                        curr_count = frequency
                    print("Count of ", i, "=", frequency)
            if metric_type == "gain":
                print("Current Entropy = ", self.new_get_entropy(out_list))
            else:
                print("Current gini index = 0", self.new_get_gini_value(out_list))
            print()
            return Tree(None, output)

        # Find the best selection metric to split data.
        best_metric = self.get_best_selection_metric(d, out_list, metric_type, features_list)

        # need to update this part of the code
        print("Decision Tree depth = ", tree_depth)
        freq_map = self.new_get_freq(out_list)
        output = None
        max_count = -math.inf

        for i in classes:
            if i not in freq_map:
                print("Count of", i, "=", 0)
            else:
                if freq_map[i] > max_count:
                    output = i
                    max_count = freq_map[i]
                print("Count of", i, "=", freq_map[i])

        if metric_type == "gain":
            print("Current Entropy = ", self.new_get_entropy(out_list))
            print("Data is split using the feature: ", best_metric,
                  " with Gain ratio= ", self.new_get_gain_ratio(x, out_list, best_metric))
        else:
            print("Current gini index = 0", self.new_get_gini_value(out_list))
            print("Data is split using the feature: ", best_metric,
                  " with Gini gain = ", self.new_get_gini_max(x, out_list, best_metric))
        print()
        values = set(x[:, best_metric])
        df_complete = pd.DataFrame(x)
        df_x = pd.DataFrame(x)
        df_complete[df_complete.shape[1]] = out_list
        curr_node = Tree(best_metric, output)
        temp_index = -1
        for i in features_list:
            if i == best_metric:
                temp_index = i
                break
        features_list.remove(best_metric)

        for i in values:
            dfx = df_x[df_x[best_metric] == i]
            dfy = df_complete[df_complete[best_metric] == i]
            node = self.decision_tree(dfx.to_numpy(), (dfy.to_numpy()[:, -1:]).flatten(), features_list, tree_depth + 1,
                                      features_list, classes)
            curr_node.add_child(i, node)

        features_list.insert(temp_index, best_metric)
        return curr_node

    def preprocessing(self, d, y, metric_type="gain"):
        feature_list = []
        for i in range(len(d[0])):
            feature_list.append(i)
        classes = set(y)
        initial_depth = 0
        self.__root = self.decision_tree(d, y, metric_type, feature_list, initial_depth, classes)

    def __predict_for(self, data, node):
        # predicts the class for a given testing point and returns the answer

        # We have reached a leaf node
        if len(node.children) == 0:
            return node.output

        val = data[node.val]  # represents the value of feature on which the split was made
        if val not in node.children:
            return node.output

        # Recursively call on the splits
        return self.__predict_for(data, node.children[val])

    def predict(self, d):
        Y = np.array([0 for i in range(len(d))])
        for i in range(len(d)):
            Y[i] = self.__predict_for(d[i], self.__root)
        return Y

    def score(self, X, Y):
        # returns the mean accuracy
        Y_pred = self.predict(X)
        count = 0
        for i in range(len(Y_pred)):
            if Y_pred[i] == Y[i]:
                count += 1
        return count / len(Y_pred)


x = np.array([[0, 0, 5],
              [0, 1, 2],
              [1, 0, 3],
              [1, 1, 4]])

y = np.array([0,
              1,
              1,
              1])
print(x)
print(y)
clf1 = Decision_tree_classification()
clf1.preprocessing(x, y)
Y_pred = clf1.predict(x)
print("Predictions :", Y_pred)
print()
print("Score :", clf1.score(x, y)) # Score on training data
print()

# clf1 = Decision_tree_classification()
# # print(__gain_ratio(x, y, 2))
# print(clf1.new_get_gain_ratio(x, y, 0))
# # print(__gini_index(y))
# print(clf1.new_get_gini_value(y))
# # print(__gini_gain(x,y,2))
# print(clf1.new_get_gini_max(x, y, 0))
# df_x = pd.DataFrame(x)
# dfx = df_x[df_x[0] == 0]
# print(dfx.to_numpy())
# df_complete = pd.DataFrame(x)
# df_complete[df_complete.shape[1]] = y
# dfy = df_complete[df_complete[0] == 0]
# yy = dfy.to_numpy()[:, -1:]
# print(yy.flatten())