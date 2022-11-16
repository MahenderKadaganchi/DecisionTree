import pandas as pd
import numpy as np
import math
from sklearn import tree
from sklearn import metrics
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
            #print("Reached Leaf node with decision Tree depth = ", tree_depth)
            output = out_list[0]
            #print("Count of ", out_list[0], "=", len(out_list))
            #if metric_type == "gain":
                #print("Current Entropy = 0")
            #else:
                #print("Current gini index = 0")
            #print()
            return Tree(None, output)

        # If there are no more features left to classify
        elif len(features_list) == 0:
            #print("Reached Leaf node with decision Tree depth = ", tree_depth)
            get_items_freq = self.new_get_freq(out_list)
            curr_count = -math.inf
            output = None
            for i in classes:
                if i in get_items_freq:
                    frequency = get_items_freq[i]
                    if frequency > curr_count:
                        output = i
                        curr_count = frequency
                    #print("Count of ", i, "=", frequency)
            #if metric_type == "gain":
                #print("Current Entropy = ", self.new_get_entropy(out_list))
            #else:
            #3print("Current gini index = 0", self.new_get_gini_value(out_list))
            #print()
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
                print()
            else:
                if freq_map[i] > max_count:
                    output = i
                    max_count = freq_map[i]
                print()

        if metric_type == "gain":
            print("Current Entropy = ", self.new_get_entropy(out_list))
            print("Data is split using the feature: ", best_metric,
                  " with Gain ratio= ", self.new_get_gain_ratio(d, out_list, best_metric))
        else:
            print("Current gini index = 0", self.new_get_gini_value(out_list))
            print("Data is split using the feature: ", best_metric,
                  " with Gini gain = ", self.new_get_gini_max(d, out_list, best_metric))
        print()
        values = set(d[:, best_metric])
        df_complete = pd.DataFrame(d)
        df_x = pd.DataFrame(d)
        df_complete[df_complete.shape[1]] = out_list
        curr_node = Tree(best_metric, output)
        index = features_list.index(best_metric)
        features_list.remove(best_metric)

        for i in values:
            dfx = df_x[df_x[best_metric] == i]
            dfy = df_complete[df_complete[best_metric] == i]
            node = self.decision_tree(dfx.to_numpy(), (dfy.to_numpy()[:, -1:]).flatten(),metric_type, features_list, tree_depth + 1, classes)
            curr_node.add_child(i, node)

        features_list.insert(index, best_metric)
        return curr_node

    def preprocessing(self, d, y, metric_type="gain"):
        features = [i for i in range(len(d[0]))]
        classes = set(y)
        initial_depth = 0
        self.__root = self.decision_tree(d, y,metric_type, features, initial_depth, classes)

    def __predict_for(self, data, node):
        # predicts the class for a given testing point and returns the answer

        # We have reached a leaf node
        #print("In Predict For: ")
        #print("Node value : ", node.val, " Node children: ", len(node.children), " Node output: ", type(node.output))
        if len(node.children) == 0:
            #print("Inside the if condition len(node.children) == 0:")
            return node.output
        #print("In Predict For after : len(node.children) == 0")

        val = data[node.val]  # represents the value of feature on which the split was made
        #print("In Predict For after val = data[node.val]")
        if val not in node.children:
            return node.output

        #print("In Predict For after if val not in node.children:")
        # Recursively call on the splits
        return self.__predict_for(data, node.children[val])

    def predict(self, d):
        Y = [0 for i in range(len(d))]
        for i in range(len(d)):
            #print("Predict: Row number = ", i,"| Data = ", d[i])
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
# # Y_pred = clf1.predict(x1)
# # print("Predictions :", Y_pred)
# print()
# # print("Score :", clf1.score(x1, y1)) # Score on training data
# print()

from sklearn import model_selection
#data = pd.read_csv("abalone-Multipleclasses.csv.data", skiprows=1, header=None)
# data = pd.read_csv("DataSet\glass-identification-7-classes.csv", skiprows=1, header=None)
# Generating a random dataset
#X, Y = datasets.make_classification(n_samples=100, n_features=5, n_classes=3, n_informative=3, random_state=0)
# To reduce the values a feature can take ,converting floats to int
# df = pd.read_csv('DataSet\\tic-tac-toe-2classes.csv')
# df = pd.read_csv('DataSet\car-evaluation-4classes.csv')
# df = pd.read_csv('DataSet\seisimic-bumps-2classes.csv')
# df = pd.read_csv('DataSet\contraceptivemethodchoice-3-classes.csv')
df = pd.read_csv('DataSet/poker-hand-training-true-9classes.csv')
is_categorical = True
if len(list(set(df.columns) - set(df._get_numeric_data().columns)))==0:
    is_categorical = False
lst = df.values.tolist()
trainDF_x, testDF_x = model_selection.train_test_split(lst, test_size=0.2)
trainDF_y =[]
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
clf2.preprocessing(np.array(trainDF_x), np.array(trainDF_y).flatten())
ourmodel_pred = clf2.predict(np.array(testDF_x))
print("Predictions : ", ourmodel_pred)
print("Predictions Type: ", type(ourmodel_pred[0]))
print(clf2.score(np.array(testDF_y), ourmodel_pred))
print()

if is_categorical:
    comp_df = df
    comp_df.fillna(0)
    comp_df = pd.get_dummies(comp_df, drop_first=True)
    #print(comp_df)
    X = comp_df.iloc[:, :-1].values
    Y = comp_df.iloc[:, -1].values.reshape(-1, 1)
    X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size=.2, random_state=41)
    clf4 = tree.DecisionTreeClassifier()
    #print(X_train)
    clf4.fit(X_train, Y_train)
    inbuiltmodel_pred = clf4.predict(X_test)

    print('Predictions Score of Inbuilt Model Categorical: ', metrics.accuracy_score(inbuiltmodel_pred, Y_test))

    from sklearn.svm import SVC
    from sklearn.metrics import accuracy_score

    svc = SVC()
    svc.fit(X_train, Y_train)
    svm_pred = svc.predict(X_test)
    print('SVM Model accuracy score with default hyperparameters Categorical: {0:0.4f}'.format(accuracy_score(svm_pred, Y_test)))

else:
    # Performing inbuilt DT on same dataset::
    clf4 = tree.DecisionTreeClassifier()
    clf4.fit(np.array(trainDF_x), np.array(trainDF_y))
    inbuiltmodel_pred = clf4.predict(np.array(testDF_x))
    print('Predictions Score of Inbuilt Model Numerical: ', clf2.score(np.array(testDF_y), inbuiltmodel_pred))

    # """""performing SVM on same dataset"""""
    from sklearn.svm import SVC
    from sklearn.metrics import accuracy_score

    svc = SVC()
    svc.fit(np.array(trainDF_x), np.array(trainDF_y).flatten())
    svm_pred = svc.predict(np.array(testDF_x))
    print('SVM Model accuracy score with default hyperparameters Numerical: {0:0.4f}'.format(accuracy_score(testDF_y, svm_pred)))


#Saving predictions generated by the three models
save_df = pd.DataFrame({"Original Test Data": testDF_y, "OurModel Predictions" : ourmodel_pred, "InBuilt DT Predictions:" : inbuiltmodel_pred, "SVM Model Predictions" : svm_pred})
save_df.to_csv("Predictions.csv", index=False)

if len(set(trainDF_y))==2:
    # Confusion matrix only for 2 classes
    import matplotlib.pyplot as plt

    from sklearn import metrics

    confusion_matrix = metrics.confusion_matrix(np.array(testDF_y), np.array(ourmodel_pred))
    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=confusion_matrix, display_labels=[False, True])

    cm_display.plot()
    plt.show()
