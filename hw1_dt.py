import numpy as np
import utils as Util
from collections import Counter
from collections import deque


class DecisionTree():
    def __init__(self):
        self.clf_name = "DecisionTree"
        self.root_node = None

    def train(self, features, labels):
        # features: List[List[float]], labels: List[int]
        # init
        assert (len(features) > 0)
        num_cls = np.unique(labels).size

        # build the tree
        self.root_node = TreeNode(features, labels, num_cls)
        if self.root_node.splittable:
            self.root_node.split()

        return

    def predict(self, features):
        # features: List[List[any]]
        # return List[int]
        y_pred = []
        for idx, feature in enumerate(features):
            pred = self.root_node.predict(feature)
            y_pred.append(pred)
        return y_pred

    def add_real_predicted_label(self, x_test, y_test):
        self.root_node.optimize_tree(x_test, y_test)


class TreeNode(object):
    def __init__(self, features, labels, num_cls):
        # features: List[List[any]], labels: List[int], num_cls: int
        self.features = features
        self.labels = labels
        self.children = []
        self.num_cls = num_cls
        # find the most common labels in current node
        count_max = 0
        for label in np.unique(labels):
            if self.labels.count(label) > count_max:
                count_max = labels.count(label)
                self.cls_max = label
                # splitable is false when all features belongs to one class
        if len(np.unique(labels)) < 2:
            self.splittable = False
        else:
            self.splittable = True

        self.dim_split = None  # the index of the feature to be split

        self.feature_uniq_split = None  # the possible unique values of the feature to be split

    # TODO: try to split current node
    def split(self):

        self.lables = np.array(self.labels)
        self.features = np.array(self.features)

        if self.num_cls == 1:
            self.splittable = False
            return self
        if self.features.size == 0:
            self.splittable = False
            return self

        branches = []
        entropy = 0
        total_label = len(self.labels)

        feat = self.features
        lab = self.labels

        for label in np.unique(lab):
            number = lab.count(label) / total_label
            if number > 0:
                entropy += -number * np.log2(number)

        arr = []
        for i in range(0, feat.shape[1]):
            attribute = feat[:, i]
            current_attribute_unique_size = np.unique(attribute).size
            current_index = i

            for unique_attribute in np.unique(attribute):
                indices_array = np.where(feat[:, i] == unique_attribute)
                labels_ar = []
                for index in indices_array[0]:
                    labels_ar.append(lab[index])

                label_per_value = []
                label_ctr = Counter(labels_ar)
                for label in np.unique(lab):
                    if label in label_ctr:
                        label_per_value.append(label_ctr[label])
                    else:
                        label_per_value.append(0)

                branches.append(label_per_value)

            gain = Util.Information_Gain(entropy, branches)

            arr.append([gain, current_attribute_unique_size, current_index])

            branches = []

        arr = sorted(arr, key=lambda x: (x[0], x[1], -x[2]), reverse=True)

        self.dim_split = arr[0][2]
        ig_attributes = np.array(arr)
        ig_attributes = ig_attributes[:, 0]

        if np.all(ig_attributes == 0):
            self.splittable = False
            return self

        unique_features = np.unique(feat[:, self.dim_split])

        for value in unique_features:
            indices_array = np.where(feat[:, self.dim_split] == value)

            labels_pass = []
            features_pass = []
            labels_left = []
            features_left = []

            for i, feature in enumerate(feat):
                if i in indices_array[0]:
                    features_pass.append(feature.tolist())
                    labels_pass.append(lab[i])
                else:
                    features_left.append(feature.tolist())
                    labels_left.append(lab[i])

            features_pass = np.delete(features_pass, self.dim_split, axis=1)

            feat = np.array(features_left)
            lab = np.array(labels_left)

            tree_node = TreeNode(features_pass, labels_pass, np.unique(labels_pass).size)
            tree_node.feature_uniq_split = value
            self.children.append(tree_node.split())

        return self

        raise NotImplementedError

    # TODO: predict the branch or the class
    def predict(self, feature):
        # feature: List[any]
        # return: int

        if not self.splittable:
            return self.cls_max

        for children in self.children:
            if feature[self.dim_split] == children.feature_uniq_split:
                feature1 = np.delete(feature, self.dim_split, None)
                return children.predict(feature1)

        return self.cls_max
        raise NotImplementedError

    def optimize_tree(self, X_test, y_test):
        queue = deque()
        queue.append(self)
        stack = []
        stack = self.create_list(stack, queue)

        c_accuracy = self.get_accuracy(X_test, y_test)
        c_node = 0

        flag = True

        while flag:
            flag = False
            for node in stack:
                if node.splittable:
                    deleted = node.children
                    node.splittable = False
                    node.children = []

                    acc = self.get_accuracy(X_test, y_test)

                    if acc > c_accuracy:
                        flag = True
                        c_node = node
                        c_accuracy = acc

                    node.splittable = True
                    node.children += deleted

            if flag:
                for child in c_node.children:
                    self.remove_children(stack, child)
                c_node.splittable = False
                c_node.children = []

    def remove_children(self, stack, child):
        if not child.children:
            stack.remove(child)
            return
        for c1 in child.children:
            self.remove_children(stack, c1)
        return

    def get_accuracy(self, X_test, y_test):
        predict = []
        real = y_test
        a = 0
        for idx, feature in enumerate(X_test):
            predict.append(self.predict(feature))

        for i in range(len(predict)):
            if predict[i] == real[i]:
                a += 1

        a /= len(real)

        return a

    def create_list(self, stack, queue):
        if queue:
            current = queue.popleft()
            stack.append(current)
            for children in current.children:
                queue.append(children)
            self.create_list(stack, queue)

        return stack
