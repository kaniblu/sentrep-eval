"""Perform classification task on a set of labelled vectors."""

import os
import yaml
import logging
import multiprocessing

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.model_selection import KFold
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC
from yaap import ArgParser
from yaap import path


def evaluate_fold(clf, x_train, y_train, x_test, y_test):
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    accuracy = accuracy_score(y_true=y_test, y_pred=y_pred)

    return accuracy


def evaluate(x, y, method, cv_folds, n_processes=None):
    if n_processes is None:
        n_processes = multiprocessing.cpu_count()

    if method == 'naivebayes':
        clf = GaussianNB()
    elif method == 'linearsvm':
        clf = LinearSVC()
    elif method == 'logistic':
        clf = LogisticRegression(n_jobs=n_processes)
    elif method == 'randomforest':
        clf = RandomForestClassifier(n_jobs=n_processes, n_estimators=50)
    else:
        raise ValueError('Unknown classifier: {}'.format(method))

    kf = KFold(n_splits=cv_folds, shuffle=True)

    ret = {}
    acc_sum = 0

    for i, (train_index, test_index) in enumerate(kf.split(x)):
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        acc = evaluate_fold(clf=clf,
                            x_train=x_train, y_train=y_train,
                            x_test=x_test, y_test=y_test)
        acc_sum += acc
        ret["fold-{}".format(i + 1)] = acc

    ret["average"] = acc_sum / cv_folds
    ret = {k: float(v) for k, v in ret.items()}

    return ret


def parse_args():
    parser = argparse.ArgumentParser(
        description='Evaluate generated sentence vectors.')
    parser.add_argument('--vectors', type=path, required=True,
                        help='The path of a sentence vector file')
    parser.add_argument('--labels', type=path, required=True,
                        help='The path of a answer file')
    parser.add_argument("--vectors-test", type=path, default=None)
    parser.add_argument("--labels-test", type=path, default=None)
    parser.add_argument("--save-path", type=path, required=True)
    parser.add_argument("--folds", default=5, type=int,
                        help='The number of folds to cross-validate')
    parser.add_argument("--method", required=True,
                        choices=['naivebayes', 'linearsvm', 'logistic'],
                        help='The name of a classifier to use')
    parser.add_argument("--processes", type=int, default=None)

    args = parser.parse_args()

    return args


def read_vector_file(vector_path):
    return np.loadtxt(vector_path, delimiter=',')


def read_answer_file(answer_path):
    label_dict = dict()
    label_ids = []
    with open(answer_path, 'r', encoding='utf-8') as f:
        for line in f:
            label = line.strip()
            label_id = label_dict.setdefault(label, len(label_dict))
            label_ids.append(label_id)
    label_ids = np.array(label_ids)
    return label_ids


def main():
    args = parse_args()

    vector_path = args.vector_path
    answer_path = args.label_path
    cv_folds = args.n_folds
    classifier_name = args.method
    result_path = args.result_path
    n_processes = args.n_processes

    print("Loading files...")
    x = read_vector_file(vector_path)
    y = read_answer_file(answer_path)

    print("Evaluating...")
    ret = evaluate(x, y, classifier_name, cv_folds, n_processes)

    ensure_dir_exists(result_path)

    print("Saving...")
    with open(result_path, "w") as f:
        yaml.dump(ret, f, default_flow_style=False)


if __name__ == '__main__':
    main()
