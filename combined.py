"""Evaluate generated sentence vectors."""

import os
import yaml
import argparse

import numpy as np

from utils.evaluate import evaluate
from utils import ensure_dir_exists
from evaluate.sentence_vector import read_answer_file
from evaluate.sentence_vector import read_vector_file
from evaluate.baselines import create_vectorizing_pipeline
from evaluate.baselines import load_data

def parse_args():
    parser = argparse.ArgumentParser(
        description='Evaluate generated sentence vectors.')
    parser.add_argument("--sent_path", required=True)
    parser.add_argument('--vector_path', required=True,
                        help='The path of a sentence vector file')
    parser.add_argument('--label_path', required=True,
                        help='The path of a answer file')
    parser.add_argument("--result_path", type=str, required=True)
    parser.add_argument('--n_folds', default=5, type=int,
                        help='The number of folds to cross-validate')
    parser.add_argument("--vector", type=str, required=True,
                        choices=["bow", "tfidf", "lsa"])
    parser.add_argument("--lsa_size", type=int, default=100)
    parser.add_argument('--method', required=True,
                        choices=['naivebayes', 'linearsvm', 'logistic', 'randomforest'],
                        help='The name of a classifier to use')
    parser.add_argument("--n_processes", type=int, default=None)

    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    sent_path = args.sent_path
    vector_path = args.vector_path
    answer_path = args.label_path
    cv_folds = args.n_folds
    classifier_name = args.method
    result_path = args.result_path
    vector_method = args.vector
    lsa_size = args.lsa_size
    n_processes = args.n_processes

    print("Loading files...")
    x = read_vector_file(vector_path)

    print("Loading data...")
    pl = create_vectorizing_pipeline(vector_method, lsa_size)
    x_b, y = load_data(sent_path, answer_path, pl)
    x = np.concatenate((x, x_b), 1)

    print("Evaluating...")
    ret = evaluate(x, y, classifier_name, cv_folds, n_processes)

    ensure_dir_exists(result_path)

    print("Saving...")
    with open(result_path, "w") as f:
        yaml.dump(ret, f, default_flow_style=False)


if __name__ == '__main__':
    main()
