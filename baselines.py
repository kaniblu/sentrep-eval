import io
import os
import yaml

import numpy as np
import configargparse as argparse
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

from utils.evaluate import evaluate
from utils import ensure_dir_exists


def parse_args():
    parser = argparse.ArgParser()
    parser.add_argument("--sent_path", type=str, required=True)
    parser.add_argument("--label_path", type=str, required=True)
    parser.add_argument("--result_path", type=str, required=True)
    parser.add_argument("--method", type=str, required=True,
                        choices=["naivebayes", "linearsvm", "logistic",
                                 "randomforest"])
    parser.add_argument("--vector", type=str, required=True,
                        choices=["bow", "tfidf", "lsa"])
    parser.add_argument("--lsa_size", type=int, default=100)
    parser.add_argument("--n_folds", type=int, default=10)
    parser.add_argument("--n_processes", type=int, default=None)

    args = parser.parse_args()

    return args


def create_vectorizing_pipeline(vector_method, lsa_size=2):
    if vector_method == "bow":
        lvl = 0
    elif vector_method == "tfidf":
        lvl = 1
    elif vector_method == "lsa":
        lvl = 2
    else:
        raise ValueError("Unrecognized vectorization method: {}".format(
            vector_method
        ))

    pipeline = []

    if lvl >= 0:
        pipeline.append(CountVectorizer())

    if lvl >= 1:
        pipeline.append(TfidfTransformer())

    if lvl >= 2:
        pipeline.append(TruncatedSVD(n_components=lsa_size))

    pipeline = make_pipeline(*pipeline)

    return pipeline


def load_data(sent_path, label_path, vector_pipeline):
    with io.open(sent_path, "r", encoding="utf-8") as f:
        sents = f.readlines()

    with io.open(label_path, "r", encoding="utf-8") as f:
        labels = f.readlines()

    sents = vector_pipeline.fit_transform(sents)
    labels_vocab = list(set(labels))
    labels = np.array([labels_vocab.index(l) for l in labels])

    if not isinstance(sents, np.ndarray):
        sents = sents.toarray()

    return sents, labels


def main():
    args = parse_args()

    sent_path = args.sent_path
    label_path = args.label_path
    result_path = args.result_path
    method = args.method
    vector = args.vector
    lsa_size = args.lsa_size
    n_folds = args.n_folds
    n_processes = args.n_processes

    print("Creating pipeline...")
    pipeline = create_vectorizing_pipeline(vector, lsa_size)

    print("Loading data...")
    x, y = load_data(sent_path, label_path, pipeline)

    print("Evaluating...")
    ret = evaluate(x, y, method, n_folds, n_processes)

    ensure_dir_exists(result_path)

    print("Saving...")
    with open(result_path, "w") as f:
        yaml.dump(ret, f, default_flow_style=False)


if __name__ == '__main__':
    main()
