# MIT License
#
# Copyright (c) 2024 by The Regents of the University of Michigan
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""
This program applies traditional SVM approach to preference learning.
In this implementation,
- FFT of signals,
- STL robustness are used as features.

Data of the form
- (preferred_signal_features-non_preferred_signal_features) : class 0,
- (non_preferred_signal_features-preferred_signal_features) : class 1.

Author: Ruya Karagulle
Date: 2023
"""

import os
import sys
import random
import pandas as pd
import numpy as np

from sklearn import svm
from sklearn.metrics import accuracy_score

curr_dir = os.getcwd()
sys.path.insert(0, f"{curr_dir}/lib")

from utils import *  # NOQA
from benchmark_utils import *  # NOQA


def SVM_experiment(
    scenario: str,
    participant: int,
    max_split_no: int,
    training_data_size: int,
    train_ind: list,
    test_ind: list,
    seed_no: int,
    flush_results: bool = False,
):
    """
    Perform traditional SVM preference learning experiment.

    Args:
        scenario (str): The scenario for the experiment.
        participant (int): The id of the human study participant.
        seed_no (int): The seed for randomization.
        flush_results (bool): Flag to print the results.

    Returns:
        list: List containing training and test accuracies for each split.
    """
    random.seed(seed_no)

    prf_wd = "./data/preferences"
    filename = f"./data/trajectories/{scenario}_tracks_100.pkl"
    preference_file = prf_wd + f"/{scenario}_preferences_{participant}.pkl"

    inputs = preprocess_signals(filename, scenario=scenario)
    preference_data = preference_loader(preference_file, scenario)

    # check data consistency
    if scenario == "stop_sign":
        data_consistency = check_preference_consistency(preference_file)
        print(
            f"Preference consistency of participant {participant}:,",
            f"{100*data_consistency}%.",
        )

    class0, class1 = get_fft_features("SVM", scenario, inputs, preference_data)

    results = []
    labels = np.concatenate(
        (np.zeros(training_data_size), np.ones(training_data_size)), axis=0
    )
    test_labels = np.zeros(len(preference_data) - training_data_size)
    for k in range(max_split_no):
        fft_trains = np.concatenate(
            (
                np.array([class0[i, :] for i in train_ind[k]]),
                np.array([class1[i, :] for i in train_ind[k]]),
            ),
            axis=0,
        )
        fft_tests = np.array([class0[i, :] for i in test_ind[k]])

        (
            X_train,
            X_test,
        ) = (
            fft_trains,
            fft_tests,
        )
        y_train, y_test = labels, test_labels

        classifier = svm.SVC()
        classifier.fit(X_train, y_train)

        y_pred = classifier.predict(X_test)
        y_train_pred = classifier.predict(X_train)

        training_accuracy = accuracy_score(y_train, y_train_pred)
        test_accuracy = accuracy_score(y_test, y_pred)

        if flush_results:
            print(
                f"Accuracy of training set for split {k},",
                f"participant {participant}: {training_accuracy}",
            )
            print(
                f"Accuracy of test set for split {k},",
                f"participant {participant}: {test_accuracy}",
            )

        results.append([k, training_accuracy, test_accuracy])
    return results


def main():
    args = argument_parser()
    participant = args["participant"]
    seed_no = args["seed"]
    scn = args["scenario"]
    all_participants = args["all_participants"]

    max_split_no = 10
    preference_data_length = 50

    train_ind = max_split_no * [[]]
    test_ind = max_split_no * [[]]
    for k in range(max_split_no):
        # train-test dataset split. we save indices of pairs.
        # In different preference data, order in pairs can be different.
        training_data_size = int(0.7 * preference_data_length)
        train_ind[k], test_ind[k] = train_test_split(
            preference_data_length, training_data_size
        )

    results = []
    if all_participants:
        for p in range(1, 9):
            p_results = SVM_experiment(
                scn,
                p,
                max_split_no,
                training_data_size,
                train_ind,
                test_ind,
                seed_no,
                flush_results=False,
            )
            df = pd.DataFrame(p_results)
            df.to_csv(
                f"./results/{scn}_p{p}_svm_w_splits.csv", encoding="utf-8"
            )

            results = results + p_results

        all_results = np.array(results)
        avg_results = np.mean(all_results, axis=0)
        print(f"Average Training Set Accuracy: {100*avg_results[1].round(3)}")
        print(f"Average Test Set Accuracy: {100*avg_results[2].round(3)}")
    else:
        results = SVM_experiment(
            scn,
            participant,
            max_split_no,
            training_data_size,
            train_ind,
            test_ind,
            seed_no,
            flush_results=False,
        )
        df = pd.DataFrame(p_results)
        df.to_csv(
            f"./results/{scn}_p{participant}_svm_w_splits.csv",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
