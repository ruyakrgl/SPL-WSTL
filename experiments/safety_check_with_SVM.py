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
This program applies traditional SVM approach to preference learning
for safety-critical applications. The test data compares rule-satisfying
behavior to rule-violating behavior.
In this implementation
- FFT of signals,
- STL robustness are used as features.
Data of the form
- (preferred_signal_features-non_preferred_signal_features): class 0,
- (non_preferred_signal_features-preferred_signal_features): class 1.

Author: Ruya Karagulle
Date: 2023
"""

import os
import sys
import numpy as np
from scipy.fft import fft2
from sklearn import svm
from sklearn.metrics import accuracy_score
import torch


curr_dir = os.getcwd()
sys.path.insert(0, f"{curr_dir}/lib")

from utils import *  # NOQA
import WSTL  # NOQA
from WSTL import Expression  # NOQA


def main():
    scn = "stop_sign"
    filename = f"./data/trajectories/{scn}_tracks_100.pkl"
    preference_file = f"./data/preferences/{scn}_preferences_1.pkl"
    preference_data = preference_loader(preference_file, scn)

    inputs = preprocess_signals(filename, scenario=scn)
    input_signals, max_input = signal_to_tensor(
        inputs, scenario=scn, padded=True
    )

    s1 = Expression("p", max_input[0][0])
    s2 = Expression("v", max_input[1])

    e1 = torch.tensor(0, dtype=torch.float, requires_grad=True)
    e2 = torch.tensor(3.0, dtype=torch.float, requires_grad=True)
    e3 = torch.tensor(0, dtype=torch.float, requires_grad=True)

    phi1 = WSTL.Eventually(
        WSTL.Always(WSTL.And(subformula1=(s1 >= e1), subformula2=(s1 <= e2)))
    )
    phi = WSTL.Always(WSTL.And(subformula1=phi1, subformula2=(s2 >= e3)))

    phi.set_weights(max_input, random=False)
    robs = phi.robustness(input_signals, scale=-1)

    # Pad signals to the same time length, then take their FFTs.
    signal_no = len(inputs[0])
    pair_no = len(preference_data)

    fft_signals = signal_no * [[]]
    sum_k = 5
    for i in range(signal_no):
        position = input_signals[0][0][i]
        speed = input_signals[1][i]
        fft_array = np.abs(fft2([position, speed])).flatten()
        fft_sum = [
            sum(fft_array[i : i + sum_k])
            for i in range(0, len(fft_array), sum_k)
        ]
        final_features = fft_sum + [robs[i].item()]

        fft_signals[i] = torch.tensor(
            final_features, dtype=torch.float64
        ).flatten()

        fft_pairs_c0 = torch.zeros((pair_no, fft_signals[0].shape[0]))
        fft_pairs_c1 = torch.zeros((pair_no, fft_signals[0].shape[0]))

    for k, p_tuple in enumerate(preference_data):
        fft_pairs_c0[k, :] = fft_signals[p_tuple[0]].flatten().reshape(
            1, -1
        ) - fft_signals[p_tuple[1]].flatten().reshape(1, -1)
        fft_pairs_c1[k, :] = fft_signals[p_tuple[1]].flatten().reshape(
            1, -1
        ) - fft_signals[p_tuple[0]].flatten().reshape(1, -1)

    violation_filename = (
        "./data/trajectories/stop_sign_tracks_violation_100.rk"
    )
    violation_inputs = preprocess_signals(violation_filename, scn)
    violation_signal_no = len(violation_inputs[0])

    violation_input_signals, _ = signal_to_tensor(
        violation_inputs, scenario=scn, padded=True
    )
    violation_robs = phi.robustness(violation_input_signals, scale=-1)

    violation_fft_signals = violation_signal_no * [[]]
    # note that we also pad violating trajectories to
    # the length of correct trajectories.
    # If violating trajectories are longer than correct ones,
    # it will throw an error.
    for i in range(violation_signal_no):
        violation_position = violation_input_signals[0][0][i]
        violation_speed = violation_input_signals[1][i]
        violation_fft_array = np.abs(
            fft2([violation_position, violation_speed])
        ).flatten()

        violation_fft_sum = [
            sum(violation_fft_array[i : i + sum_k])
            for i in range(0, len(violation_fft_array), sum_k)
        ]

        violation_final_features = violation_fft_sum + [
            violation_robs[i].item()
        ]
        violation_fft_signals[i] = torch.tensor(
            violation_final_features, dtype=torch.float64
        ).flatten()

    violation_pairs = torch.zeros(
        violation_signal_no, len(violation_fft_signals[0]), dtype=torch.float64
    )

    for k in range(violation_signal_no):
        violation_pairs[k, :] = fft_signals[k].flatten().reshape(
            1, -1
        ) - violation_fft_signals[k].flatten().reshape(1, -1)

    svm_fft_data = np.concatenate((fft_pairs_c0, fft_pairs_c1), axis=0)
    labels = np.concatenate((np.zeros(pair_no), np.ones(pair_no)), axis=0)

    violation_labels = np.zeros(violation_signal_no)

    violation_feed = 0
    X_train = np.concatenate(
        (svm_fft_data, violation_pairs[:violation_feed]), axis=0
    )
    y_train = np.concatenate(
        (labels, violation_labels[:violation_feed]), axis=0
    )
    X_test = violation_pairs[violation_feed:]
    y_test = violation_labels[violation_feed:]

    classifier = svm.SVC()
    classifier.fit(X_train, y_train)

    y_pred = classifier.predict(X_test)
    y_train_pred = classifier.predict(X_train)

    training_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_pred)

    print(f"Accuracy of training set: {training_accuracy}")
    print(f"Accuracy of test set: {test_accuracy}")


if __name__ == "__main__":
    main()
