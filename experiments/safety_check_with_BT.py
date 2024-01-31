"""
This code applies Bradley-Terry Preference learning model to learn a
set of weights for safety-critical applications.
The test data compares rule-satisfying behavior to rule-violating behavior.

In this implementation, FFT of signals are used as features.

Note: In our paper, the dataset is of the form S = {(s_i^+, s_i^)}_{i=1}^P,
      where s_i^+ is preferred over s_i^-.

Author: Ruya Karagulle
Date: 2023
"""

import os
import sys
import numpy as np
import torch

curr_dir = os.getcwd()
sys.path.insert(0, f"{curr_dir}/lib")

from benchmark_utils import *  # NOQA
from utils import *  # NOQA


def main():
    scn = "stop_sign"
    filename = f"./data/trajectories/{scn}_tracks_100.pkl"
    preference_file = f"./data/preferences/{scn}_preferences_1.pkl"
    preference_data = preference_loader(preference_file, scn)

    inputs = preprocess_signals(filename, scenario=scn)
    signal_no = len(inputs[0])

    # check daga consistency
    data_consistency = check_preference_consistency(preference_file)
    print(f"Preference consistency of participant: {100*data_consistency}%.")

    pairs = get_fft_features("BT", scn, inputs, preference_data)

    # load violating trajectories data
    violation_filename = (
        "./data/trajectories/stop_sign_tracks_violation_100.pkl"
    )
    violation_inputs = preprocess_signals(violation_filename, scn)

    all_inputs = (
        np.concatenate((inputs[0], violation_inputs[0]), axis=0),
        np.concatenate((inputs[1], violation_inputs[1]), axis=0),
    )

    violation_preferences = [[i, i + signal_no] for i in range(signal_no)]

    violation_pairs = get_fft_features(
        "BT", scn, all_inputs, violation_preferences
    )

    violation_feed = 0
    X_train = torch.cat((pairs, violation_pairs[:violation_feed]), axis=0)
    X_test = violation_pairs[violation_feed:]

    w, _ = BT_learn(X_train, learn_method="SGD")

    training_accuracy = decide_preference(w, X_train)
    test_accuracy = decide_preference(w, X_test)

    print(f"Accuracy of training set: {training_accuracy}")
    print(f"Accuracy of test set: {test_accuracy}")


if __name__ == "__main__":
    main()
