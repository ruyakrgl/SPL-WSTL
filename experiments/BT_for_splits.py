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
This program applies Bradley-Terry Preference learning model
to learn a set of weights.
In this implementation,
- FFT of signals,
- STL robustness are used as features.

Note: The dataset is of the form S = {(s_i^+, s_i^)}_{i=1}^P,
      where s_i^+ is preferred over s_i^-.

Author: Ruya Karagulle
Date: 2023
"""

import os
import sys
import random
import pandas as pd
import numpy as np

curr_dir = os.getcwd()
sys.path.insert(0, f"{curr_dir}/lib")

from utils import *  # NOQA
from benchmark_utils import *  # NOQA


def BT_experiment(
    scenario: str, participant: int, seed_no: int, flush_results: bool = True
):
    """
    Perform Bradley-Terry Preference learning experiment.

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
            f"Preference consistency of participant {participant}:",
            f"{100*data_consistency}%.",
        )

    pairs = get_fft_features("BT", scenario, inputs, preference_data)

    max_split_no = 10
    train_ind = max_split_no * [[]]
    test_ind = max_split_no * [[]]
    for k in range(max_split_no):
        # train-test dataset split. we save indices of pairs.
        # in different preference data, order in pairs can be different.
        training_data_size = int(0.7 * len(preference_data))
        train_ind[k], test_ind[k] = train_test_split(
            len(preference_data), training_data_size
        )

    results = []
    for k in range(max_split_no):
        fft_trains = [pairs[i, :, :] for i in train_ind[k]]
        fft_tests = [pairs[i, :, :] for i in test_ind[k]]

        w, _ = BT_learn(fft_trains, learn_method="SGD")
        training_accuracy = decide_preference(w, fft_trains)
        test_accuracy = decide_preference(w, fft_tests)

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
    """
    Main function to run Bradley-Terry Preference learning
    for different splits.
    """
    args = argument_parser()
    participant = args["participant"]
    seed_no = args["seed"]
    scn = args["scenario"]
    all_participants = args["all_participants"]

    results = []
    if all_participants:
        for p in range(1, 9):
            p_results = BT_experiment(scn, p, seed_no, flush_results=False)
            df = pd.DataFrame(p_results)
            df.to_csv(
                f"./results/{scn}_p{participant}_bt_w_splits.csv",
                encoding="utf-8",
            )

            results = results + p_results

        all_results = np.array(results)
        avg_results = np.mean(all_results, axis=0)
        print(f"Average Training Set Accuracy: {100*avg_results[1].round(3)}")
        print(f"Average Test Set Accuracy: {100*avg_results[2].round(3)}")
    else:
        results = BT_experiment(scn, participant, seed_no, flush_results=True)
        df = pd.DataFrame(p_results)
        df.to_csv(
            f"./results/{scn}_p{participant}_bt_w_splits.csv", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
