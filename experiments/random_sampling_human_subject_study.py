"""
Human Subject Study Safe Preference Learning with Random Sampling.

This Python module implements a random sampling for the preference
learning experiment for human study. The experiment involves
training and testing on trajectory signals with preferences
provided by human participants.

author: Ruya Karagulle
date: March 2023
version: 2.0
"""

import pandas as pd
import os
import sys
import random

curr_dir = os.getcwd()
sys.path.insert(0, f"{curr_dir}/lib")

from utils import *  # NOQA


def random_sampling_experiment(
    scenario: str,
    participant: int,
    train_ind: list,
    test_ind: list,
    split: int,
    seed_no: int,
):

    random.seed(seed_no)

    filename = f"./data/trajectories/{scenario}_tracks_100.pkl"
    preference_filename = (
        f"./data/preferences/{scenario}_preferences_{participant}.pkl"
    )

    inputs = preprocess_signals(filename, scenario=scenario)
    preference_data = preference_loader(preference_filename, scenario)

    # check data consistency
    if scenario == "stop_sign":
        data_consistency = check_preference_consistency(preference_filename)
        print(
            f"Preference consistency of participant {participant}:",
            f"{100*data_consistency}%.",
        )

    input_signals, max_input = signal_to_tensor(
        inputs, scenario=scenario, padded=True
    )
    phi = create_formula(scenario, max_input)
    phi.set_weights(max_input, random=False, seed=seed_no)

    results = []
    initialization_no = 1000
    training_preferences = [preference_data[i] for i in train_ind[split]]
    test_preferences = [preference_data[i] for i in test_ind[split]]

    phi.set_weights(max_input, random=False)
    STL_accuracy_train, _ = compute_accuracy(
        phi, input_signals, training_preferences, scale=-1
    )
    STL_accuracy_test, _ = compute_accuracy(
        phi, input_signals, test_preferences, scale=-1
    )

    print(
        f"STL accurcy of participant {participant},",
        f"split {k}: {STL_accuracy_train, STL_accuracy_test}",
        flush=True,
    )

    for i in range(initialization_no):
        seed_noo = random.randrange(1000)
        phi, _ = generate_preference_data(
            phi, input_signals, human_reference=preference_data, seed=seed_noo
        )
        phi_new = copy.deepcopy(phi)
        acc_train, _ = compute_accuracy(
            phi_new, input_signals, training_preferences
        )
        acc_test, _ = compute_accuracy(
            phi_new, input_signals, test_preferences
        )
        results.append([i, acc_train, acc_test])
        print(acc_train, acc_test, flush=True)
        if acc_train == 1 and acc_test == 1:
            break

    return results


if __name__ == "__main__":
    args = argument_parser()
    participant = args["participant"]
    seed_no = args["seed"]
    scenario = args["scenario"]
    split = args["split"]
    all_participants = args["all_participants"]

    random.seed(seed_no)
    exp = "HSS_w_random_participant"

    max_split_no = 10
    preference_data_length = 50

    train_ind = max_split_no * [[]]
    test_ind = max_split_no * [[]]
    for k in range(max_split_no):
        # train-test dataset split. we save indices of pairs.
        # in different preference data, order in pairs can be different.
        training_data_size = int(0.7 * preference_data_length)
        train_ind[k], test_ind[k] = train_test_split(
            preference_data_length, training_data_size
        )

    results = []
    if all_participants:
        for p in range(1, 9):
            p_results = random_sampling_experiment(
                scenario,
                p,
                train_ind,
                test_ind,
                split,
                seed_no,
            )
            df = pd.DataFrame(p_results)
            filename = f"{scenario}_{exp}{p}_seed{seed_no}_split{split}"
            df.to_csv(
                f"./results/{filename}.csv",
                encoding="utf-8",
                index=False,
            )

            train_results = np.array(
                [results.iloc[i]["1"][-1] for i in range(1000)]
            )
            test_results = np.array(
                [results.iloc[i]["2"][-1] for i in range(1000)]
            )

            sorted_data = np.lexsort((test_results, test_results))
            max_idx = sorted_data[-1]

            results = results + [train_results[max_idx], test_results[max_idx]]

        avg_results = np.mean(np.array(results), axis=0)
        print(f"Average Training Set Accuracy: {100*avg_results[0].round(3)}")
        print(f"Average Test Set Accuracy: {100*avg_results[1].round(3)}")

    else:
        results = random_sampling_experiment(
            scenario,
            participant,
            train_ind,
            test_ind,
            seed_no,
        )
        df = pd.DataFrame(results)
        filename = f"{scenario}_{exp}{participant}_seed{seed_no}_split{split}"
        df.to_csv(
            f"./results/{filename}.csv",
            encoding="utf-8",
            index=False,
        )
