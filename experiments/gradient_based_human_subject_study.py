"""
Human Subject Study Safe Preference Learning with Gradient-Based Methods.

This Python module implements a gradient-based preference learning experiment
for human study safe scenarios. The experiment involves training and testing
on trajectory signals with preferences provided by human participants.

author: Ruya Karagulle
date: March 2023
version: 2.0
"""

import os
import sys
import random
import torch
import pandas as pd

curr_dir = os.getcwd()
sys.path.insert(0, f"{curr_dir}/lib")

from utils import *  # NOQA
from learners import *  # NOQA


def gradient_based_experiment(
    scenario: str,
    participant: int,
    train_ind: list,
    test_ind: list,
    seed_no: int,
    learning_settings: dict,
):
    """
    Perform gradient-based preference learning experiment.

    Args:
        scenario (str): The scenario for the experiment.
        participant (int): The id of the human study participant.
        seed_no (int): The seed for randomization.
        learning_settings (dict): Dictionary containing learning settings.

    Returns:
        list: List containing training and test accuracies for each split.
    """
    random.seed(seed_no)

    # Load signals
    filename = f"./data/trajectories/{scenario}_tracks_100.pkl"
    preference_filename = (
        f"./data/preferences/{scenario}_preferences_{participant}.pkl"
    )

    # check data consistency
    if scenario == "stop_sign":
        data_consistency = check_preference_consistency(preference_filename)
        print(
            f"Preference consistency of participant {participant}:",
            f"{100*data_consistency}%.",
        )

    inputs = preprocess_signals(filename, scenario)
    preference_data = preference_loader(preference_filename, scenario)

    input_signals, max_input = signal_to_tensor(inputs, scenario, padded=True)
    phi = create_formula(scenario, max_input)

    initialization_no = 10

    results = []
    if split is not None:
        k = split

        training_preferences = [preference_data[i] for i in train_ind[k]]
        test_preferences = [preference_data[i] for i in test_ind[k]]

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

        results.append(
            single_instance_learner(
                [k, "STL"],
                input_signals,
                max_input,
                phi,
                training_preferences,
                test_preferences,
                learning_settings,
            )
        )
        for i in range(initialization_no):
            results.append(
                single_instance_learner(
                    [k, "random"],
                    input_signals,
                    max_input,
                    phi,
                    training_preferences,
                    test_preferences,
                    learning_settings,
                )
            )

    else:
        for k in range(max_split_no):
            training_preferences = [preference_data[i] for i in train_ind[k]]
            test_preferences = [preference_data[i] for i in test_ind[k]]

            phi.set_weights(max_input, random=False)
            STL_accuracy_train, _ = compute_accuracy(
                phi, input_signals, training_preferences, scale=-1
            )
            STL_accuracy_test, _ = compute_accuracy(
                phi, input_signals, test_preferences, scale=-1
            )

            print(
                f"STL accuracy of participant {participant},",
                f"split {k}: {STL_accuracy_train, STL_accuracy_test}",
                flush=True,
            )

            results.append(
                single_instance_learner(
                    [k, "STL"],
                    input_signals,
                    max_input,
                    phi,
                    training_preferences,
                    test_preferences,
                    learning_settings,
                )
            )
            for i in range(initialization_no):
                results.append(
                    single_instance_learner(
                        [k, "random"],
                        input_signals,
                        max_input,
                        phi,
                        training_preferences,
                        test_preferences,
                        learning_settings,
                    )
                )

    return results


if __name__ == "__main__":
    args = argument_parser()
    participant = args["participant"]
    seed_no = args["seed"]
    scenario = args["scenario"]
    split = args["split"]
    all_participants = args["all_participants"]

    random.seed(seed_no)

    lr = 1e-4
    scale = 10**10
    learning_settings = {
        "learning_rate": lr,
        "scale": scale,
        "stopping_condition": 1e-4,
        "visuals": True,
        "tensorboard": False,
    }

    torch.manual_seed(random.randrange(1000))
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
            p_results = gradient_based_experiment(
                scenario,
                p,
                train_ind,
                test_ind,
                seed_no,
                learning_settings,
            )
            df = pd.DataFrame(p_results)
            df.to_csv(
                f"./results/{scenario}_HSS_w_batching_lr{lr}.csv",
                encoding="utf-8",
                index=False,
            )

            train_results = np.array(
                [results.iloc[i]["train accuracy"][-1] for i in range(11)]
            )
            test_results = np.array(
                [results.iloc[i]["test accuracy"][-1] for i in range(11)]
            )

            sorted_data = np.lexsort((test_results, test_results))
            max_idx = sorted_data[-1]

            results = results + [train_results[max_idx], test_results[max_idx]]

        avg_results = np.mean(np.array(results), axis=0)
        print(f"Average Training Set Accuracy: {100*avg_results[0].round(3)}")
        print(f"Average Test Set Accuracy: {100*avg_results[1].round(3)}")

    else:
        results = gradient_based_experiment(
            scenario,
            participant,
            train_ind,
            test_ind,
            seed_no,
            learning_settings,
        )
        df = pd.DataFrame(results)
        df.to_csv(
            f"./results/{scenario}_HSS_w_batching_lr{lr}.csv",
            encoding="utf-8",
            index=False,
        )
