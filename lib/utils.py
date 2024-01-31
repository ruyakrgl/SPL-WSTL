"""
Utility module for SPL-WSTL.

This module contains various utility functions used across SPL-WSTL.
These functions cover tasks such as data preprocessing,
                                    signal handling,
                                    formula manipulation,
                                    and synthetic data generation.


Author: Ruya Karagulle
Date: 2023
"""

import os
import sys
import torch
import copy
import pickle
import numpy as np
import random
import argparse


curr_dir = os.getcwd()
sys.path.insert(0, f"{curr_dir}/lib")

import WSTL  # NOQA
from WSTL import Expression  # NOQA

# --- ARGUMENT PARSER ---


def argument_parser():
    """
    Parse command-line arguments.

    Returns:
        dict: Dictionary containing parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Experiment arguments.")
    parser.add_argument(
        "--scenario",
        type=str,
        default="stop_sign",
        choices=["stop_sign", "pedestrian"],
        help="Choose scenario.",
    )
    parser.add_argument(
        "--participant",
        type=int,
        default=1,
        choices=np.arange(1, 9),
        help="The id of human study participant. Choose from 1-4.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="The seed for randomization. Experiments are done with seed 0.",
    )
    parser.add_argument(
        "--split", type=int, default=None, help="The train-test split"
    )
    parser.add_argument(
        "--all_participants",
        action="store_true",
        help="Flag to run the experiment for all participants",
    )
    return vars(parser.parse_args())


# --- UTILITY FUNCTIONS ---
def pad_signal(signal: torch.Tensor, max_length: int):
    """
    Pads the signal tensor with its last element to
    the specified maximum length.

    Args:
        signal (torch.Tensor): Input signal tensor.
        max_length (int): Target maximum length.

    Returns:
        torch.Tensor: Padded signal tensor.
    """
    padded_signal = (
        torch.cat(
            (
                signal,
                torch.tensor(
                    (max_length - signal.shape[0]) * [signal[-1].item()]
                ),
            ),
            axis=0,
        )
        .unsqueeze(0)
        .unsqueeze(-1)
    )
    return padded_signal


def create_input_signals(
    scenario: str,
    position: torch.Tensor,
    speed: torch.Tensor,
    pedestrian: torch.Tensor = None,
):
    """
    Create input signals and max signal for the specified scenario.

    Args:
        scenario (str): Either 'stop_sign' or 'pedestrian'.
        position (torch.Tensor): Tensor representing position signals.
        speed (torch.Tensor): Tensor representing speed signals.
        pedestrian (torch.Tensor, optional): Tensor representing pedestrian
                                            signals for 'pedestrian' scenario.

    Returns:
        Tuple: Tuple containing input signals and max signal.
    """

    if scenario not in ["stop_sign", "pedestrian"]:
        raise TypeError(
            f"create_input_signals():{scenario} is not a valid scenario."
        )

    if scenario == "stop_sign":
        input_signals = ((position, position), speed)
        max_signal = (
            (position[0, :, :].unsqueeze(0), position[0, :, :].unsqueeze(0)),
            speed[0, :, :].unsqueeze(0),
        )

    else:
        input_signals = (
            ((pedestrian, position), (position, pedestrian)),
            speed,
        )
        max_signal = (
            (
                (
                    pedestrian[0, :, :].unsqueeze(0),
                    position[0, :, :].unsqueeze(0),
                ),
                (
                    position[0, :, :].unsqueeze(0),
                    pedestrian[0, :, :].unsqueeze(0),
                ),
            ),
            speed[0, :, :].unsqueeze(0),
        )
    return input_signals, max_signal


def get_robustness(
    formula: WSTL.WSTL_Formula, signals: tuple or list, scale: float
):
    """
    Computes robustness values of given signals and formula formula.

    Parameters:
        formula (WSTL.WSTL_Formula): WSTL formula. WSTL_formula type.
        signals (tuple or list): set of signals.
        scale (float): min/max softening scale.

    Returns:
        robustness_list (list): tensor of robustness values.
    """

    if isinstance(signals, tuple):
        return formula.robustness(signals, scale=scale)  # calculates
    elif isinstance(signals, list):
        robustness_list = torch.tensor(len(signals))
        for i, signal in enumerate(signals):
            robustness_list[i] = formula.robustness(signal, scale=scale)
        return robustness_list


def create_formula(scenario, max_input):
    if scenario == "stop_sign":
        # Create the formula: [](<>[](0 <= x_stop-x <= 3) & v >= 0)
        # (the car eventually stops before stop sign and never reverses.)
        s1 = Expression("p", max_input[0][0])
        s2 = Expression("v", max_input[1])

        e1 = torch.tensor(0, dtype=torch.float, requires_grad=True)
        e2 = torch.tensor(3.0, dtype=torch.float, requires_grad=True)
        e3 = torch.tensor(0, dtype=torch.float, requires_grad=True)

        formula1 = WSTL.Eventually(
            WSTL.Always(
                WSTL.And(subformula1=(s1 >= e1), subformula2=(s1 <= e2))
            )
        )
        formula = WSTL.Always(
            WSTL.And(subformula1=formula1, subformula2=(s2 >= e3))
        )

    elif scenario == "pedestrian":
        s1 = Expression("p", max_input[0][0][1])
        s2 = Expression("v", max_input[1])
        s_p = Expression("ped", max_input[0][0][0])

        e1 = torch.tensor(0, dtype=torch.float, requires_grad=True)
        e_lim = torch.tensor(20, dtype=torch.float, requires_grad=True)

        # # create the formula
        formula_effect = WSTL.And(
            subformula1=WSTL.BoolTrue(s_p), subformula2=(s1 <= e1)
        )
        formula_cause = WSTL.Until(
            subformula1=(s1 <= e1),
            subformula2=WSTL.Negation(WSTL.BoolTrue(s_p)),
        )
        formula_implies = WSTL.Or(
            subformula1=(WSTL.Negation(formula_effect)),
            subformula2=formula_cause,
        )
        formula_speedlim = WSTL.Always(subformula=s2 <= e_lim)
        formula = WSTL.And(
            subformula1=WSTL.Always(subformula=formula_implies),
            subformula2=formula_speedlim,
        )
    return formula


def print_robustness(
    formula: WSTL.WSTL_Formula,
    input_signals: tuple or list,
    preference_data: list,
):
    """
    It prints robustness values in pairs.

    Parameters:
    - formula (WSTL.WSTL_Formula): the formula to evaluate for robustness.
    - input_signals (tuple or list): input signals for the formula.
    - preference_data (list of tuples): Tuples with indices for preferred
                                        and non-preferred signals.

    Prints:
    Robustness values for pairs of preferred and non-preferred signals.
    """
    robustness_list = (
        get_robustness(formula, input_signals, scale=-1).detach().numpy()
    )
    for i, preference_tuple in enumerate(preference_data):
        preferred_rob = robustness_list[preference_tuple[0]]
        non_preferred_rob = robustness_list[preference_tuple[1]]
        print(
            f"Pair {i:^3}: preferred signal: {preferred_rob.item():.4f}",
            f"--- non preffered signal: {non_preferred_rob.item():.4f}",
        )


# --- DATA LOADERS ---
def signal_to_tensor(signals: list, scenario: str, padded=False):
    """
    It transforms signals list into a tuple of tensors
    (or list of tuple of tensors if not padded)
    of suitable form for the scenario.

    Args:
        signals (list): Signals. List of lists or list of arrays.
        scenario (str): Either 'stop_sign' or 'pedestrian'.
        padded (bool): If True, function returns a of tensors
                       with the same signal length.
                       If False (default), it returns a tuple of
                       lists of tensors.

    Returns:
        input_signals (tuple): input signals of the suitable form
        max_signal (tuple): the tuple of the suitable
                            form with the maximum time length.

    Note:
        max_signal is required for weight initialization.
    """

    if scenario not in ["stop_sign", "pedestrian"]:
        raise TypeError(
            f"signal_to_tensor():{scenario} is not a valid scenario."
        )

    no_signals = len(signals[0])

    if padded:
        all_padded_position = torch.Tensor()
        all_padded_v = torch.Tensor()
        if scenario == "pedestrian":
            all_padded_pedestrian = torch.Tensor()
        else:
            None

        max_length = max([len(p) for p in signals[0]])
        for i in range(no_signals):
            _position_tensor = torch.tensor(signals[0][i], requires_grad=False)
            _padded_position = pad_signal(_position_tensor, max_length)
            all_padded_position = torch.cat(
                (all_padded_position, _padded_position), axis=0
            )

            _v_tensor = torch.tensor(signals[1][i], requires_grad=False)
            _padded_v = pad_signal(_v_tensor, max_length)
            all_padded_v = torch.cat((all_padded_v, _padded_v), axis=0)

            if scenario == "pedestrian":
                _pedestrian_tensor = torch.tensor(
                    signals[2][i], requires_grad=False
                )
                _padded_pedestrian = pad_signal(_pedestrian_tensor, max_length)
                all_padded_pedestrian = torch.cat(
                    (all_padded_pedestrian, _padded_pedestrian), axis=0
                )

        input_signals, max_signal = create_input_signals(
            scenario, all_padded_position, all_padded_v, all_padded_pedestrian
        )

    else:
        input_signals = []
        max_time_length = 0
        for i in range(no_signals):
            _position_tensor = (
                torch.tensor(signals[0][i], requires_grad=False)
                .unsqueeze(0)
                .unsqueeze(-1)
            )
            _v_tensor = (
                torch.tensor(signals[1][i], requires_grad=False)
                .unsqueeze(0)
                .unsqueeze(-1)
            )

            if scenario == "pedestrian":
                _pedestrian_tensor = (
                    torch.tensor(signals[2][i], requires_grad=False)
                    .unsqueeze(0)
                    .unsqueeze(-1)
                )
            else:
                _pedestrian_tensor = None

            signal_tuple, _ = create_input_signals(
                scenario, _position_tensor, _v_tensor, _pedestrian_tensor
            )
            input_signals.append(signal_tuple)
            time_length = _position_tensor.shape[1]
            if time_length > max_time_length:
                idx = i
        max_signal = input_signals[idx]

    return input_signals, max_signal


def preprocess_signals(filename: str, scenario: str):
    """
    Signal data loader function.
    It transforms simulation units to [m] and [m/s]
    and transforms signals into predicates of the form f(x).

    Parameters:
    - filename (str): The name of the file containing signal data.
    - scenario (str): The scenario type, either "stop_sign" or "pedestrian".

    Returns:
    tuple: A tuple of signal data, depending on the scenario:
        - For 'stop_sign': (pos, v), where pos is the position signal
                                       and v is the speed signal.
        - For 'pedestrian': (pos, v, ped), where pos is the position,
                            v is the speed, and ped is the pedestrian
                            boolean signal.

    Note:
    - The function reads signal data from the specified file
      and preprocesses it according to the given scenario.
    - Simulation units are converted to [m] and [m/s].
    - For 'stop_sign' scenario, the position signal is adjusted
      to represent the distance to the stop sign.
    - For 'pedestrian' scenario, if the pedestrian is on the road,
      ped value is True, else it is False.
    - The conversion factors and parameters are defined
      based on the specific scenario requirements.

    """

    if scenario not in ["stop_sign", "pedestrian"]:
        raise TypeError(
            f"preprocess_signals():{scenario} is not a valid scenario."
        )

    with open(filename, "rb") as f:
        s_tracks = pickle.load(f)

    if scenario == "stop_sign":
        # Parameters related to the simulation
        dpi = 50  # dpi of videos
        in2m = 0.4  # conversion to m
        stop_correct = 1200  # pixel value of correct position

        # calibration (conversion to [m] and [m/s]). Make it numpy arrays.
        # Initially, position signal was the distance from the origin
        # (leftmost part of the simulation screen),
        # now, we change it to the distance to stop sign.
        pos = (
            (stop_correct - np.array(list(zip(*s_tracks))[0], dtype=object))
            / dpi
            / in2m
        )
        v = np.array(list(zip(*s_tracks))[1], dtype=object) / dpi / in2m
        return pos, v

    elif scenario == "pedestrian":
        # Parameters related to the simulation
        dpi = 693  # dpi of videos
        in2m = 0.2  # conversion to m
        x_stop = 48 * dpi * in2m

        no_pairs = len(s_tracks)
        # calibration (conversilon to [m] abnd [m/s])
        pos = (2 * no_pairs) * [[]]
        v = (2 * no_pairs) * [[]]
        ped = (2 * no_pairs) * [[]]

        for i in range(no_pairs):
            pos[i] = (s_tracks[i][0][0] - x_stop) / dpi / in2m
            pos[i + no_pairs] = (s_tracks[i][1][0] - x_stop) / dpi / in2m
            v[i] = (s_tracks[i][0][2]) / dpi / in2m

            # forgot to adjust initial velocity during simulation.
            # this line is to prevent velocity jump.
            s_tracks[i][1][2][0] = s_tracks[i][1][2][1]

            v[i + no_pairs] = (s_tracks[i][1][2]) / dpi / in2m
            ped[i] = -1 * np.ones(
                max(s_tracks[i][0].shape[1], s_tracks[i][1].shape[1])
            )
            ped[i][: s_tracks[i][2].shape[1]] = 1
            ped[i + no_pairs] = ped[i]
        return pos, v, ped


def preference_loader(filename: str, scenario: str):
    """
    Loads preferences from the file.

    Parameters:
    - filename (str): The name of the file containing preference data.
    - scenario (str): The scenario type, either "stop_sign" or "pedestrian".

    Returns:
    raw_preference_data (list): List of lists. Each inner list is of the form
                        [preferred_signal_index, non_preferred_signal_index].
    """
    if scenario not in ["stop_sign", "pedestrian"]:
        raise TypeError(
            f"preference_loader(): {scenario} is not a valid scenario."
        )

    with open(filename, "rb") as f:
        raw_preference_data = pickle.load(f)

    if scenario == "stop_sign":
        return raw_preference_data[0]
    else:
        return raw_preference_data


def generate_preference_data(
    formula: WSTL.WSTL_Formula,
    signals: tuple or list,
    scale=-1,
    human_reference: list = None,
    seed: int = None,
):
    """
    Synthetic preference data generator. Used in landscape analysis.
    It returns a formula and a preference data such that
    the accuracy of the pref. set with the weights in the formula is 100%.
    inputs:
        formula: WSTL formula
        signals: set of signals. it is either tuples of tensors
                                or list of tuples of tensors.
        scale: to be used in softmin/softmax for robustness computation.
        human_reference: a reference for synthetic data to be produced.
                         if not none, function takes the same pairs.
    outputs:
        formula: WSTL formula with new weights.
        preference_data: preference data with 100% accuracy
                         given the new weights.

    Remark:
    1. the robustness distance between signals in one pair
       should be greater than a threshold.
    """
    # if human reference is given, use that set of pairs
    # but change preferences accordingly.
    # else create a random set of pairs.
    if seed is not None:
        torch.manual_seed(seed)
    if human_reference:
        raw_preference_data = human_reference
    else:
        raw_preference_data = np.array(range(len(signals[1])))
        # raw_preference_data = np.array(range(len(signals[1][0])))
        random.shuffle(raw_preference_data)
        raw_preference_data = raw_preference_data.reshape(-1, 2).tolist()

    # number of pairs permitted to have close robustness values.
    max_close_robs = 50
    no_close_robs = max_close_robs
    preference_data = 50 * [[]]
    while no_close_robs >= max_close_robs:
        formula.set_weights(signals, random=True, scale=scale)
        robustness_list = get_robustness(formula, signals, scale)
        threshold = (
            0.05 * (max(robustness_list) - min(robustness_list)).item()
        )  # return this later.

        no_close_robs = 0
        for i, pref_tuple in enumerate(raw_preference_data):
            preferred_robustness = (
                robustness_list[pref_tuple[0]].detach().flatten().numpy()
            )
            non_preferred_robustness = (
                robustness_list[pref_tuple[1]].detach().flatten().numpy()
            )
            if preferred_robustness - non_preferred_robustness > 0:
                preference_data[i] = [pref_tuple[0], pref_tuple[1]]
            elif preferred_robustness - non_preferred_robustness < 0:
                preference_data[i] = [pref_tuple[1], pref_tuple[0]]
            abs_dif = np.abs(preferred_robustness - non_preferred_robustness)
            if abs_dif < threshold:
                no_close_robs += 1
                if no_close_robs >= max_close_robs:
                    break

    return formula, preference_data


# --- STATS FUNCTIONS ---
def compute_accuracy(
    formula: WSTL.WSTL_Formula, signals: tuple or list, preferences: list
):
    """
    Compute accuracy of the given WSTL formula on the given preferece data.
    Accuracy is the ratio of correctly ordered pairs over all pairs.

    Parameters:
    - formula (WSTL.WSTL_Formula): The WSTL formula with weights attribute.
    - signals (tuple or list): Input signals.
    - preferences (list): List of indices of the form
                          [[i0, j0], [i1, j1], ..., [iN, jN]].
      For each pair in preference data, the first signal index refers
      to the preferred signal.

    Returns:
    - accuracy (float): The ratio of correctly ordered pairs over all pairs.
                        Accuracy is a scalar between 0 and 1 (both included).
    - accuracy_list (numpy.ndarray): Array with 1s for correct orders
                                     and 0s for incorrect orders.
    """

    correct_orders = 0
    total_length = len(preferences)

    robust_list = get_robustness(formula, signals, scale=-1).detach().numpy()
    accuracy_list = np.zeros(total_length)
    for i, preference_tuple in enumerate(preferences):
        preferred_signal = robust_list[preference_tuple[0]]
        non_preferred_signal = robust_list[preference_tuple[1]]

        if preferred_signal > non_preferred_signal:
            correct_orders += 1
            accuracy_list[i] = 1

    accuracy = correct_orders / total_length
    return accuracy, accuracy_list


def check_preference_consistency(filename: str):
    """
    Computes the consistency of the preferences in the file.

    Parameters:
    - filename (str): The name of the file containing preference data.

    Returns:
    float: The consistency ratio (percentage) of preferences.

    Notes:
    - It only works for the stop_sign scenario.
    """
    with open(filename, "rb") as f:
        raw_preference_data = pickle.load(f)
    consistency = sum(
        [
            raw_preference_data[0][i] == raw_preference_data[1][i]
            for i in range(len(raw_preference_data[0]))
        ]
    )
    return consistency / len(raw_preference_data[0])


# --- INITIALIZERS ---
def noisy_weight_initialization(
    formula: WSTL.WSTL_Formula, noise_variance: float
):
    """
    Adds random noise to ground truth weights
    and returns the modified formula.

    Parameters:
    - formula (WSTL.WSTL_Formula): The original formula with
                                   ground truth weights.
    - noise_variance (float): The variance of the random noise to be added.

    Returns:
    Formula: The formula with weights modified by adding random noise.

    Notes:
    - We use `clamp_` to ensure that the weights stay in the positive quadrant.
    """
    formula_copy = copy.deepcopy(formula)
    w_temp = copy.deepcopy(formula.weights)
    with torch.no_grad():
        for p in w_temp.parameters():
            p.add_((noise_variance**0.5) * torch.randn(p.shape))
            p.data.clamp_(0.001)
    formula_copy.weights = w_temp
    return formula_copy


def train_test_split(preference_data_size: int, trainining_data_size: int):
    """
    Returns a random train-test data split in the form of indices.

    Parameters:
    - preference_data_size (int): size of the preference data
    - training_data_size (int): size of the intended training data

    Returns:
    - training_indices (list): indices for the preference data
                               that is in the training set
    - test_indices (list): indices for the preference data
                           that is in the test set
    """
    index_list = np.arange(preference_data_size)
    random.shuffle(index_list)
    training_indices = index_list[:trainining_data_size]
    test_indices = index_list[trainining_data_size:]
    return training_indices, test_indices
