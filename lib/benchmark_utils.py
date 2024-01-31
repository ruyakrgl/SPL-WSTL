"""
Utilities for the benchmark models used in the paper.
"""

import os
import sys
import torch
from scipy.fft import fft2

torch.manual_seed(0)

curr_dir = os.getcwd()
sys.path.insert(0, f"{curr_dir}/lib")

from utils import *  # NOQA


def compute_BT(preferece_pair: list, w: torch.Tensor, M: float):
    """
    Computes Bradley-Terry preference value for a pair of signals.

    Parameters:
    - preference_pair (list): List containing two torch.Tensor instances
                              representing signals for pairwise comparison.
    - w (torch.Tensor): Weight vector.
    - M (float): Scaling parameter for the exponential part of
                 the Bradley-Terry model.

    Returns:
    BT_value (torch.Tensor): Bradley-Terry preference value for
                             the given pair of signals
    """
    BT_value = torch.exp(M * torch.dot(w, preferece_pair[0])) / (
        torch.exp(M * torch.dot(w, preferece_pair[0]))
        + torch.exp(M * torch.dot(w, preferece_pair[1]))
    )  # Bradley-Terry preference model
    return BT_value


def get_fft_features(
    benchmark: str,
    scenario: str,
    inputs: list,
    preference_data: list,
    bin_size: int = 5,
):

    input_signals, max_input = signal_to_tensor(
        inputs, scenario=scenario, padded=True
    )
    phi = create_formula(scenario, max_input)
    phi.set_weights(max_input, random=False)
    robs = phi.robustness(input_signals, scale=-1)

    # Pad signals to the same time length, then take their FFTs.
    no_signals = len(inputs[0])
    no_pairs = len(preference_data)

    fft_signals = no_signals * [[]]
    max_length = max([k.shape[0] for k in inputs[0]])
    for i in range(no_signals):
        pos_padded = np.pad(
            inputs[0][i], (0, max_length - inputs[0][i].shape[0]), "edge"
        )
        if scenario == "stop_sign":
            v_padded = np.pad(
                inputs[1][i], (0, max_length - inputs[1][i].shape[0]), "edge"
            )
            fft_array = np.abs(fft2([pos_padded, v_padded])).flatten()
            fft_sum = [
                sum(fft_array[i : i + bin_size])
                for i in range(0, len(fft_array), bin_size)
            ]

            final_features = fft_sum + [robs[i].item()]
            fft_signals[i] = torch.tensor(
                final_features, dtype=torch.float64
            ).flatten()

        elif scenario == "pedestrian":
            v_padded = np.pad(
                20 - inputs[1][i],
                (0, max_length - inputs[1][i].shape[0]),
                "edge",
            )
            ped_padded = np.pad(
                inputs[2][i], (0, max_length - inputs[2][i].shape[0]), "edge"
            )
            fft_array = np.abs(
                fft2([pos_padded, v_padded, ped_padded])
            ).flatten()
            fft_sum = [
                sum(fft_array[i : i + bin_size])
                for i in range(0, len(fft_array), bin_size)
            ]
            final_features = fft_sum + [robs[i].item()]
            fft_signals[i] = torch.tensor(
                final_features, dtype=torch.float64
            ).flatten()

    if benchmark == "SVM":
        fft_pairs_c0 = torch.zeros((no_pairs, fft_signals[0].shape[0]))
        fft_pairs_c1 = torch.zeros((no_pairs, fft_signals[0].shape[0]))

        for k, p_tuple in enumerate(preference_data):
            fft_pairs_c0[k, :] = fft_signals[p_tuple[0]].flatten().reshape(
                1, -1
            ) - fft_signals[p_tuple[1]].flatten().reshape(1, -1)
            fft_pairs_c1[k, :] = fft_signals[p_tuple[1]].flatten().reshape(
                1, -1
            ) - fft_signals[p_tuple[0]].flatten().reshape(1, -1)
        return fft_pairs_c0, fft_pairs_c1
    elif benchmark == "BT":
        fft_pairs = torch.zeros(
            no_pairs, 2, fft_signals[0].shape[0], dtype=torch.float64
        )
        for k, p_tuple in enumerate(preference_data):
            fft_pairs[k, :, :] = torch.cat(
                (
                    fft_signals[p_tuple[0]].flatten().reshape(1, -1),
                    fft_signals[p_tuple[1]].flatten().reshape(1, -1),
                ),
                axis=0,
            )
        return fft_pairs


def compute_loss(w: torch.Tensor, train_data: list):
    """
    Computes the loss function given a set of pairwise comparisons
    using the Bradley-Terry preference model.

    Parameters:
    - w (torch.Tensor): Weight vector.
    - train_data (list): List of pairwise comparisons

    Returns:
    loss (torch.Tensor): Loss value computed using
                         the Bradley-Terry preference model.

    Notes:
    - Parameter M is needed to make the exponential part of p_0 computable.
       Otherwise, exp() goes to infinity.
    - The log(1+x) term in the loss function is used to avoid log(0)
      and infinity. An additional value is added to the loss (N*log(2))
      where N is the number of pairwise comparisons to make the final loss 0.
    """

    p_0 = torch.zeros(len(train_data))
    M = 0.001
    for i, data in enumerate(train_data):
        p_0[i] = compute_BT(data, w, M)
    loss = len(train_data) * torch.log(torch.tensor(2)) - torch.sum(
        torch.log(1 + p_0)
    )
    return loss


def decide_preference(w, data):
    """
    Computes preference accuracy over a dataset as a percentage.

    Parameters:
    - w (torch.Tensor): Weight vector.
    - data (list): List of pairwise comparisons, where each element is
                   a tuple of two torch.Tensor instances representing signals.

    Returns:
    float: Preference accuracy over the dataset as a percentage.
    """

    correct_preferences = 0
    M = 0.001
    for d in data:
        p_0 = compute_BT(d, w, M)
        if p_0 > 0.5:
            correct_preferences += 1
    accuracy = correct_preferences / len(data)
    return accuracy


def BT_learn(
    preference_data: list,
    learn_method: str = "SGD",
    learning_rate: float = 0.1,
    iters: int = 10000,
):
    """
    It learns weights for the Bradley-Terry preference model.

    Parameters:
    - preference_data (list): List of pairwise comparisons,
                              where each element is a tuple of
                              two torch.Tensor instances representing signals.
    - learn_method (str): Learning method, either 'SGD'
                                           or 'Adam'.
    - learning_rate (float): Learning rate for the optimizer.
    - iters (int): Number of iterations for the learning process.

    Returns:
    w (torch.Tensor): the learned weight vector
    loss_hist (list): the loss history during training.

    Notes:
    1. Due to convergence issues, everything is in float64 type.
       No computational issues occur since it is not a big dataset,
       but be aware.

    """
    w = torch.nn.Parameter(
        torch.rand(
            size=(len(preference_data[0][0]),),
            dtype=torch.float64,
            requires_grad=True,
        )
    )

    if learn_method == "SGD":
        optimizer = torch.optim.SGD([w], lr=learning_rate)
    elif learn_method == "Adam":
        optimizer = torch.optim.Adam([w], lr=learning_rate)
    else:
        raise ValueError("Not a valid learning method.")

    loss_hist = torch.zeros(iters)  # saves the loss value history.
    for epoch in range(iters):
        optimizer.zero_grad()
        loss = compute_loss(w, preference_data)
        loss.backward()
        optimizer.step()

        loss_hist[epoch] = loss.item()
        # stoppign condition: looks at relative difference
        # between last two loss values.
        if epoch >= 1 and torch.abs(
            (loss_hist[epoch] - loss_hist[epoch - 1]) / loss_hist[epoch]
        ) < 10 ** (-10):
            print(f"stopping condition is met. Returning at epoch {epoch}.")
            break
    return w, loss_hist[:epoch]
