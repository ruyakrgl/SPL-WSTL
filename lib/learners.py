"""
This module contains functions for learning weight values for WSTL formulas
and analyzing the learning process.

Note:
- The learning process involves optimization with Adam optimizer.

author: Ruya Karagulle
date: 2023
"""

import numpy as np
import os
import sys
import copy
import random
import torch

curr_dir = os.getcwd()
sys.path.insert(0, f"{curr_dir}/lib")

import utils  # NOQA
import WSTL  # NOQA


def sigmoid_func(
    preferred_robustness: torch.Tensor,
    non_preferred_robustness: torch.Tensor,
    shift: float,
    M: int,
):
    """
    Sigmoid function aproximation to make indicator function differentiable.

    Parameters:
    - preferred_robustness (torch.Tensor): WSTL robustness value of
                                           the preferred signal.
    - non_preferred_robustness (torch.Tensor): WSTL robustness value of the
                                               non-preferred signal.
    - sh (float): Epsilon shift for adjusting the condition
                  on equality of robustness values.
    - M (float): A large number used in the sigmoid function approximation.

    Returns:
    sigmoid_value (torch.Tensor: Sigmoid function approximation of the
                    indicator function with the condition
                    "preferred_robustness > non_preferred_robustness".
                    - When preferred_robustness > non_preferred_robustness,
                        sigm_exponential -> +inf and sigm_func -> 0.
                    - When preferred_robustness < non_preferred_robustness,
                        sigm_exponential -> 0 and sigm_func -> 1.

    Notes:
    - epsilon_shift adjusts the condition on equality of robustness values.
    - The while loop handles infinity adjustment to prevent NaN gradients.
    - If you need larger M values and/or more precision,
      consider changing the data type to float64.
    """

    epsilon_shift = shift
    sigm_exponential = torch.exp(
        M * (preferred_robustness - non_preferred_robustness - epsilon_shift)
    )

    # infinity adjustment
    while torch.isinf(sigm_exponential) or torch.isinf(-sigm_exponential):
        M = 0.7 * M
        sigm_exponential = torch.exp(
            M
            * (preferred_robustness - non_preferred_robustness - epsilon_shift)
        )

    sigmoid_value = 1 / (1 + sigm_exponential)
    return sigmoid_value


def compute_loss(
    formula: WSTL.WSTL_Formula,
    signals: tuple or list,
    preferences: list,
    initial_norm: float,
    scale: float,
    M: int,
):
    """
    Computes loss function.

    Parameters:
    - formula (WSTL.WSTL_Formula): WSTL formula.
    - signals (tuple or list): Signal set.
    - preferences (list): List of preference pairs, each containing indices
                          [preferred_signal, non_preferred_signal].
    - initial_norm (float): Norm of initial weight valeus.
    - scale (float): Min/max softening scale for robustness computation.
    - M (float): Large number used in the sigmoid function approximation.

    Returns:
    torch.Tensor: The computed loss value.
    """

    robustness_list = utils.get_robustness(formula, signals, scale)
    shift = 0.01 * (max(robustness_list) - min(robustness_list))
    loss = 0
    for preference_tuple in preferences:
        loss = loss + sigmoid_func(
            robustness_list[preference_tuple[0]],
            robustness_list[preference_tuple[1]],
            shift,
            M,
        )

    loss = loss + 0.5 * torch.log(
        1
        + torch.exp(
            0.5
            * (
                torch.sqrt(
                    sum(p.pow(2.0).sum() for p in formula.weights.parameters())
                )
                - np.sqrt(initial_norm)
            )
            ** 2
        )
    )
    return loss


def get_learning_settings(**kwargs):
    """
    Gets learning settings based on provided keyword arguments.

    Parameters:
    **kwargs: Keyword arguments for learning settings.

    Returns:
    list: List containing the following learning settings:
        - learning_rate (float): Learning rate for optimization.
                                 Default is 1e-5.
        - scale (float): Min/max softening scale for robustness computation.
                         Default is -1.
        - stopping_condition (None or int): Number of iterations
                                            for stopping condition.
                                            Default is None.
        - correct_norm_bound (None or float): Bound for regularization term
                                              in the loss function.
                                              Default is None.
        - visualization (bool): Flag for enabling visualization.
                                Default is False.
        - tensorboard_writer (bool
                        or SummaryWriter): Tensorboard writer if tensorboard
                                           is active, else False.
    """

    learning_rate = kwargs.get("learning_rate", 1e-5)
    scale = kwargs.get("scale", -1)
    stopping_condition = kwargs.get("stopping_condition", None)
    correct_norm_bound = kwargs.get("correct_norm", None)
    visualization = kwargs.get("visuals", False)

    tensorboard_active = kwargs.get("tensorboard", False)
    if tensorboard_active:
        from torch.utils.tensorboard import SummaryWriter

        tensorboard_writer = SummaryWriter()
    else:
        tensorboard_writer = False

    return [
        learning_rate,
        scale,
        stopping_condition,
        correct_norm_bound,
        visualization,
        tensorboard_writer,
    ]


def WSTL_learner(
    formula: WSTL.WSTL_Formula,
    input_signals: tuple or list,
    train_preference_data: list,
    test_preference_data: list,
    **kwargs,
):
    """
    Learns weight values using ADAM as optimization method.

    Parameters:
    - formula (WSTL.WSTL_Formula): Initial WSTL formula to be optimized.
    - input_signals (tuple or list): Signal set.
    - train_preference_data (list): List of preference pairs for training,
                                    each containing indices
                                    [preferred_signal, non_preferred_signal].
    - test_preference_data (list): List of preference pairs for testing,
                                   each containing indices
                                   [preferred_signal, non_preferred_signal].
    - **kwargs: Additional keyword arguments for learning settings.
                See get_learning_settings function for details.

    Returns:
    list: List containing the following elements based on
          the specified learning settings:
        - [formula]: The optimized WSTL formula.
        - [formula,
           loss_list,
           train_acc_list,
           test_acc_list]: If visualization is enabled.

           Returns the formula,
           loss history,
           training accuracy history,
           and testing accuracy history.

    Notes:
    - Loss function includes regularization term to prevent overfitting.
    - The learning process can be visualized by enabling
      the visualization flag.
    - Tensorboard is used for visualization if tensorboard flag is specified.
    """

    [
        learning_rate,
        scale,
        stopping_condition,
        correct_norm_bound,
        visualization,
        tensorboard_writer,
    ] = get_learning_settings(**kwargs)

    loss_list = []
    if visualization:
        train_acc_list = []
        test_acc_list = []

    optimizer = torch.optim.Adam(
        formula.weights.parameters(), lr=learning_rate
    )

    if correct_norm_bound is not None:
        initial_norm = correct_norm_bound
    else:
        initial_norm = sum(
            p.pow(2.0).sum() for p in formula.weights.parameters()
        ).item()

    batch_size = 5
    no_batches = int(len(train_preference_data) / batch_size)
    max_epoch = 10000
    M = 1000
    for t in range(max_epoch):
        batch_idx = list(range(len(train_preference_data)))
        random.shuffle(batch_idx)
        for i in range(no_batches):
            batch_train_data = [
                train_preference_data[bid]
                for bid in batch_idx[(i * batch_size) : (i + 1) * batch_size]
            ]
            loss = compute_loss(
                formula,
                input_signals,
                batch_train_data,
                initial_norm,
                scale,
                M,
            )

            loss.backward(retain_graph=True)
            optimizer.step()
            for p in formula.weights.parameters():
                p.data.clamp_(1e-5)
            formula.update_weights()
            optimizer.zero_grad()

        loss_sum = compute_loss(
            formula,
            input_signals,
            train_preference_data,
            initial_norm,
            scale,
            M,
        )

        if t % 100 == 0:
            print(
                f"Learning status: {100*t/max_epoch}% complete.",
                f"==== Loss:{loss_sum.item():^30}",
            )

        loss_list.append(loss_sum.item())

        if stopping_condition is not None:
            if len(loss_list) > 2:
                if abs(loss_list[-2] - loss_list[-1]) <= stopping_condition:
                    break

        if visualization:
            train_acc_list.append(
                utils.compute_accuracy(
                    formula, input_signals, train_preference_data
                )[0]
            )
            test_acc_list.append(
                utils.compute_accuracy(
                    formula, input_signals, test_preference_data
                )[0]
            )

            if tensorboard_writer:
                tensorboard_writer.add_scalar("train loss/epoch", loss, t)
                tensorboard_writer.add_scalar(
                    "train accuracy/epoch", train_acc_list[-1], t
                )

    if visualization:
        if tensorboard_writer:
            tensorboard_writer.close()
        return [formula, loss_list, train_acc_list, test_acc_list]
    else:
        return [formula]


def single_instance_learner(
    instance_info: list,
    input_signals: list or tuple,
    inputs_max: tuple,
    formula: WSTL.WSTL_Formula,
    training_preferences: list,
    test_preferences: list,
    learning_settings: dict,
):
    """
    WSTL formula learner initialized based on specified instance information.

    Parameters:
    - instance_info (list): List containing information about the instance,
                            including weight set number
                            and initialization type.
    - input_signals (tuple or list): Signal set.
    - inputs_max (torch.Tensor): Maximum input values
                                 for weight initialization.
    - formula (WSTL.WSTL_Formula): Initial WSTL formula to be optimized.
    - training_preferences (list): List of preference pairs for training,
                                   each containing indices
                                   [preferred_signal, non_preferred_signal].
    - test_preferences (list): List of preference pairs for testing,
                               each containing indices
                               [preferred_signal, non_preferred_signal].
    - learning_settings (dict): Dictionary containing learning settings
                                for WSTL learner.
                                See get_learning_settings function for details.

    Returns:
    dict or tuple: Results of the single instance learner,
                   including loss history,
                             training accuracy history,
                             test accuracy history,
                             or final statistics.
    """

    weight_set_no = instance_info[0]
    init_type = instance_info[1]
    if init_type not in ["STL", "noisy", "random", "given_weights"]:
        raise TypeError(
            f"single_instance_learner(): {init_type} is not a valid instance."
        )

    if init_type == "noisy":
        try:
            noise_level = instance_info[2]
            seed_no = instance_info[3]
        except IndexError:
            raise IndexError("noise_level value is missing.")

    formula_learner = copy.deepcopy(formula)
    if init_type == "STL":
        print("weight initialization with weights = 1.")
        formula_learner.set_weights(inputs_max, random=False)

    elif init_type == "noisy":
        print(f"weight initialization with noise level {noise_level}")
        torch.manual_seed(seed_no)
        formula_learner = copy.deepcopy(
            utils.noisy_weight_initialization(formula, noise_level)
        )
        formula_learner.update_weights()
        while (
            utils.compute_accuracy(
                formula_learner, input_signals, training_preferences
            )
            < 0.8
        ):
            formula_learner = copy.deepcopy(
                utils.noisy_weight_initialization(formula, noise_level)
            )
            formula_learner.update_weights()

    elif init_type == "random":
        print("weight initialization with random weights.")
        formula_learner.set_weights(inputs_max, random=True)

    elif init_type == "given_weights":
        print("weight initialization with given weights.")

    initial_training_accuracy = utils.compute_accuracy(
        formula_learner, input_signals, training_preferences
    )
    initial_test_accuracy = utils.compute_accuracy(
        formula_learner, input_signals, test_preferences
    )

    learner_results = WSTL_learner(
        formula_learner,
        input_signals,
        training_preferences,
        test_preferences,
        **learning_settings,
    )
    formula_learner = learner_results[0]

    final_training_accuracy = utils.compute_accuracy(
        formula_learner, input_signals, training_preferences
    )
    final_test_accuracy = utils.compute_accuracy(
        formula_learner, input_signals, test_preferences
    )

    if len(learner_results) > 1:
        loss_list = learner_results[1]
        training_accuracy_list = learner_results[2]
        test_accuracy_list = learner_results[3]
        return {
            "Iteration": [weight_set_no, init_type],
            "Loss": loss_list,
            "train accuracy": training_accuracy_list,
            "test accuracy": test_accuracy_list,
        }
    else:
        return formula_learner, {
            "Training Set Statistics": [
                initial_training_accuracy,
                final_training_accuracy,
            ],
            "Test Set Statistics": [
                initial_test_accuracy,
                final_test_accuracy,
            ],
        }
