"""
This modules contructs a computation graph for WSTL weighted robustness
and performs the computation. It returns the weightd robustness values
of a set of signals with different weight valuations. These weight
valuations can be set as all 1 (STL case) or can be set randomly.

Implementation is inspired from STLCG code base.
See: https://github.com/StanfordASL/stlcg

Author: Ruya Karagulle
Date: September 2022
"""

import torch
import numpy as np

LARGE_NUMBER = 10**6


class Maxish(torch.nn.Module):
    """Define max and softmax function."""

    def __init__(self):
        super(Maxish, self).__init__()

    def forward(self, x, scale):
        """
        Forward pass of the class.

        Args:
            x (torch.Tensor or Expression): signal to take the max over
            scale: scaling factor for softmax.
                   If scale = -1 it is the max function.
            axis (int): axis to take the max on

        Return:
            max_value (torch.Tensor): max value. Dimensions kept same
        """
        if isinstance(x, Expression):
            assert (
                x.value is not None
            ), "Input Expression does not have numerical values"
            x = x.value
        if scale > 0:
            return torch.logsumexp(x * scale, dim=1, keepdim=True) / scale
        else:
            return x.max(dim=1, keepdim=True)[0]


class Minish(torch.nn.Module):
    """
    Define min and softmin function."""

    def __init__(self, name="Minish input"):
        super(Minish, self).__init__()
        self.input_name = name

    def forward(self, x, scale):
        """
        Forward pass of the class.

        Args:
            x (torch.Tensor or Expression): signal to take the min over
            scale: scaling factor for softmax.
                   If scale = -1 it is the min function.
            axis (int): axis to take the min on

        Return:
            min_value (torch.Tensor): min value. Dimensions kept same
        """
        if isinstance(x, Expression):
            assert (
                x.value is not None
            ), "Input Expression does not have numerical values"
            x = x.value
        if scale > 0:
            return -torch.logsumexp(-x * scale, dim=1, keepdim=True) / scale
        else:
            return x.min(dim=1, keepdim=True)[0]


class WSTL_Formula(torch.nn.Module):
    """
    Define an WSTL formula.

    Attributes:
    - weights (torch.nn.ParameterDict): weight valuations associated
                                          with each subformula

    Methods:
    - robustnes: Computes robustness for a given time intance.
    - set_weights: Initializes weight values.
    - update_weights: Updates weight values for sublayers.
    - forward: Computes weighted robustness for given input signals
               and for all weight valuations.
    """

    def __init__(self):
        super(WSTL_Formula, self).__init__()
        self.weights = torch.nn.ParameterDict({})

    def robustness(self, inputs, scale: int = -1):
        """
        Return WSTL weighted robustness value for given input signals
        and for all weight valuation samples at t=0.
        Note that robustness is computed per each time instant.

        Args:
        - inputs (Expression or torch.Tensor or tuple): Input signals
                                                        for robustness.
        - scale (int): Scaling factor for robustness computation.

        Returns:
        - torch.Tensor: WSTL weighted robustness values.
        """
        return self.forward(inputs, scale=scale)[:, 0, :].unsqueeze(1)

    def set_weights(self, inputs, random=False, seed=None, **kwargs):
        """
        Initialize weight values.
        If random = False, it initializes weights at 1.
        If random = True, it initializes weights uniformly
                          random between given range.

        Args:
            inputs (Expression or torch.Tensor or tuple): Input signals.
            random (bool): Flag for random initialization.
            seed (int): Seed for reproducibility.
            **kwargs: Additional keyword arguments.
        """
        if seed is not None:
            torch.manual_seed(seed)
        self.weight_assignment(inputs, random, **kwargs)

    def update_weights(self):
        """
        Update weight values for sublayers.
        When a change in weight values of the main formula is on effect,
        this change needs be executed for sublayers.
        This function takes the main formula,
        and applies the changes on sublayers.
        """
        self.weight_update()

    def forward(formula, inputs, **kwargs):
        """
        Computes weighted robustness for input signals
        and for all weight valuations.

        Args:
        - inputs (Expression or torch.Tensor or tuple): Input signals.
        - **kwargs: Additional keyword arguments.

        Returns:
        - torch.Tensor: Weighted robustness values.
        """
        sc = kwargs.get("scale", -1)
        if isinstance(inputs, Expression):
            assert (
                inputs.value is not None
            ), "Input Expression does not have numerical values"
            return formula.robustness_value(inputs.value, scale=sc)
        elif isinstance(inputs, torch.Tensor):
            return formula.robustness_value(inputs, scale=sc)
        elif isinstance(inputs, tuple):
            return formula.robustness_value(
                convert_to_input_values(inputs), scale=sc
            )
        else:
            raise ValueError("Not a invalid input trace")


class Temporal_Operator(WSTL_Formula):
    """
    Define Temporal operators in the syntax: Always, Eventually.
    Until is defined separately.

    Attributes:
    - interval (Interval): Time interval associated with
                           the temporal operator.
    - subformula (WSTL_Formula): Subformula encapsulated
                                 by the temporal operator.
    - operation (None): Placeholder for the specific temporal operation.

    Methods:
    - weight_assignment: Assigns weight values to temporal operators
                         given a range.
    - weight_update: Updates weights in temporal operators.
    - robustness_value: Computes weighted robustness of temporal operators.
    """

    def __init__(self, subformula, interval=None):
        """
        Initialize a Temporal_Operator.

        Args:
        - subformula (WSTL_Formula): Subformula encapsulated
                                     by the temporal operator.
        - interval (list or None): Time interval associated with
                                   the temporal operator. None for operators
                                   with [0, infinity] interval.
        """
        super(Temporal_Operator, self).__init__()
        self.subformula = subformula
        self.interval = interval
        self.operation = None

    def weight_assignment(self, inputs, random, **kwargs):
        """
        Assign weight values to temporal operators given a range.

        Args:
        - inputs (Expression or torch.Tensor): Input signals.
        - random (bool): Flag for random initialization.
        - **kwargs: Additional keyword arguments.
        """
        sc = kwargs.get("scale", -1)
        self.subformula.weight_assignment(inputs, random, **kwargs)
        for keys in self.subformula.weights.keys():
            self.weights[keys] = self.subformula.weights[keys]
        trace = self.subformula(inputs, scale=sc)
        self.compute_weights(trace.shape[1], random)

    def weight_update(self):
        """Update weights in temporal operators."""
        for keys in self.subformula.weights.keys():
            self.subformula.weights[keys] = self.weights[keys]
        self.subformula.weight_update()

    def robustness_value(self, inputs, **kwargs):
        """
        Compute weighted robustness of temporal operators.

        Args:
        - inputs (Expression or torch.Tensor or tuple): Input signals.
        - **kwargs: Additional keyword arguments.

        Returns:
        - torch.Tensor: Weighted robustness values
        """
        sc = kwargs.get("scale", -1)
        trace = self.subformula(inputs, scale=sc)
        outputs = self.compute_robustness(trace, scale=sc)
        return outputs


class Logic_Operator(WSTL_Formula):
    """
    Defines Logic Operators in the syntax: And, Or.
    """

    def __init__(self, subformula1, subformula2):
        super(Logic_Operator, self).__init__()
        self.subformula1 = subformula1
        self.subformula2 = subformula2
        self.operation = None

    def weight_assignment(self, inputs, random, **kwargs):
        # define weights that come from subformulas
        self.subformula1.weight_assignment(inputs[0], random, **kwargs)
        self.subformula2.weight_assignment(inputs[1], random, **kwargs)
        for keys in self.subformula1.weights.keys():
            self.weights[keys] = self.subformula1.weights[keys]
        for keys in self.subformula2.weights.keys():
            self.weights[keys] = self.subformula2.weights[keys]
        self.compute_weights(random)

    def weight_update(self):
        for keys in self.subformula1.weights.keys():
            self.subformula1.weights[keys] = self.weights[keys]
        for keys in self.subformula2.weights.keys():
            self.subformula2.weights[keys] = self.weights[keys]
        self.subformula1.weight_update()
        self.subformula2.weight_update()

    def robustness_value(self, inputs, **kwargs):
        sc = kwargs.get("scale", -1)

        trace1 = self.subformula1(inputs[0], scale=sc)
        trace2 = self.subformula2(inputs[1], scale=sc)
        trace_length = min(trace1.shape[1], trace2.shape[1])

        trace = torch.cat(
            [trace1[:, :trace_length, :], trace2[:, :trace_length, :]], axis=-1
        )

        outputs = self.compute_robustness(trace, scale=sc)
        return outputs


class Predicate(WSTL_Formula):
    """
    Defines predicates.
    """

    def __init__(self, lhs="x", val="c"):
        """
        Predicates have a signal function and a value to compare.
        """
        super(Predicate, self).__init__()
        self.lhs = lhs
        self.val = val

    def weight_assignment(self, input, random, **kwargs):
        self.weights

    def weight_update(self):
        pass

    def robustness_value(self, inputs, **kwargs):
        scale = kwargs["scale"]
        return self.calculate_robustness(inputs, scale=scale)


class BoolTrue(Predicate):
    """
    Defines Boolean True. Note that Boolean True is not a predicate.
    Comparison value is left undefined.
    """

    def __init__(self, lhs="x"):
        super(BoolTrue, self).__init__()
        assert isinstance(lhs, str) | isinstance(
            lhs, Expression
        ), "Input needs to be a string (input name) or Expression"
        self.lhs = lhs

    def robustness_value(self, trace, **kwargs):
        """
        Robustness of the Boolean True is defined as infinity.
        """
        if isinstance(trace, Expression):
            trace = trace.value
        return LARGE_NUMBER * trace

    def __str__(self):
        lhs_str = self.lhs
        if isinstance(self.lhs, Expression):
            lhs_str = self.lhs.name
        return lhs_str


class LessThan(Predicate):
    """
    lhs <= val where lhs is the signal, and val is the constant.
    lhs can be a string or an Expression
    val can be a float, int, Expression, or tensor. It cannot be a string.
    """

    def __init__(self, lhs="x", val="c"):
        super(LessThan, self).__init__()
        assert isinstance(lhs, str) | isinstance(
            lhs, Expression
        ), "LHS of input needs to be a string (input name) or Expression"
        assert not isinstance(val, str), "value on the rhs cannot be a string"
        self.lhs = lhs
        self.val = val

    def robustness_value(self, trace, **kwargs):
        if isinstance(trace, Expression):
            trace = trace.value
        if isinstance(self.val, Expression):
            return self.val.value - trace
        else:
            return self.val - trace

    def __str__(self):
        lhs_str = self.lhs
        if isinstance(self.lhs, Expression):
            lhs_str = self.lhs.name
        if isinstance(
            self.val, str
        ):  # could be a string if robustness_trace is never called
            return lhs_str + " <= " + self.val
        if isinstance(self.val, Expression):
            return lhs_str + " <= " + self.val.name
        if isinstance(self.val, torch.Tensor):
            return lhs_str + " <= " + tensor_to_str(self.val)
        # if self.value is a single number (e.g., int, or float)
        return lhs_str + " <= " + str(self.val)


class GreaterThan(Predicate):
    """
    lhs >= val where lhs is the signal, and val is the constant.
    lhs can be a string or an Expression
    val can be a float, int, Expression, or tensor. It cannot be a string.
    """

    def __init__(self, lhs="x", val="c"):
        super(GreaterThan, self).__init__()
        assert isinstance(lhs, str) | isinstance(
            lhs, Expression
        ), "LHS of expression needs to be a string (input name) or Expression"
        assert not isinstance(val, str), "value on the rhs cannot be a string"
        self.lhs = lhs
        self.val = val

    def robustness_value(self, trace, **kwargs):
        if isinstance(trace, Expression):
            trace = trace.value
        if isinstance(self.val, Expression):
            return trace - self.val.value
        else:
            return trace - self.val

    def __str__(self):
        lhs_str = self.lhs
        if isinstance(self.lhs, Expression):
            lhs_str = self.lhs.name
        if isinstance(
            self.val, str
        ):  # could be a string if robustness_trace is never called
            return lhs_str + " >= " + self.val
        if isinstance(self.val, Expression):
            return lhs_str + " >= " + self.val.name
        if isinstance(self.val, torch.Tensor):
            return lhs_str + " >= " + tensor_to_str(self.val)
        # if self.value is a single number (e.g., int, or float)
        return lhs_str + " >= " + str(self.val)


class Equal(Predicate):
    def __init__(self, lhs="x", val="c"):
        super(Equal, self).__init__()
        assert isinstance(lhs, str) | isinstance(
            lhs, Expression
        ), "LHS of expression needs to be a string (input name) or Expression"
        assert not isinstance(val, str), "value on the rhs cannot be a string"
        self.lhs = lhs
        self.val = val

    def robustness_value(self, trace, **kwargs):
        if isinstance(trace, Expression):
            trace = trace.value
        if isinstance(self.val, Expression):
            return -torch.abs(trace - self.val.value)
        else:
            return -torch.abs(trace - self.val)

    def __str__(self):
        lhs_str = self.lhs
        if isinstance(self.lhs, Expression):
            lhs_str = self.lhs.name

        if isinstance(
            self.val, str
        ):  # could be a string if robustness_trace is never called
            return lhs_str + " = " + self.val
        if isinstance(self.val, Expression):
            return lhs_str + " = " + self.val.name
        if isinstance(self.val, torch.Tensor):
            return lhs_str + " = " + tensor_to_str(self.val)
        # if self.value is a single number (e.g., int, or float)
        return lhs_str + " = " + str(self.val)


class Negation(WSTL_Formula):
    """
    not Subformula.
    """

    def __init__(self, subformula):
        super(Negation, self).__init__()
        self.subformula = subformula

    def weight_assignment(self, inputs, random, **kwargs):
        self.subformula.weight_assignment(inputs, random, **kwargs)
        self.compute_weights()

    def compute_weights(self):
        pass

    def weight_update(self):
        pass

    def robustness_value(self, inputs, **kwargs):
        sc = kwargs["scale"]
        return -self.subformula(inputs, scale=sc)

    def __str__(self):
        return "¬(" + str(self.subformula) + ")"


class And(Logic_Operator):
    """
    Defines and operator. And operator needs two subformula.
    """

    def __init__(self, subformula1, subformula2):
        super(And, self).__init__(
            subformula1=subformula1, subformula2=subformula2
        )
        self.subformula1 = subformula1
        self.subformula2 = subformula2
        self.operation = Minish()

    def compute_weights(self, random=False, **kwargs):
        if random:
            self.weights[
                "("
                + str(self.subformula1).replace(".", "")
                + ") ∧ ("
                + str(self.subformula2).replace(".", "")
                + ")"
            ] = torch.nn.Parameter(
                0.01
                + 1.49
                * torch.rand(size=(2,), dtype=torch.float, requires_grad=True)
            )
        else:
            self.weights[
                "("
                + str(self.subformula1).replace(".", "")
                + ") ∧ ("
                + str(self.subformula2).replace(".", "")
                + ")"
            ] = torch.nn.Parameter(
                torch.tensor([1] * 2, dtype=torch.float, requires_grad=True)
            )

    def compute_robustness(self, input_, **kwargs):
        sc = kwargs.get("scale", -1)
        output_ = torch.Tensor()
        for i in range(input_.shape[1]):
            output_ = torch.cat(
                (
                    output_,
                    self.operation(
                        self.weights[
                            "("
                            + str(self.subformula1).replace(".", "")
                            + ") ∧ ("
                            + str(self.subformula2).replace(".", "")
                            + ")"
                        ]
                        * input_[:, i, :],
                        sc,
                    )[:, None, :],
                ),
                axis=1,
            )
        return output_

    def __str__(self):
        return (
            "(" + str(self.subformula1) + ") ∧ (" + str(self.subformula2) + ")"
        )


class Or(Logic_Operator):
    """
    Defines and operator. Or operator needs two subformula.
    """

    def __init__(self, subformula1, subformula2):
        super(Or, self).__init__(
            subformula1=subformula1, subformula2=subformula2
        )
        self.subformula1 = subformula1
        self.subformula2 = subformula2
        self.operation = Maxish()

    def compute_weights(self, random=False, **kwargs):
        if random:
            self.weights[
                "("
                + str(self.subformula1).replace(".", "")
                + ") ∨ ("
                + str(self.subformula2).replace(".", "")
                + ")"
            ] = torch.nn.Parameter(
                0.01
                + 1.49
                * torch.rand(size=(2,), dtype=torch.float, requires_grad=True)
            )
        else:
            self.weights[
                "("
                + str(self.subformula1).replace(".", "")
                + ") ∨ ("
                + str(self.subformula2).replace(".", "")
                + ")"
            ] = torch.nn.Parameter(
                torch.tensor([1] * 2, dtype=torch.float, requires_grad=True)
            )

    def compute_robustness(self, input_, **kwargs):
        if self.operation is None:
            raise Exception()
        sc = kwargs.get("scale", -1)

        output_ = torch.Tensor()
        for i in range(input_.shape[1]):
            output_ = torch.cat(
                (
                    output_,
                    self.operation(
                        self.weights[
                            "("
                            + str(self.subformula1).replace(".", "")
                            + ") ∨ ("
                            + str(self.subformula2).replace(".", "")
                            + ")"
                        ]
                        * input_[:, i, :],
                        sc,
                    )[:, None, :],
                ),
                axis=1,
            )
        return output_

    def __str__(self):
        return (
            "(" + str(self.subformula1) + ") ∨ (" + str(self.subformula2) + ")"
        )


class Always(Temporal_Operator):
    """
    Defines Always operator.
    ALways operator needs one subformula and an interval.
    If interval is not defined then it is accepted as [0,inf).
    """

    def __init__(self, subformula, interval=None):
        super(Always, self).__init__(subformula=subformula, interval=interval)
        self.operation = Minish()
        self.oper = "min"

    def compute_weights(self, input_length, random):
        if self.interval is None:
            if random:
                self.weights[
                    "◻ "
                    + str(self.interval).replace(".", "")
                    + "( "
                    + str(self.subformula).replace(".", "")
                    + " )"
                ] = torch.nn.Parameter(
                    0.01
                    + 1.49
                    * torch.rand(
                        size=(input_length,),
                        dtype=torch.float,
                        requires_grad=True,
                    )
                )
            else:
                self.weights[
                    "◻ "
                    + str(self.interval).replace(".", "")
                    + "( "
                    + str(self.subformula).replace(".", "")
                    + " )"
                ] = torch.nn.Parameter(
                    torch.tensor(
                        [1] * input_length,
                        dtype=torch.float,
                        requires_grad=True,
                    )
                )
        else:
            if random:
                self.weights[
                    "◻ "
                    + str(self.interval).replace(".", "")
                    + "( "
                    + str(self.subformula).replace(".", "")
                    + " )"
                ] = torch.nn.Parameter(
                    0.01
                    + 1.49
                    * torch.rand(
                        size=(input_length - self.interval[0],),
                        dtype=torch.float,
                        requires_grad=True,
                    )
                )
            else:
                self.weights[
                    "◻ "
                    + str(self.interval).replace(".", "")
                    + "( "
                    + str(self.subformula).replace(".", "")
                    + " )"
                ] = torch.nn.Parameter(
                    torch.tensor(
                        [1] * (input_length - self.interval[0]),
                        dtype=torch.float,
                        requires_grad=True,
                    )
                )

    def compute_robustness(self, x, **kwargs):
        if self.operation is None:
            raise Exception()
        sc = kwargs.get("scale", -1)

        w = torch.reshape(
            self.weights[
                "◻ "
                + str(self.interval).replace(".", "")
                + "( "
                + str(self.subformula).replace(".", "")
                + " )"
            ],
            (-1, 1),
        )
        if self.interval is None:
            interval = [0, x.shape[1]]
            input_ = x
            output_ = torch.Tensor()
            for i in range(x.shape[1]):
                output_ = torch.cat(
                    (
                        output_,
                        self.operation(
                            w[i + interval[0] : interval[1]]
                            * input_[:, i:, :],
                            sc,
                        ),
                    ),
                    axis=1,
                )
        else:
            input_ = x[:, self.interval[0] :, :]
            output_ = torch.Tensor()
            for i in range(x.shape[1] - self.interval[1]):
                output_ = torch.cat(
                    (
                        output_,
                        self.operation(
                            w[
                                i : i
                                + (self.interval[1] - self.interval[0])
                                + 1
                            ]
                            * input_[
                                :,
                                i : i
                                + (self.interval[1] - self.interval[0])
                                + 1,
                                :,
                            ],
                            sc,
                        ),
                    ),
                    axis=1,
                )
        return output_

    def __str__(self):
        if self.interval is None:
            int = [0, np.inf]
        else:
            int = self.interval
        return "◻ " + str(int) + "( " + str(self.subformula) + " )"


class Eventually(Temporal_Operator):
    """
    Defines Eventually operator.
    Eventually operator needs one subformula and an interval.
    If interval is not defined then it is accepted as [0,inf).
    """

    def __init__(self, subformula, interval=None):
        super(Eventually, self).__init__(
            subformula=subformula, interval=interval
        )
        self.operation = Maxish()
        self.oper = "min"

    def compute_weights(self, input_shape, random):
        if self.interval is None:
            if random:
                self.weights[
                    "♢ "
                    + str(self.interval).replace(".", "")
                    + "( "
                    + str(self.subformula).replace(".", "")
                    + " )"
                ] = torch.nn.Parameter(
                    0.01
                    + 1.49
                    * torch.rand(
                        size=(input_shape,),
                        dtype=torch.float,
                        requires_grad=True,
                    )
                )
            else:
                self.weights[
                    "♢ "
                    + str(self.interval).replace(".", "")
                    + "( "
                    + str(self.subformula).replace(".", "")
                    + " )"
                ] = torch.nn.Parameter(
                    torch.tensor(
                        [1] * input_shape,
                        dtype=torch.float,
                        requires_grad=True,
                    )
                )
        else:
            if random:
                self.weights[
                    "♢ "
                    + str(self.interval).replace(".", "")
                    + "( "
                    + str(self.subformula).replace(".", "")
                    + " )"
                ] = torch.nn.Parameter(
                    0.01
                    + 1.49
                    * torch.rand(
                        size=(input_shape - self.interval[0],),
                        dtype=torch.float,
                        requires_grad=True,
                    )
                )
            else:
                self.weights[
                    "♢ "
                    + str(self.interval).replace(".", "")
                    + "( "
                    + str(self.subformula).replace(".", "")
                    + " )"
                ] = torch.nn.Parameter(
                    torch.tensor(
                        [1] * (input_shape - self.interval[0]),
                        dtype=torch.float,
                        requires_grad=True,
                    )
                )

    def compute_robustness(self, x, **kwargs):
        if self.operation is None:
            raise Exception()
        sc = kwargs.get("scale", -1)

        w = torch.reshape(
            self.weights[
                "♢ "
                + str(self.interval).replace(".", "")
                + "( "
                + str(self.subformula).replace(".", "")
                + " )"
            ],
            (-1, 1),
        )
        if self.interval is None:
            interval = [0, x.shape[1]]
            input_ = x
            output_ = torch.Tensor()
            for i in range(x.shape[1]):
                output_ = torch.cat(
                    (
                        output_,
                        self.operation(
                            w[i + interval[0] : interval[1]]
                            * input_[:, i:, :],
                            sc,
                        ),
                    ),
                    axis=1,
                )
        else:
            input_ = x[:, self.interval[0] :, :]
            output_ = torch.Tensor()
            for i in range(x.shape[1] - self.interval[1]):
                output_ = torch.cat(
                    (
                        output_,
                        self.operation(
                            w[
                                i : i
                                + (self.interval[1] - self.interval[0])
                                + 1
                            ]
                            * input_[
                                :,
                                i : i
                                + (self.interval[1] - self.interval[0])
                                + 1,
                                :,
                            ],
                            sc,
                        ),
                    ),
                    axis=1,
                )
        return output_

    def __str__(self):
        if self.interval is None:
            int = [0, np.inf]
        else:
            int = self.interval
        return "♢ " + str(int) + "( " + str(self.subformula) + " )"


class Until(WSTL_Formula):
    def __init__(self, subformula1, subformula2, interval=None):
        super(Until, self).__init__()
        self.subformula1 = subformula1
        self.subformula2 = subformula2
        self.interval = interval

    def weight_assignment(self, inputs, random, **kwargs):
        sc = kwargs.get("scale", -1)

        self.subformula1.weight_assignment(inputs[0], random)
        self.subformula2.weight_assignment(inputs[1], random)
        for keys in self.subformula1.weights.keys():
            self.weights[keys] = self.subformula1.weights[keys]
        for keys in self.subformula2.weights.keys():
            self.weights[keys] = self.subformula2.weights[keys]

        trace1 = self.subformula1(inputs[0], scale=sc)
        trace2 = self.subformula2(inputs[1], scale=sc)
        trace_length = min(trace1.shape[1], trace2.shape[1])

        return self.compute_weights(trace_length, random)

    def weight_update(self):
        for keys in self.subformula1.weights.keys():
            self.subformula1.weights[keys] = self.weights[keys]
        for keys in self.subformula2.weights.keys():
            self.subformula2.weights[keys] = self.weights[keys]
        self.subformula1.weight_update()
        self.subformula2.weight_update()

    def robustness_value(self, inputs, **kwargs):
        sc = kwargs.get("scale", -1)

        trace1 = self.subformula1(inputs[0], scale=sc)
        trace2 = self.subformula2(inputs[1], scale=sc)

        trace_length = min(trace1.shape[1], trace2.shape[1])
        trace = torch.cat(
            [trace1[:, :trace_length, :], trace2[:, :trace_length, :]], axis=-1
        )

        return self.compute_robustness(trace, scale=sc)

    def compute_weights(self, input_shape, random):
        if self.interval is None:
            if random:
                self.weights[
                    "("
                    + str(self.subformula1).replace(".", "")
                    + ")"
                    + " U "
                    + "("
                    + str(self.subformula2).replace(".", "")
                    + ")"
                ] = torch.nn.Parameter(
                    0.01
                    + 1.49
                    * torch.rand(
                        size=(2, input_shape),
                        dtype=torch.float,
                        requires_grad=True,
                    )
                )
            else:
                self.weights[
                    "("
                    + str(self.subformula1).replace(".", "")
                    + ")"
                    + " U "
                    + "("
                    + str(self.subformula2).replace(".", "")
                    + ")"
                ] = torch.nn.Parameter(
                    torch.tensor(
                        [[1, 1]] * input_shape,
                        dtype=torch.float,
                        requires_grad=True,
                    )
                )
        else:
            if random:
                self.weights[
                    "("
                    + str(self.subformula1).replace(".", "")
                    + ")"
                    + " U "
                    + "("
                    + str(self.subformula2).replace(".", "")
                    + ")"
                ] = torch.nn.Parameter(
                    0.01
                    + 1.49
                    * torch.randint(
                        size=(2, input_shape - self.interval[0]),
                        dtype=torch.float,
                        requires_grad=True,
                    )
                )
            else:
                self.weights[
                    "("
                    + str(self.subformula1).replace(".", "")
                    + ")"
                    + " U "
                    + "("
                    + str(self.subformula2).replace(".", "")
                    + ")"
                ] = torch.nn.Parameter(
                    torch.tensor(
                        [[1, 1]] * (input_shape - self.interval[0]),
                        dtype=torch.float,
                        requires_grad=True,
                    )
                )

    def compute_robustness(self, x, scale=-1):
        assert isinstance(
            self.subformula1, WSTL_Formula
        ), "Subformula1 needs to be an STL formula."
        assert isinstance(
            self.subformula2, WSTL_Formula
        ), "Subformula2 needs to be an STL formula."

        w = torch.reshape(
            self.weights[
                "("
                + str(self.subformula1).replace(".", "")
                + ")"
                + " U "
                + "("
                + str(self.subformula2).replace(".", "")
                + ")"
            ],
            (-1, 2),
        )
        output_ = torch.Tensor()

        mins = Minish()
        maxs = Maxish()

        if self.interval is None:
            for i in range(x.shape[1] - 2):
                internal_trace = torch.Tensor()
                for k in range(i + 1, x.shape[1]):
                    internal_min = mins(x[:, i:k, 0], scale)
                    min_compare = torch.cat(
                        (
                            x[:, k, 1].reshape(x.shape[0], 1, 1),
                            internal_min.reshape(internal_min.shape[0], 1, 1),
                        ),
                        axis=1,
                    )
                    internal_trace = torch.cat(
                        (
                            internal_trace,
                            mins(w[k, :].reshape(-1, 1) * min_compare, scale),
                        ),
                        axis=1,
                    )
                output_ = torch.cat(
                    (output_, maxs(internal_trace, scale)), axis=1
                )
        else:
            for i in range(x.shape[1] - self.interval[1]):
                internal_trace = torch.Tensor()
                for k in range(i + self.interval[0], i + self.interval[1] + 1):
                    internal_min = mins(x[:, i:k, 0], scale)
                    min_compare = torch.cat(
                        (
                            x[:, k, 1].reshape(x.shape[0], 1, 1),
                            internal_min.reshape(internal_min.shape[0], 1, 1),
                        ),
                        axis=1,
                    )
                    internal_trace = torch.cat(
                        (
                            internal_trace,
                            mins(w[i, :].reshape(-1, 1) * min_compare, scale),
                        ),
                        axis=1,
                    )
                output_ = torch.cat(
                    (output_, maxs(internal_trace, scale)), axis=1
                )
        return output_

    def __str__(self):
        if self.interval is None:
            int = [0, np.inf]
        else:
            int = self.interval
        return (
            "( "
            + str(self.subformula1)
            + " )"
            + "U"
            + str(int)
            + "( "
            + str(self.subformula2)
            + " )"
        )


class Expression(torch.nn.Module):
    """
    Wraps a pytorch arithmetic operation, so that we can
    intercept and overload comparison operators.
    Expression allows us to express tensors using
    their names to make it easier to code up and read,
    but also keep track of their numeric values.
    """

    def __init__(self, name, value):
        super(Expression, self).__init__()
        self.name = name
        self.value = value

    def set_name(self, new_name):
        self.name = new_name

    def set_value(self, new_value):
        self.value = new_value

    def __neg__(self):
        return Expression(-self.value)

    def __add__(self, other):
        if isinstance(other, Expression):
            return Expression(
                self.name + "+" + other.name, self.value + other.value
            )
        else:
            return Expression(self.name + "+other", self.value + other)

    def __radd__(self, other):
        return self.__add__(other)
        # No need for the case when "other" is an Expression, since that
        # case will be handled by the regular add

    def __sub__(self, other):
        if isinstance(other, Expression):
            return Expression(
                self.name + "-" + other.name, self.value - other.value
            )
        else:
            return Expression(self.name + "-other", self.value - other)

    def __rsub__(self, other):
        return Expression(other - self.value)
        # No need for the case when "other" is an Expression, since that
        # case will be handled by the regular sub

    def __mul__(self, other):
        if isinstance(other, Expression):
            return Expression(
                self.name + "*" + other.name, self.value * other.value
            )
        else:
            return Expression(self.name + "*other", self.value * other)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(a, b):
        # This is the new form required by Python 3
        numerator = a
        denominator = b
        num_name = "num"
        denom_name = "denom"
        if isinstance(numerator, Expression):
            num_name = numerator.name
            numerator = numerator.value
        if isinstance(denominator, Expression):
            denom_name = denominator.name
            denominator = denominator.value
        return Expression(num_name + "/" + denom_name, numerator / denominator)

    # Comparators
    def __le__(lhs, rhs):
        assert isinstance(lhs, str) | isinstance(
            lhs, Expression
        ), "LHS of LessThan needs to be a string or Expression"
        assert not isinstance(rhs, str), "RHS cannot be a string"
        return LessThan(lhs, rhs)

    def __ge__(lhs, rhs):
        assert isinstance(lhs, str) | isinstance(
            lhs, Expression
        ), "LHS of GreaterThan needs to be a string or Expression"
        assert not isinstance(rhs, str), "RHS cannot be a string"
        return GreaterThan(lhs, rhs)

    def __eq__(lhs, rhs):
        assert isinstance(lhs, str) | isinstance(
            lhs, Expression
        ), "LHS of Equal needs to be a string or Expression"
        assert not isinstance(rhs, str), "RHS cannot be a string"
        return Equal(lhs, rhs)

    def __str__(self):
        return str(self.name)


def tensor_to_str(tensor):
    """
    turn tensor into a string for printing
    """
    device = tensor.device.type
    req_grad = tensor.requires_grad
    if not req_grad:
        return "input"
    tensor = tensor.detach()
    if device == "cuda":
        tensor = tensor.cpu()
    return str(tensor.numpy())


def convert_to_input_values(inputs):
    x_, y_ = inputs
    if isinstance(x_, Expression):
        assert (
            x_.value is not None
        ), "Input Expression does not have numerical values"
        x_ret = x_.value
    elif isinstance(x_, torch.Tensor):
        x_ret = x_
    elif isinstance(x_, tuple):
        x_ret = convert_to_input_values(x_)
    else:
        raise ValueError("First argument is an invalid input trace")

    if isinstance(y_, Expression):
        assert (
            y_.value is not None
        ), "Input Expression does not have numerical values"
        y_ret = y_.value
    elif isinstance(y_, torch.Tensor):
        y_ret = y_
    elif isinstance(y_, tuple):
        y_ret = convert_to_input_values(y_)
    else:
        raise ValueError("Second argument is an invalid input trace")

    return (x_ret, y_ret)
