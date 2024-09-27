import tempfile

import torch
from maraboupy import Marabou
from nnet_saver import nnet_exporter


def quantitative_Marabou(model, step_num, max_radius, points):
    radii = []
    for idx, point in enumerate(points):
        radius = 0
        increment = max_radius / 2
        # print("bek")
        for _ in range(step_num):
            with tempfile.NamedTemporaryFile(suffix=".nnet") as tmpfile:
                nnet_exporter(model, tmpfile.name, points)
                if verify_l_inf_ball(model, tmpfile.name, point, radius + increment) == "unsat":
                    radius += increment
            increment = increment / 2
        radii.append(radius)
    return torch.tensor(radii)


def verify_l_inf_ball(model, nnet_file, point, max_radius):
    """
    Verifies that the class label of a neural network does not change within an L_inf ball
    around each point in the input set.

    Args:
    - model: PyTorch model to verify.
    - step_num: Number of steps for verification iterations.
    - max_radius: Maximum L_inf ball radius to verify around each point.
    - points: Torch Tensor containing the datapoints to verify.

    Returns:
    - results: A dictionary where the key is the point index, and the value is the verification result.
    """
    network = Marabou.read_nnet(nnet_file)

    # Set the input bounds to define the L_inf ball around the point
    input_vars = network.inputVars[0][0]  # Input variables from the Marabou network

    # We define the L_inf ball by adding constraints to each input variable
    for i in range(len(input_vars)):
        lower_bound = point[i].item() - max_radius
        upper_bound = point[i].item() + max_radius
        # print(input_vars[i])
        # print(network.getInputMinimum(input_vars[i]))
        network.setLowerBound(input_vars[i], max(lower_bound, network.getInputMinimum(input_vars[i])))
        network.setUpperBound(input_vars[i], min(upper_bound, network.getInputMaximum(input_vars[i])))
        # network.addInequality([input_vars[i]], [-1], -lower_bound)

    # Get the current class label from the network by feeding the input point
    output_class = model(point.unsqueeze(0)).argmax().item()
    # print(f"output class: {output_class}")
    # Set output constraints for robustness verification
    output_vars = network.outputVars[0][0]  # Output variables from the Marabou network
    max_var = network.getNewVariable()

    # Add a max constraint to ensure max_var is the maximum of all output variables
    network.addMaxConstraint(set(output_vars), max_var)
    network.addInequality([max_var, output_vars[output_class]], [-1, 1], -10**-6)
    # for i in range(len(output_vars)):
    #     if i != output_class:
    #         network.addInequality([output_vars[output_class], output_vars[i]], [1, -1], 0)

    # Perform the verification
    options = Marabou.createOptions(verbosity=0, timeoutInSeconds=60, numWorkers=8)
    code, vals, stats = network.solve(verbose=False, options=options)
    return code
