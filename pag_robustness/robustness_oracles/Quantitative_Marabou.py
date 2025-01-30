import concurrent.futures
import tempfile

import torch
from maraboupy import Marabou

from pag_robustness.nnet_saver import nnet_exporter


def parallel_worker(model, nnet_file, point, max_radius, step_num):
    """ Worker function to process a single point. """
    radius = 0
    increment = max_radius / 2

    for _ in range(step_num):
        # Verify l_inf ball using the pre-exported .nnet file
        code= verify_l_inf_ball(model, nnet_file, point, radius + increment)
        if code == "unsat":
            radius += increment
        increment /= 2
        # print(increment)
    # print(radius)
    return radius


def quantitative_Marabou(model, step_num, max_radius, points):
    radii = []
    # Export the model to a single temporary .nnet file outside of the worker
    with tempfile.NamedTemporaryFile(suffix=".nnet") as tmpfile:
        nnet_exporter(model, tmpfile.name, points)
        nnet_file = tmpfile.name
        # Create a process pool and pass the file and points to workers
        with concurrent.futures.ProcessPoolExecutor(max_workers=15) as executor:
            futures = [
                executor.submit(parallel_worker, model, nnet_file, point, max_radius, step_num)
                for point in points
            ]

            # Collect results as they complete
            for idx, future in enumerate(futures):
                radius = future.result()
                radii.append(radius)

                if idx % 10 == 0:
                    print(f"Processed {idx} points")

    return torch.tensor(radii)


def verify_l_inf_ball(model, nnet_file, point, max_radius) -> (str, float):
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

    # print("nro")
    network = Marabou.read_nnet(nnet_file)

    # Set the input bounds to define the L_inf ball around the point
    input_vars = network.inputVars[0][0]  # Input variables from the Marabou network

    # We define the L_inf ball by adding constraints to each input variable
    for i in range(len(input_vars)):
        lower_bound = point[i].item() - max_radius
        upper_bound = point[i].item() + max_radius
        # print(input_vars[i])
        # print(network.getInputMinimum(input_vars[i]))
        network.setLowerBound(input_vars[i], max(lower_bound, -0.4242))#network.getInputMinimum(input_vars[i])))
        network.setUpperBound(input_vars[i], min(upper_bound, 2.8215))#network.getInputMaximum(input_vars[i])))
        # network.addInequality([input_vars[i]], [-1], -lower_bound)
        # network.ad
    # Get the current class label from the network by feeding the input point
    output_class = model(point.unsqueeze(0)).argmax().item()
    # print(f"output class: {output_class}")
    # Set output constraints for robustness verification
    output_vars = network.outputVars[0][0]  # Output variables from the Marabou network
    max_var = network.getNewVariable()

    # Add a max constraint to ensure max_var is the maximum of all output variables
    network.addMaxConstraint(set(output_vars), max_var)
    network.addInequality([max_var, output_vars[output_class]], [-1, 1], -10 ** -6)
    # for i in range(len(output_vars)):
    #     if i != output_class:
    #         network.addInequality([output_vars[output_class], output_vars[i]], [1, -1], 0)

    # Perform the verification
    options = Marabou.createOptions(verbosity=0, timeoutInSeconds=60, numWorkers=8)
    code, vals, stats = network.solve(verbose=False, options=options)
    # print("test")
    # print((torch.tensor(list(vals[i] for i in network.inputVars[0][0])) - point).abs().max())
    # print(code)
    return code #, (torch.tensor(list(vals[i] for i in network.inputVars[0][0])) - point).abs().max()
