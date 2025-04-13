import concurrent.futures
import tempfile
import auto_LiRPA
import numpy as np
import torch
from maraboupy import Marabou

from auto_LiRPA import PerturbationLpNorm, BoundedTensor, BoundedModule

# git clone https://github.com/Verified-Intelligence/auto_LiRPA
# cd auto_LiRPA
# pip install .

from pag_robustness.nnet_saver import nnet_exporter
#
#
# def parallel_worker(model, nnet_file, point, max_radius, step_num):
#     """ Worker function to process a single point. """
#     radius = 0
#     increment = max_radius / 2
#
#     for _ in range(step_num):
#         # Verify l_inf ball using the pre-exported .nnet file
#         code= verify_l_inf_ball(model, nnet_file, point, radius + increment)
#         if code == "unsat":
#             radius += increment
#         increment /= 2
#         # print(increment)
#     # print(radius)
#     return radius


def find_robustness_radius(model, point, max_radius=0.1, step_num=5):
    low, high = 0, max_radius
    best_eps = 0

    for _ in range(step_num):
        mid = (low + high) / 2
        perturbation = PerturbationLpNorm(norm=np.inf, eps=mid)
        bounded_input = BoundedTensor(point.unsqueeze(0), perturbation)
        # print("test_11")
        output_bounds = model.compute_bounds(x=(bounded_input,), method="backward")
        # print("test_12")
        # Check if classification changes
        # print(output_bounds)
        if torch.argmax(output_bounds[0]) == torch.argmax(output_bounds[1]):
            best_eps = mid  # Still robust
            low = mid
        else:
            high = mid  # Not robust
    return best_eps


def quantitative_lirpa(model, step_num, max_radius, points):
    radii = []
    # Export the model to a single temporary .nnet file outside of the worker
    # with tempfile.NamedTemporaryFile(suffix=".nnet") as tmpfile:
    #     nnet_exporter(model, tmpfile.name, points)
    #     nnet_file = tmpfile.name
    #     # Create a process pool and pass the file and points to workers
    print(points[1, :].shape)
    b_model = BoundedModule(model, torch.empty((1,784)))

    for idx, point in enumerate(points.cuda()):
        # print("test")
        radius = find_robustness_radius(b_model, point, max_radius, step_num)
        radii.append(radius)
        if idx % 10 == 0:
            print(f"Processed {idx} points")
    # with concurrent.futures.ProcessPoolExecutor(max_workers=15) as executor:
    #     futures = [
    #         executor.submit(find_robustness_radius, BoundedModule(model, torch.empty((1, 3, 32, 32))), point, max_radius, step_num)
    #         for point in points
    #     ]
    #     print("test")
        # Collect results as they complete
        # for idx, future in enumerate(futures):
        #     radius = future.result()
        #     radii.append(radius)
        #
        #     if idx % 10 == 0:
        #         print(f"Processed {idx} points")
    return torch.tensor(radii)

