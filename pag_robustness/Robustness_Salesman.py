import bisect

import torch
from matplotlib import pyplot as plt


class Robustness_Salesman:
    """
    This class takes your confidence and leaves you with free guarantees.
    """

    def __init__(self, sample_confidences: torch.Tensor, sample_robustnesses: torch.Tensor):
        assert len(sample_confidences) == len(sample_robustnesses)
        self.n = len(sample_confidences)
        self.max_kappa = sorted(sample_confidences)[self._max_kappa_idx()]
        print(self.max_kappa)
        self.guarantee_dict = {}

        while True:
            # find the index of the minimum robustness
            _, ind = torch.min(sample_robustnesses, dim=0)
            kappa = sample_confidences[ind]
            self.guarantee_dict.update({min(kappa, self.max_kappa): sample_robustnesses[ind]})
            if kappa >= self.max_kappa:
                break
            remaining = sample_confidences > kappa
            sample_confidences = sample_confidences[remaining]
            sample_robustnesses = sample_robustnesses[remaining]

        self.sorted_keys = sorted(self.guarantee_dict.keys())

    def minimum_robustness(self, confidence):
        if confidence > self.max_kappa:
            confidence = self.max_kappa  # it needs to be a tiny bit smaller than the max for numeric reasons

        pos = bisect.bisect_left(self.sorted_keys, confidence)
        key = self.sorted_keys[pos]
        return self.guarantee_dict[key]

    def _max_kappa_idx(self) -> int:
        """
        TODO this is not yet correct
        """
        return int(self.n * 0.83)


if __name__ == "__main__":
    n = 500000

    sample_robustnesses = torch.randn(size=(n, ))
    sample_confidences = sample_robustnesses + torch.randn(size=(n, )) * 0.8
    print(f"median confidence: {sample_confidences.median()}")
    rob = Robustness_Salesman(sample_confidences, sample_robustnesses)

    x = sample_confidences.numpy()
    y = sample_robustnesses.numpy()
    print(rob.minimum_robustness(0.9))
    print(rob.guarantee_dict)

    # Create the scatter plot
    plt.scatter(x, y, color='blue', alpha=0.05)

    robustnesses = list(rob.guarantee_dict.values())
    confidences = list(rob.guarantee_dict.keys())
    # Create a scatter plot with robustness on x-axis and confidence on y-axis
    plt.scatter(confidences, robustnesses, color='red', alpha=0.7)

    # Label the axes
    plt.ylabel('Robustness')
    plt.xlabel('Confidence')



    # Add a title
    plt.title('Robustness vs Confidence')

    # Display the plot
    plt.show()

