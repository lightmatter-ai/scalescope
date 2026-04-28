"""Pipeline-parallel modeling.

Provides abstractions for pipeline-parallel execution to estimate the bubble
overhead and total comm time contribution.
"""

from abc import ABC, abstractmethod

from scalescope.stats import CommStats
from scalescope.workload_spec import PipelineParallelVariant


class PipelineParallelStrategy(ABC):
    """Abstract base class for pipeline parallel communication strategies."""

    @abstractmethod
    def forward_comm(self, **kwargs) -> CommStats:
        """Calculates forward pass communication for pipeline parallelism."""
        pass

    @abstractmethod
    def backward_comm(self, **kwargs) -> CommStats:
        """Calculates backward pass communication for pipeline parallelism."""
        pass


class GPipePipelineStrategy(PipelineParallelStrategy):
    """Models the communication overhead of the pipeline bubble."""

    def forward_comm(self, **kwargs) -> CommStats:
        # The pipeline bubble does not add communication overhead to the forward pass.
        return CommStats(0, 0, 0, 0)

    def backward_comm(self, **kwargs) -> CommStats:
        num_stages = kwargs["num_stages"]
        num_microbatches = kwargs["num_microbatches"]
        batch_time = kwargs["batch_time"]

        if num_microbatches <= 0:
            raise ValueError("num_microbatches must be > 0")

        if num_stages <= 1:
            return CommStats(0, 0, 0, 0)

        # Basic GPipe fill-drain bubble:
        #   bubble = (p - 1) * t_micro_stage
        # with batch_time interpreted as stage time over all microbatches:
        #   batch_time = m * t_micro_stage
        # so:
        #   bubble = ((p - 1) / m) * batch_time
        bubble_time = (num_stages - 1) / num_microbatches * batch_time

        # The bubble is a time overhead, not a volume of data transferred.
        # We model it as a communication stat for consistency.
        return CommStats(
            startup_latency=0, transmission_time=bubble_time, comm_vol=0, compute_time=0
        )


def get_pp_strategy(
    variant: PipelineParallelVariant,
) -> PipelineParallelStrategy:
    """Factory to create a pipeline parallel strategy."""
    match variant:
        case PipelineParallelVariant.GPIPE:
            return GPipePipelineStrategy()
        case _:
            raise NotImplementedError(f"PP variant {variant} is not supported.")
