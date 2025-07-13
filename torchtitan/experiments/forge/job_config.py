# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import asdict, dataclass, field
from typing import Any

from torchtitan.config_manager import (
    ActivationCheckpoint,
    Checkpoint,
    Comm,
    Experimental,
    FaultTolerance,
    Float8,
    LRScheduler,
    Model,
    MX,
    Optimizer,
    Parallelism,
    Training,
)


@dataclass
class ForgeJobConfig:
    model: Model = field(default_factory=Model)
    optimizer: Optimizer = field(default_factory=Optimizer)
    lr_scheduler: LRScheduler = field(default_factory=LRScheduler)
    training: Training = field(default_factory=Training)
    parallelism: Parallelism = field(default_factory=Parallelism)
    checkpoint: Checkpoint = field(default_factory=Checkpoint)
    activation_checkpoint: ActivationCheckpoint = field(
        default_factory=ActivationCheckpoint
    )
    float8: Float8 = field(default_factory=Float8)
    mx: MX = field(default_factory=MX)
    comm: Comm = field(default_factory=Comm)
    fault_tolerance: FaultTolerance = field(default_factory=FaultTolerance)
    experimental: Experimental = field(default_factory=Experimental)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
