# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from executorch.backends.arm._passes.scalars_to_attribute_pass import (
    ScalarsToAttributePass,
)
from executorch.backends.arm.test import common


def rewrite(module, inputs):
    return (
        ScalarsToAttributePass()
        .call(torch.export.export(module, inputs).module())
        .graph_module
    )


def subs(graph_module):
    return [
        node
        for node in graph_module.graph.nodes
        if node.op == "call_function" and node.target is torch.ops.aten.sub.Tensor
    ]


class Rsub(torch.nn.Module):
    def __init__(self, alpha):
        super().__init__()
        self.alpha = alpha

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.rsub(x, 1.0, alpha=self.alpha)


alpha_test_data = {
    "default": 1,
    "positive": 2,
    "negative": -2,
    "fractional": 0.5,
}


@common.parametrize("alpha", alpha_test_data)
def test_rsub_keeps_its_alpha(alpha) -> None:
    """The rewrite swaps the operands, and alpha scales the one that moves.

    Either spelling holds a single sub, so only the value separates them.

    """
    module = Rsub(alpha).eval()
    inputs = (torch.tensor([0.7, -0.25, 2.0]),)
    rewritten = rewrite(module, inputs)

    assert len(subs(rewritten)) == 1
    assert subs(rewritten)[0].kwargs.get("alpha", 1) == alpha
    torch.testing.assert_close(rewritten(*inputs), module(*inputs))


class TwoRsubs(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.rsub(x, 1.0, alpha=2) + torch.rsub(x, 2.0, alpha=3)


def test_each_rsub_is_rewritten_once() -> None:
    """The rewrite reads the whole argument list, so running it per converted
    argument would emit one sub per scalar rather than one per rsub.
    """
    module = TwoRsubs().eval()
    inputs = (torch.tensor([0.7, -0.25, 2.0]),)
    rewritten = rewrite(module, inputs)

    assert [node.kwargs.get("alpha", 1) for node in subs(rewritten)] == [2, 3]
    torch.testing.assert_close(rewritten(*inputs), module(*inputs))


class IntRsub(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.rsub(x, 1, alpha=2)


def test_an_all_integer_rsub_is_left_alone() -> None:
    """An integer scalar over an integer tensor converts nothing, so the op
    keeps its own spelling rather than becoming a sub.
    """
    module = IntRsub().eval()
    inputs = (torch.tensor([1, 2, 3], dtype=torch.int32),)
    rewritten = rewrite(module, inputs)

    targets = [
        node.target for node in rewritten.graph.nodes if node.op == "call_function"
    ]
    assert targets == [torch.ops.aten.rsub.Scalar]
    torch.testing.assert_close(rewritten(*inputs), module(*inputs))
