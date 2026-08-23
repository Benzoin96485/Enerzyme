"""Regression tests for UMA flow backbone sync with fairchem GP scatter API."""

from __future__ import annotations

import inspect

from enerzyme.models.esen import flow_umabackbone as fub


def test_flow_forward_uses_scatter_target_not_gp_node_offset():
  src = inspect.getsource(fub._escnmd_flow_forward)
  assert "scatter_target" in src
  assert "gp_node_offset" not in src
  assert "gp_ctx" in src
