"""`warm()`: which failures are a deployment starting, and which are a deployment refusing.

The distinction is the whole job, and it is made from the *text of a failed call* rather than
from the deployment's own status -- so both directions have been got wrong in production:

- a freshly created deployment 404s while it loads weights, which was read as permanent, and
  a real roster row was skipped with `Initializing Replica Count: 1` on the record;
- a deployment in `RESOURCE_EXHAUSTED / no available capacity` returns a scaling-shaped error
  and got waited out for the full 900s budget even though it could never start.

These pin the first. The second is a documented limitation of inferring state from a call.
"""

from unittest.mock import MagicMock, patch

import pytest

from src.metaeval.sources.tau2_source import warm

DEDICATED = ("fireworks_ai/accounts/fireworks/models/gemma-4-26b-a4b-it"
             "#accounts/<account>/deployments/meta-eval-gemma-4-26b-a4b-it")
SERVERLESS = "fireworks_ai/accounts/fireworks/models/deepseek-v4-flash"

NOT_FOUND = ("NotFoundError: Fireworks_aiException - "
             '{"error":{"message":"Model not found, inaccessible, and/or not deployed"}}')
SCALING = "DEPLOYMENT_SCALING_UP: deployment is scaling up"


def _completion(*outcomes):
    """A litellm.completion whose calls raise/succeed in the given order."""
    def side_effect(*_a, **_kw):
        o = outcomes[min(len(seen), len(outcomes) - 1)]
        seen.append(o)
        if isinstance(o, str):
            raise RuntimeError(o)
        return MagicMock()
    seen: list = []
    m = MagicMock(side_effect=side_effect)
    m.seen = seen
    return m


@pytest.fixture(autouse=True)
def _no_sleeping():
    """`warm` sleeps 30s between polls. A test that proves patience must not take minutes."""
    with patch("src.metaeval.sources.tau2_source.time.sleep"):
        yield


def test_a_provisioning_deployment_is_waited_for_not_written_off():
    """The bug that cost the gemma-4-26b row.

    Fireworks does not route a `#deployments/<id>` address until a replica registers, so a
    deployment three minutes into loading a 26B model returns exactly the same "Model not
    found" as a typo does.
    """
    c = _completion(NOT_FOUND, NOT_FOUND, None)
    with patch("litellm.completion", c):
        assert warm([DEDICATED], verbose=False) == []
    assert len(c.seen) == 3, "it retried rather than giving up on the first 404"


def test_a_serverless_404_is_still_permanent():
    """`deepseek-v4-flash` returns the same message because the model was *removed*. Retrying
    that for 900s is the failure the `#` gate exists to avoid reintroducing."""
    c = _completion(NOT_FOUND)
    with patch("litellm.completion", c):
        assert warm([SERVERLESS], verbose=False) == [SERVERLESS]
    assert len(c.seen) == 1, "no waiting on a model that does not exist"


def test_scaling_up_is_waited_for_on_either_address():
    for model in (DEDICATED, SERVERLESS):
        c = _completion(SCALING, None)
        with patch("litellm.completion", c):
            assert warm([model], verbose=False) == []
        assert len(c.seen) == 2


def test_an_unrelated_error_is_not_retried():
    """An auth failure or a bad request is not a cold start, and must surface at once."""
    c = _completion("AuthenticationError: invalid or expired token")
    with patch("litellm.completion", c):
        assert warm([DEDICATED], verbose=False) == [DEDICATED]
    assert len(c.seen) == 1


def test_the_budget_bounds_the_wait():
    """A deployment that never comes up must not hang the sweep. This is also the backstop for
    the case `warm` cannot recognise -- `RESOURCE_EXHAUSTED` looks like scaling from here."""
    c = _completion(NOT_FOUND)
    with patch("litellm.completion", c):
        assert warm([DEDICATED], budget_s=0, verbose=False) == [DEDICATED]
    assert len(c.seen) == 1, "budget already spent, so one attempt and out"


def test_a_warm_deployment_costs_one_call():
    c = _completion(None)
    with patch("litellm.completion", c):
        assert warm([DEDICATED], verbose=False) == []
    assert len(c.seen) == 1


def test_models_are_deduplicated_and_ordered():
    """Two roster rows can name one deployment; warming it twice wastes a cold start."""
    c = _completion(None)
    with patch("litellm.completion", c):
        assert warm([DEDICATED, DEDICATED, SERVERLESS], verbose=False) == []
    assert len(c.seen) == 2


def test_the_failures_are_returned_and_the_successes_are_not():
    """`run_evaluators` turns this list into skipped rows, so a partial result must be exact."""
    def side_effect(*_a, model=None, **_kw):
        if model == SERVERLESS:
            raise RuntimeError(NOT_FOUND)
        return MagicMock()
    with patch("litellm.completion", MagicMock(side_effect=side_effect)):
        assert warm([DEDICATED, SERVERLESS], verbose=False) == [SERVERLESS]
