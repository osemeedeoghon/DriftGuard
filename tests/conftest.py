"""Shared pytest configuration.

Streamlit emits a "missing ScriptRunContext" warning for every
``st.session_state`` access outside a running server. Several tests exercise
session-state logic deliberately in bare mode, which produced hundreds of
lines of noise and buried real failures. The warning is expected here, so it
is silenced for the suite rather than worked around in each test.
"""

from __future__ import annotations

import logging

import pytest


@pytest.fixture(autouse=True, scope="session")
def _quiet_streamlit_bare_mode_warnings() -> None:
    logging.getLogger("streamlit.runtime.scriptrunner_utils.script_run_context").setLevel(
        logging.ERROR
    )
