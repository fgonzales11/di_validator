"""Regression coverage for the public API exercised by ``run_book.ipynb``."""

import pytest

import multiconductor as mc


RUN_BOOK_PUBLIC_API = (
    "DFData",
    "FaultSpec",
    "ProfileBinding",
    "ResultSpec",
    "TerminalRef",
    "calc_sc",
    "create_dtoc",
    "create_measurement",
    "create_poly_cost",
    "create_switch",
    "estimate",
    "from_pickle",
    "generate_nminus1_cases",
    "get_equivalent",
    "run_contingency",
    "run_diagnostics",
    "run_opf",
    "run_pf",
    "run_protection",
    "run_timeseries",
    "solve_equivalent",
)


@pytest.mark.parametrize("name", RUN_BOOK_PUBLIC_API)
def test_run_book_public_api_is_callable(name):
    assert callable(getattr(mc, name))
