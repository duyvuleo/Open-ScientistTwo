"""Explicit upstream coverage selection for local tests and CI."""

import os


def pytest_addoption(parser):
    parser.addoption("--require-upstreams", metavar="NAMES", default=None,
                     help="Fail on missing upstreams: all, or paperorchestra,mlagentbench,mle-bench")


def pytest_configure(config):
    required = config.getoption("--require-upstreams")
    if required is not None:
        os.environ["SCIENTISTTWO_REQUIRE_UPSTREAMS"] = required
