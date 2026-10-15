"""Shared pytest setup: the header names the JAX backend and devices the tests run on."""
import jax


def pytest_report_header(config):
    return f"jax {jax.__version__}: backend {jax.default_backend()}, devices {jax.devices()}"
