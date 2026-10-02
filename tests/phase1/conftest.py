"""Pytest fixtures and configuration."""

import pytest

from agentic_sentiment.phase1.config import (
    DataConfig,
    ModelConfig,
    TrainingConfig,
    get_default_config,
)


@pytest.fixture
def data_config():
    return DataConfig()


@pytest.fixture
def model_config():
    return ModelConfig()


@pytest.fixture
def training_config():
    return TrainingConfig()


@pytest.fixture
def default_config():
    return get_default_config()
