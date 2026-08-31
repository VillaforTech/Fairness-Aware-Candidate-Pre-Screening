"""Tests for configuration system."""

import tempfile
from pathlib import Path

import pytest


class TestConfig:
    """Tests for Config class."""

    def test_default_config(self):
        """Test default configuration values."""
        from fairness_project.config import Config

        config = Config()

        assert config.seed == 42
        assert config.verbose is True
        assert config.model.model_type == "xgb"
        assert config.model.random_state == 42

    def test_config_dataclass_fields(self):
        """Test that config has expected fields."""
        from fairness_project.config import Config

        config = Config()

        assert hasattr(config, "data")
        assert hasattr(config, "model")
        assert hasattr(config, "fairness")
        assert hasattr(config, "output")


class TestSeedManagement:
    """Tests for seed management."""

    def test_set_seed(self):
        """Test that set_seed changes numpy random state."""
        import numpy as np

        from fairness_project.config import set_seed

        set_seed(42)
        val1 = np.random.rand()

        set_seed(42)
        val2 = np.random.rand()

        assert val1 == val2

    def test_get_random_state_from_config(self):
        """Test getting random state from config."""
        from fairness_project.config import Config, get_random_state

        config = Config()
        config.seed = 123

        assert get_random_state(config) == 123

    def test_get_random_state_default(self):
        """Test default random state."""
        from fairness_project.config import get_random_state

        assert get_random_state(None) == 42


class TestConfigLoading:
    """Tests for config loading/saving."""

    def test_config_from_dict(self):
        """Test creating config from dictionary."""
        from fairness_project.config import config_from_dict

        data = {
            "seed": 123,
            "model": {"model_type": "lr", "random_state": 123},
        }

        config = config_from_dict(data)

        assert config.seed == 123
        assert config.model.model_type == "lr"
        assert config.model.random_state == 123

    @pytest.mark.parametrize(
        "payload",
        [
            {"unknown": True},
            {"model": {"imaginary_parameter": 1}},
            {"schema_version": "1.0"},
        ],
    )
    def test_config_rejects_unknown_or_unsupported_keys(self, payload):
        from fairness_project.config import config_from_dict

        with pytest.raises(ValueError):
            config_from_dict(payload)

    def test_load_yaml_config(self):
        """Test loading config from YAML file."""
        from fairness_project.config import load_yaml_config

        # Create temp YAML file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("seed: 999\nmodel:\n  model_type: rf\n")
            temp_path = f.name

        try:
            data = load_yaml_config(temp_path)

            assert data["seed"] == 999
            assert data["model"]["model_type"] == "rf"
        finally:
            Path(temp_path).unlink()

    def test_load_config_returns_default_when_no_path(self):
        """Test that load_config returns default when path is None."""
        from fairness_project.config import Config, load_config

        config = load_config(None)

        assert isinstance(config, Config)
        assert config.seed == 42

    def test_save_and_load_config_roundtrip(self):
        """Test saving and loading config produces same values."""
        from fairness_project.config import Config, load_config, save_config

        original = Config()
        original.seed = 999
        original.model.model_type = "lr"

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test_config.yaml"
            save_config(original, path)
            loaded = load_config(path)

            assert loaded.seed == 999
            assert loaded.model.model_type == "lr"


class TestGlobalConfig:
    """Tests for global config management."""

    def test_get_config_returns_default(self):
        """Test get_config returns default config."""
        from fairness_project.config import Config, get_config

        config = get_config()

        assert isinstance(config, Config)

    def test_set_and_get_config(self):
        """Test setting and getting global config."""
        from fairness_project.config import Config, get_config, set_config

        custom = Config()
        custom.seed = 777

        set_config(custom)
        retrieved = get_config()

        assert retrieved.seed == 777

    def test_init_config_with_seed_override(self):
        """Test init_config with seed override."""
        from fairness_project.config import init_config

        config = init_config(seed=555)

        assert config.seed == 555
        assert config.model.random_state == 555
