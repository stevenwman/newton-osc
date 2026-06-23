"""Minimal config exports for the vendored FastSAC closure."""
from jax_rl.configs.networks_config import EncoderConfig, PolicyHeadConfig
from jax_rl.configs.fast_sac_config import FastSACConfig

__all__ = ["EncoderConfig", "PolicyHeadConfig", "FastSACConfig"]
