"""Tests for configuration dataclasses."""

import pytest
from automlip.config import (
    Config, SystemConfig, DFTConfig, TrainerConfig,
    ActiveLearningConfig, SchedulerConfig,
)


def test_default_config():
    cfg = Config()
    assert cfg.mode == "full"
    assert cfg.trainer.backend == "gap"
    assert cfg.trainer.device == "auto"
    assert cfg.scheduler.backend == "local"
    assert cfg.active.max_iterations == 50


def test_nequip_config():
    cfg = Config(
        trainer=TrainerConfig(
            backend="nequip",
            device="cuda",
            nequip_r_max=6.0,
            nequip_num_layers=5,
        ),
    )
    assert cfg.trainer.backend == "nequip"
    assert cfg.trainer.nequip_r_max == 6.0
    assert cfg.trainer.nequip_num_layers == 5


def test_train_only_mode():
    cfg = Config(
        mode="train_only",
        initial_data_path="/some/data.extxyz",
        system=SystemConfig(elements=["Si"]),
    )
    assert cfg.mode == "train_only"
    assert cfg.initial_data_path == "/some/data.extxyz"


def test_slurm_config():
    cfg = Config(
        scheduler=SchedulerConfig(
            backend="slurm",
            account="proj123",
            cores_per_node=36,
            cores_per_dft_job=12,
        ),
    )
    assert cfg.scheduler.backend == "slurm"
    assert cfg.scheduler.cores_per_dft_job == 12


def test_system_config():
    sc = SystemConfig(
        elements=["B", "N", "C"],
        composition={"B": 0.4, "N": 0.4, "C": 0.2},
        n_atoms=200,
        density=2.1,
    )
    assert len(sc.elements) == 3
    assert abs(sum(sc.composition.values()) - 1.0) < 1e-10
