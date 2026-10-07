#!/usr/bin/env python3
"""
Jetson Orin Nano hardware profile simulator.

Models the CPU thread constraints, memory limits, and thermal throttling
behavior of a Jetson Orin Nano so the detection pipeline can be benchmarked
in a simulation environment without physical hardware.

Usage:
    from jetson_sim import JetsonProfile
    profile = JetsonProfile("jetson_sim/config.yaml")
    threads = profile.cpu_threads_for_inference()  # e.g. 4
    throttle = profile.thermal_factor(elapsed_seconds=45)  # e.g. 1.15
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml


@dataclass
class CPUSpec:
    cores: int = 6
    threads: int = 6
    max_freq_mhz: int = 2000
    sim_cpu_threads: int = 4


@dataclass
class GPUSpec:
    cuda_cores: int = 1024
    tensor_cores: int = 32
    memory_mb: int = 8192
    memory_bandwidth_gb_s: float = 51.2
    max_freq_mhz: int = 1300
    tdp_w: float = 15.0


@dataclass
class ThermalSpec:
    throttle_temp_c: int = 80
    sustained_seconds: int = 30
    max_throttle_factor: float = 1.5


@dataclass
class SimSettings:
    limit_threads: bool = True
    enable_throttle: bool = True
    memory_limit_mb: int = 6144
    inference_width: int = 640
    inference_height: int = 640
    target_fps: float = 25.0
    target_latency_ms: float = 35.0


class JetsonProfile:
    """Read-only hardware profile loaded from config.yaml."""

    def __init__(self, config_path: Optional[str] = None):
        if config_path is None:
            config_path = str(Path(__file__).resolve().parent / "config.yaml")
        with open(config_path) as f:
            cfg = yaml.safe_load(f)

        self.name: str = cfg["platform"]["name"]
        self.soc: str = cfg["platform"]["soc"]
        self.architecture: str = cfg["platform"]["architecture"]

        cpu = cfg.get("cpu", {})
        self.cpu = CPUSpec(
            cores=cpu.get("cores", 6),
            threads=cpu.get("threads", 6),
            max_freq_mhz=cpu.get("max_freq_mhz", 2000),
            sim_cpu_threads=cpu.get("sim_cpu_threads", 4),
        )

        gpu = cfg.get("gpu", {})
        self.gpu = GPUSpec(
            cuda_cores=gpu.get("cuda_cores", 1024),
            tensor_cores=gpu.get("tensor_cores", 32),
            memory_mb=gpu.get("memory_mb", 8192),
            memory_bandwidth_gb_s=gpu.get("memory_bandwidth_gb_s", 51.2),
            max_freq_mhz=gpu.get("max_freq_mhz", 1300),
            tdp_w=gpu.get("tdp_w", 15.0),
        )

        th = cfg.get("thermal", {})
        self.thermal = ThermalSpec(
            throttle_temp_c=th.get("throttle_temp_c", 80),
            sustained_seconds=th.get("sustained_seconds", 30),
            max_throttle_factor=th.get("max_throttle_factor", 1.5),
        )

        sim = cfg.get("simulation", {})
        self.sim = SimSettings(
            limit_threads=sim.get("limit_threads", True),
            enable_throttle=sim.get("enable_throttle", True),
            memory_limit_mb=sim.get("memory_limit_mb", 6144),
            inference_width=sim.get("inference_width", 640),
            inference_height=sim.get("inference_height", 640),
            target_fps=sim.get("target_fps", 25.0),
            target_latency_ms=sim.get("target_latency_ms", 35.0),
        )

        # Track when continuous inference started (for throttle simulation)
        self._inference_start: Optional[float] = None

    # ------------------------------------------------------------------
    # CPU thread limiting
    # ------------------------------------------------------------------
    def cpu_threads_for_inference(self) -> int:
        """Return the number of CPU threads to use for inference in simulation."""
        return self.cpu.sim_cpu_threads if self.sim.limit_threads else self.cpu.threads

    def apply_cpu_affinity(self) -> None:
        """Attempt to restrict CPU affinity to simulate Jetson core count (Linux only)."""
        if not self.sim.limit_threads:
            return
        try:
            os.sched_setaffinity(0, list(range(self.cpu_threads_for_inference())))
        except (AttributeError, OSError):
            pass  # Not on Linux or no permission — ignore silently

    # ------------------------------------------------------------------
    # Thermal throttling simulation
    # ------------------------------------------------------------------
    def start_inference_timer(self) -> None:
        """Call once before the inference loop to begin throttling simulation."""
        self._inference_start = time.time()

    def thermal_factor(self, elapsed_seconds: Optional[float] = None) -> float:
        """
        Return a multiplier (>=1.0) that simulates how much slower
        inference becomes due to sustained-load thermal throttling.

        - 0–30 s: no throttling (factor = 1.0)
        - 30–90 s: linear ramp from 1.0 to max_throttle_factor
        - >90 s: capped at max_throttle_factor
        """
        if not self.sim.enable_throttle:
            return 1.0
        if elapsed_seconds is None:
            if self._inference_start is None:
                return 1.0
            elapsed_seconds = time.time() - self._inference_start

        sustained = self.thermal.sustained_seconds
        max_factor = self.thermal.max_throttle_factor

        if elapsed_seconds <= sustained:
            return 1.0

        # Linear ramp over the next 60 seconds (sustained → sustained+60)
        ramp_seconds = 60.0
        ratio = min((elapsed_seconds - sustained) / ramp_seconds, 1.0)
        return 1.0 + (max_factor - 1.0) * ratio

    # ------------------------------------------------------------------
    # Memory guard
    # ------------------------------------------------------------------
    def check_memory(self, used_mb: float) -> bool:
        """Return True if `used_mb` is within the simulated memory limit."""
        return used_mb <= self.sim.memory_limit_mb

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    def summary(self) -> str:
        """Return a human-readable summary of the hardware profile."""
        return (
            f"Platform: {self.name}\n"
            f"  CPU: {self.cpu.cores} cores @ {self.cpu.max_freq_mhz} MHz "
            f"(sim threads: {self.cpu_threads_for_inference()})\n"
            f"  GPU: {self.gpu.cuda_cores} CUDA cores, {self.gpu.tensor_cores} Tensor cores, "
            f"{self.gpu.memory_mb} MB\n"
            f"  Power: {self.gpu.tdp_w}W TDP\n"
            f"  Thermal throttle: >{self.thermal.sustained_seconds}s (max {self.thermal.max_throttle_factor}x)\n"
            f"  Target FPS: {self.sim.target_fps} @ {self.sim.inference_width}x{self.sim.inference_height}\n"
            f"  Target latency: {self.sim.target_latency_ms} ms"
        )


class SimulationConfig:
    """Convenience class for passing simulation settings to the pipeline."""

    def __init__(self, profile: JetsonProfile):
        self.profile = profile
        self.inference_width = profile.sim.inference_width
        self.inference_height = profile.sim.inference_height
        self.cpu_threads = profile.cpu_threads_for_inference()
        self.memory_limit_mb = profile.sim.memory_limit_mb
        self.target_fps = profile.sim.target_fps
        self.target_latency_ms = profile.sim.target_latency_ms
