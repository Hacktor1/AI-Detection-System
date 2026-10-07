"""
Jetson simulation environment for AI-Detection-System.

Provides hardware profile simulation, dual-camera input generation,
and benchmark utilities for testing the detection pipeline on a
simulated Jetson Orin Nano without physical hardware.
"""
from .hardware_profile import JetsonProfile, SimulationConfig

__all__ = ["JetsonProfile", "SimulationConfig"]
