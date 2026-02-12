#!/usr/bin/env python3
"""
Setup script for Autonomous Solar Vehicle package.

This package provides ML-based autonomous navigation for a solar-powered
vehicle prototype running on Raspberry Pi 4.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
this_directory = Path(__file__).parent
long_description = ""
readme_path = this_directory / "README.md"
if readme_path.exists():
    long_description = readme_path.read_text(encoding="utf-8")

# Core dependencies that work on both dev machine and RPi
INSTALL_REQUIRES = [
    "numpy>=1.23.0,<2.0.0",
    "scipy>=1.9.0",
    "pandas>=1.5.0",
    "scikit-learn>=1.2.0",
    "opencv-python-headless>=4.7.0",
    "Pillow>=9.4.0",
    "pyserial>=3.5",
    "pynmea2>=1.18.0",
    "python-dotenv>=1.0.0",
    "PyYAML>=6.0",
    "structlog>=23.1.0",
    "schedule>=1.2.0",
]

# Development dependencies
DEV_REQUIRES = [
    "pytest>=7.3.0",
    "pytest-cov>=4.1.0",
    "pytest-mock>=3.10.0",
    "hypothesis>=6.75.0",
    "flake8>=6.0.0",
    "black>=23.3.0",
    "isort>=5.12.0",
    "mypy>=1.3.0",
]

# Training dependencies (for dev machine with GPU)
TRAINING_REQUIRES = [
    "tensorflow>=2.10.0,<2.14.0",
    "tensorboard>=2.10.0",
    "jupyter>=1.0.0",
    "matplotlib>=3.7.0",
    "seaborn>=0.12.0",
]

# Raspberry Pi specific dependencies
RPI_REQUIRES = [
    "RPi.GPIO>=0.7.0",
    "tflite-runtime>=2.10.0",
]

setup(
    name="autonomous_solar_vehicle",
    version="1.0.0",
    author="FYP Team - Capital University of Science & Technology",
    author_email="fyp@cust.edu.pk",
    description="ML-based autonomous navigation system for solar-powered vehicle",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/cust-fyp/autonomous-solar-vehicle",
    project_urls={
        "Bug Tracker": "https://github.com/cust-fyp/autonomous-solar-vehicle/issues",
        "Documentation": "https://github.com/cust-fyp/autonomous-solar-vehicle/wiki",
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: System :: Hardware :: Hardware Drivers",
    ],
    package_dir={"": "."},
    packages=find_packages(where=".", exclude=["tests*", "notebooks*", "scripts*"]),
    python_requires=">=3.9,<3.12",
    install_requires=INSTALL_REQUIRES,
    extras_require={
        "dev": DEV_REQUIRES,
        "training": TRAINING_REQUIRES,
        "rpi": RPI_REQUIRES,
        "all": DEV_REQUIRES + TRAINING_REQUIRES,
    },
    entry_points={
        "console_scripts": [
            "asv-run=src.core.main_loop:main",
            "asv-train=training.train_decision_model:main",
            "asv-calibrate=scripts.calibrate_sensors:main",
            "asv-diagnostics=scripts.run_diagnostics:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
