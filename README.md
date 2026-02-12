# Autonomous Solar Vehicle - ML Navigation System

**Final Year Project - Capital University of Science & Technology, Islamabad**

A production-grade ML-based autonomous navigation system for a solar-powered vehicle prototype running on Raspberry Pi 4.

## Overview

This system enables a small-scale solar-powered vehicle to navigate autonomously from Point A to Point B, detecting and avoiding obstacles in real time. It uses sensor fusion from 5 ultrasonic sensors and 2 cameras to make driving decisions.

### Key Features

- **ML-Based Decision Making**: Neural network trained on 50,000+ synthetic samples
- **3-Layer Safety Architecture**: Hard safety rules that cannot be overridden by ML
- **Sensor Fusion**: Combines ultrasonic distance sensors with camera object detection
- **Real-Time Performance**: < 50ms decision latency on Raspberry Pi 4
- **TensorFlow Lite Deployment**: INT8 quantized models for efficient inference

## Hardware Requirements

### Platform
- **Raspberry Pi 4 Model B** (4GB RAM recommended)
- Raspberry Pi OS 64-bit (Debian Bullseye)
- Python 3.9+

### Sensors
| Component | Quantity | Interface | Purpose |
|-----------|----------|-----------|---------|
| HC-SR04 Ultrasonic | 5 | GPIO | Distance measurement |
| Pi Camera v2 | 1 | CSI | Front object detection |
| USB Camera | 1 | USB | Rear monitoring |
| NEO-6M GPS (optional) | 1 | UART | Position tracking |

### Motor Control
- L298N Dual H-Bridge Motor Driver
- 4x DC Motors (tank-style differential steering)

## Project Structure

```
autonomous_solar_vehicle/
├── config/                 # Configuration (GPIO pins, thresholds, paths)
├── src/
│   ├── sensors/            # Ultrasonic, camera, sensor fusion
│   ├── perception/         # Object detection, distance estimation
│   ├── navigation/         # Path planning, GPS handling
│   ├── decision/           # Safety governor, ML engine, rule fallback
│   ├── control/            # Motor control, steering
│   └── core/               # Main loop, state machine
├── training/               # ML training pipeline
├── tests/                  # Unit and integration tests
├── scripts/                # Utility scripts
├── models/                 # Trained models
├── data/                   # Training data
└── logs/                   # Runtime and training logs
```

## Installation

### On Development Machine

```bash
# Clone the repository
git clone https://github.com/your-repo/autonomous-solar-vehicle.git
cd autonomous-solar-vehicle

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
.\venv\Scripts\activate  # Windows

# Install with development dependencies
pip install -e ".[dev,training]"
```

### On Raspberry Pi

```bash
# Install system dependencies
sudo apt update
sudo apt install -y python3-pip python3-venv python3-picamera2

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install package
pip install -e ".[rpi]"
```

## Quick Start

### 1. Generate Training Data

```bash
python -m training.generate_synthetic_data
```

### 2. Train the Decision Model

```bash
python -m training.train_decision_model
```

### 3. Convert to TFLite

```bash
python -m training.convert_to_tflite --quantization int8
```

### 4. Run the Vehicle

```bash
# On Raspberry Pi with hardware
python -m src.core.main_loop

# In simulation mode (for testing)
python -m src.core.main_loop --simulation
```

## Safety Architecture

The system implements a 3-layer safety architecture:

```
┌─────────────────────────────────────────────────────────┐
│  Layer 3: SAFETY GOVERNOR (Outermost)                   │
│  - front < 20cm → STOP (always)                         │
│  - person detected → STOP (always)                      │
│  - sensor error → STOP                                  │
│  *** CANNOT be overridden by ML ***                     │
├─────────────────────────────────────────────────────────┤
│  Layer 2: ML DECISION ENGINE                            │
│  - Neural network inference                             │
│  - Operates only when Layer 3 permits                   │
├─────────────────────────────────────────────────────────┤
│  Layer 1: RULE ENGINE (Fallback)                        │
│  - Activates if ML confidence < 0.7                     │
│  - Deterministic expert rules                           │
└─────────────────────────────────────────────────────────┘
```

## Decision Model

### Input Features (12)

| # | Feature | Description | Range |
|---|---------|-------------|-------|
| 0 | front_distance | Front ultrasonic (cm) | 0-400 |
| 1 | front_wp_distance | Front waterproof ultrasonic (cm) | 0-400 |
| 2 | right_distance | Right ultrasonic (cm) | 0-400 |
| 3 | left_distance | Left ultrasonic (cm) | 0-400 |
| 4 | rear_distance | Rear ultrasonic (cm) | 0-400 |
| 5 | front_min_distance | min(front, front_wp) | 0-400 |
| 6 | camera_object_detected | Object detected (0/1) | 0-1 |
| 7 | camera_object_class | Object class ID | 0-10 |
| 8 | camera_object_distance | Estimated distance (cm) | 0-400 |
| 9 | camera_object_position | Position (left/center/right) | 0-2 |
| 10 | current_speed | Current PWM % | 0-100 |
| 11 | heading_error | Path deviation (degrees) | -90-90 |

### Output Actions (7)

| ID | Action | Description |
|----|--------|-------------|
| 0 | FORWARD | Continue straight |
| 1 | SLOW_DOWN | Reduce speed by 30-50% |
| 2 | TURN_LEFT | Steer left |
| 3 | TURN_RIGHT | Steer right |
| 4 | STOP | Full stop |
| 5 | REVERSE_LEFT | Back up turning left |
| 6 | REVERSE_RIGHT | Back up turning right |

### Architecture

```
Input (12 features, normalized 0-1)
    ↓
Dense(64, ReLU) → BatchNorm → Dropout(0.2)
    ↓
Dense(32, ReLU) → BatchNorm → Dropout(0.2)
    ↓
Dense(16, ReLU)
    ↓
Dense(7, Softmax)
    ↓
Output (7 action probabilities)
```

## Performance Targets

| Metric | Target | Actual |
|--------|--------|--------|
| Decision model accuracy | ≥ 98% | TBD |
| Inference latency (decision) | < 50ms | TBD |
| Inference latency (camera) | < 150ms | TBD |
| Overall loop frequency | ≥ 5 Hz | TBD |
| Missed obstacle rate (< 30cm) | 0% | TBD |

## GPIO Pin Configuration

### Ultrasonic Sensors

| Sensor | Trigger | Echo |
|--------|---------|------|
| US_FRONT | GPIO 17 | GPIO 27 |
| US_FRONT_WP | GPIO 22 | GPIO 23 |
| US_RIGHT | GPIO 24 | GPIO 25 |
| US_LEFT | GPIO 5 | GPIO 6 |
| US_REAR | GPIO 12 | GPIO 16 |

### Motor Driver (L298N)

| Motor | Enable | IN1 | IN2 |
|-------|--------|-----|-----|
| Left | GPIO 13 | GPIO 19 | GPIO 26 |
| Right | GPIO 18 | GPIO 20 | GPIO 21 |

## Testing

```bash
# Run all tests
make test

# Run specific test categories
make test-safety      # Critical safety tests
make test-sensors     # Sensor tests
make test-integration # Full pipeline tests

# Run with coverage
pytest tests/ -v --cov=src --cov-report=html
```

## Deployment

```bash
# Deploy to Raspberry Pi
make deploy

# Or manually:
rsync -avz --exclude '__pycache__' ./ pi@<RPI_IP>:~/autonomous_solar_vehicle/
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make changes with tests
4. Ensure all tests pass (especially safety tests)
5. Submit a pull request

## License

MIT License - See LICENSE file for details.

## Authors

FYP Team - Capital University of Science & Technology, Islamabad

## Acknowledgments

- TensorFlow Lite for efficient on-device inference
- OpenCV for computer vision
- RPi.GPIO for hardware interface
