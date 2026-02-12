"""
Training Module for Autonomous Solar Vehicle.

This module provides the ML training pipeline:
- generate_synthetic_data: Generate training data from expert rules
- data_preprocessor: Clean, normalize, and split data
- train_decision_model: Train the decision neural network
- evaluate_model: Evaluate model performance
- convert_to_tflite: Convert to TFLite for deployment

Training Workflow:
1. Generate synthetic data from expert rules (50,000+ samples)
2. Preprocess and split data (80/10/10 train/val/test)
3. Train decision model
4. Evaluate on test set
5. Convert to TFLite INT8 quantized model
"""
