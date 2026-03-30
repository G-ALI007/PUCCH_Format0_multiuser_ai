"""
================================================================================
PUCCH Format 0 - ML Decoder with DTX Detection
Configuration File (5 Classes)
================================================================================

This configuration extends the base 4-class system to include
DTX (Discontinuous Transmission) detection as a 5th class.

This is a novel contribution not implemented in the original paper:
"Machine Learning Decoder for 5G NR PUCCH Format 0"

The original paper stated:
"False detections can easily be incorporated into our framework 
by adding an additional class label whose inputs would be instances of AWGN"

We implement this proposal and evaluate its performance.

================================================================================
"""

import os
import datetime


class ConfigDTX:
    """
    Configuration for 5-class PUCCH Format 0 decoder (including DTX).

    Classes:
        0: ACK=0, SR=0 (NACK, no SR)     → mcs=0
        1: ACK=0, SR=1 (NACK, +SR)       → mcs=3
        2: ACK=1, SR=0 (ACK, no SR)      → mcs=6
        3: ACK=1, SR=1 (ACK, +SR)        → mcs=9
        4: DTX (No Transmission)          → no signal
    """

    # =========================================================================
    # EXPERIMENT IDENTIFICATION
    # =========================================================================

    EXPERIMENT_NAME = "PUCCH_Format0_ML_Decoder_DTX"
    EXPERIMENT_VERSION = "1.0"
    EXPERIMENT_TIMESTAMP = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # =========================================================================
    # RANDOM SEEDS
    # =========================================================================

    MASTER_SEED = 42
    NUMPY_SEED = 42
    TENSORFLOW_SEED = 42
    SKLEARN_SEED = 42

    # =========================================================================
    # DATA PARAMETERS
    # =========================================================================

    DATA_DIR = "./"

    SNR_VALUES = [0, 5, 10, 15, 20]
    TRAIN_SNR = 10

    NUM_FEATURES = 24
    NUM_SUBCARRIERS = 12

    # *** KEY CHANGE: 5 classes instead of 4 ***
    NUM_CLASSES = 5

    # Class definitions (including DTX)
    CLASS_LABELS = {
        0: "ACK=0, SR=0 (NACK, no SR)",
        1: "ACK=0, SR=1 (NACK, +SR)",
        2: "ACK=1, SR=0 (ACK, no SR)",
        3: "ACK=1, SR=1 (ACK, +SR)",
        4: "DTX (No Transmission)"
    }

    # Number of UCI classes (without DTX)
    NUM_UCI_CLASSES = 4

    # DTX class index
    DTX_CLASS = 4

    # MCS values (DTX has no mcs)
    MCS_VALUES = {
        0: 0,
        1: 3,
        2: 6,
        3: 9,
        4: None  # DTX has no cyclic shift
    }

    # CSV format
    EXPECTED_NUM_COLUMNS = 25
    FEATURE_COLUMNS_REAL = [f"real_{i}" for i in range(1, 13)]
    FEATURE_COLUMNS_IMAG = [f"imag_{i}" for i in range(1, 13)]
    FEATURE_COLUMNS = FEATURE_COLUMNS_REAL + FEATURE_COLUMNS_IMAG
    LABEL_COLUMN = "label"

    # UCI data file pattern
    UCI_FILE_PATTERN = "pucch_f0_dataset_SNR_{snr}dB.csv"

    # DTX data file pattern
    DTX_FILE_PATTERN = "pucch_f0_DTX_dataset_SNR_{snr}dB.csv"

    # =========================================================================
    # MATLAB PARAMETERS (for documentation)
    # =========================================================================

    MATLAB_SAMPLES_PER_SNR_UCI = 200000
    MATLAB_SAMPLES_PER_CLASS_UCI = 50000
    MATLAB_SAMPLES_PER_SNR_DTX = 50000
    MATLAB_TOTAL_SAMPLES_PER_SNR = 250000  # 200000 UCI + 50000 DTX

    MATLAB_ALLOWED_SLOTS = [13, 14]
    MATLAB_SYMBOL_START = 13
    MATLAB_SYMBOL_COUNT = 1
    MATLAB_INITIAL_CYCLIC_SHIFT = 0
    MATLAB_FREQUENCY_HOPPING = "neither"
    MATLAB_CHANNEL_MODEL = "TDL-C"
    MATLAB_DELAY_SPREAD = 300e-9
    MATLAB_MAX_DOPPLER = 100
    MATLAB_NCELL_ID = 2
    MATLAB_SUBCARRIER_SPACING = 15
    MATLAB_NSIZE_GRID = 25
    MATLAB_NUM_TX_ANTENNAS = 1
    MATLAB_NUM_RX_ANTENNAS = 1

    # =========================================================================
    # DATA SPLIT
    # =========================================================================

    TRAIN_RATIO = 0.75
    VALIDATION_RATIO = 0.25
    STRATIFY_SPLIT = True

    # =========================================================================
    # PREPROCESSING
    # =========================================================================

    NORMALIZE_FEATURES = False
    NORMALIZATION_TYPE = "standard"

    # =========================================================================
    # NEURAL NETWORK ARCHITECTURE
    # =========================================================================

    INPUT_SIZE = 24
    HIDDEN_LAYERS = [128, 128]
    HIDDEN_ACTIVATION = "relu"

    # *** KEY CHANGE: 5 output neurons instead of 4 ***
    OUTPUT_SIZE = 5

    OUTPUT_ACTIVATION = "softmax"
    DROPOUT_RATE = 0.5
    USE_DROPOUT = True
    KERNEL_INITIALIZER = "glorot_uniform"

    # =========================================================================
    # TRAINING PARAMETERS
    # =========================================================================

    NUM_EPOCHS = 200
    BATCH_SIZE = 256
    OPTIMIZER = "sgd"
    LEARNING_RATE = 1e-3
    MOMENTUM = 0.9
    USE_NESTEROV = False
    LOSS_FUNCTION = "categorical_crossentropy"

    # =========================================================================
    # CALLBACKS
    # =========================================================================

    USE_EARLY_STOPPING = True
    EARLY_STOPPING_MONITOR = "val_loss"
    EARLY_STOPPING_MODE = "min"
    EARLY_STOPPING_PATIENCE = 20
    EARLY_STOPPING_MIN_DELTA = 0.0001
    EARLY_STOPPING_RESTORE_BEST = True

    USE_MODEL_CHECKPOINT = True
    CHECKPOINT_MONITOR = "val_accuracy"
    CHECKPOINT_MODE = "max"
    CHECKPOINT_SAVE_BEST_ONLY = True

    USE_REDUCE_LR = True
    REDUCE_LR_MONITOR = "val_loss"
    REDUCE_LR_FACTOR = 0.5
    REDUCE_LR_PATIENCE = 10
    REDUCE_LR_MIN_LR = 1e-6

    # =========================================================================
    # EXPERIMENT REPETITION
    # =========================================================================

    NUM_EXPERIMENT_RUNS = 1
    USE_DIFFERENT_SEEDS = True
    SEED_INCREMENT = 100

    # =========================================================================
    # OUTPUT PATHS (separate from base experiment)
    # =========================================================================

    RESULTS_DIR = "./results_dtx/"
    MODELS_DIR = "./models_dtx/"
    PLOTS_DIR = "./plots_dtx/"
    LOGS_DIR = "./logs_dtx/"

    MODEL_FILENAME = "pucch_f0_nn_decoder_dtx.h5"
    SCALER_FILENAME = "feature_scaler_dtx.pkl"
    TRAINING_HISTORY_FILENAME = "training_history_dtx.csv"
    RESULTS_FILENAME = "results_summary_dtx.csv"

    # =========================================================================
    # DTX-SPECIFIC METRICS
    # =========================================================================

    # 3GPP requirements
    # False Alarm: probability of detecting UCI when DTX was sent
    # Should be very low (typically < 1%)
    FALSE_ALARM_REQUIREMENT = 0.01  # 1%

    # Missed Detection: probability of not detecting UCI when it was sent
    # Should be very low (typically < 1%)
    MISSED_DETECTION_REQUIREMENT = 0.01  # 1%

    # =========================================================================
    # VISUALIZATION
    # =========================================================================

    FIGURE_SIZE_SMALL = (8, 6)
    FIGURE_SIZE_MEDIUM = (10, 6)
    FIGURE_SIZE_LARGE = (12, 8)
    FIGURE_SIZE_WIDE = (14, 5)
    FIGURE_DPI = 150
    FONT_SIZE_SMALL = 10
    FONT_SIZE_MEDIUM = 12
    FONT_SIZE_LARGE = 14
    FONT_SIZE_TITLE = 16

    # =========================================================================
    # UTILITY METHODS
    # =========================================================================

    @classmethod
    def create_directories(cls):
        """Create output directories."""
        for directory in [cls.RESULTS_DIR, cls.MODELS_DIR, cls.PLOTS_DIR, cls.LOGS_DIR]:
            if directory and not os.path.exists(directory):
                os.makedirs(directory)
                print(f"Created: {directory}")

    @classmethod
    def get_run_seed(cls, run_index: int) -> int:
        """Get seed for a specific run."""
        if cls.USE_DIFFERENT_SEEDS:
            return cls.MASTER_SEED + (run_index * cls.SEED_INCREMENT)
        return cls.MASTER_SEED

    @classmethod
    def get_uci_filepath(cls, snr_db: int) -> str:
        """Get UCI data file path."""
        filename = cls.UCI_FILE_PATTERN.format(snr=snr_db)
        return os.path.join(cls.DATA_DIR, filename)

    @classmethod
    def get_dtx_filepath(cls, snr_db: int) -> str:
        """Get DTX data file path."""
        filename = cls.DTX_FILE_PATTERN.format(snr=snr_db)
        return os.path.join(cls.DATA_DIR, filename)

    @classmethod
    def get_model_filepath(cls, run_index: int = None) -> str:
        """Get model file path."""
        if run_index is not None:
            name, ext = os.path.splitext(cls.MODEL_FILENAME)
            filename = f"{name}_run{run_index}{ext}"
        else:
            filename = cls.MODEL_FILENAME
        return os.path.join(cls.MODELS_DIR, filename)

    @classmethod
    def print_config(cls):
        """Print configuration summary."""
        print("\n" + "=" * 70)
        print("DTX EXPERIMENT CONFIGURATION")
        print("=" * 70)

        print(f"\nExperiment: {cls.EXPERIMENT_NAME} v{cls.EXPERIMENT_VERSION}")

        print(f"\n--- Data ---")
        print(f"SNR values: {cls.SNR_VALUES} dB")
        print(f"Training SNR: {cls.TRAIN_SNR} dB")
        print(f"Features: {cls.NUM_FEATURES}")
        print(f"Classes: {cls.NUM_CLASSES} (4 UCI + 1 DTX)")

        print(f"\n--- Classes ---")
        for c, label in cls.CLASS_LABELS.items():
            marker = " ← NEW" if c == cls.DTX_CLASS else ""
            print(f"  Class {c}: {label}{marker}")

        print(f"\n--- Data Split ---")
        print(f"Train/Val: {cls.TRAIN_RATIO}/{cls.VALIDATION_RATIO}")

        print(f"\n--- Neural Network ---")
        print(f"Input: {cls.INPUT_SIZE}")
        print(f"Hidden: {cls.HIDDEN_LAYERS}")
        print(f"Output: {cls.OUTPUT_SIZE} (was 4, now 5)")
        print(f"Dropout: {cls.DROPOUT_RATE}")

        print(f"\n--- Training ---")
        print(f"Epochs: {cls.NUM_EPOCHS}")
        print(f"Batch: {cls.BATCH_SIZE}")
        print(
            f"Optimizer: SGD(lr={cls.LEARNING_RATE}, momentum={cls.MOMENTUM})")

        print(f"\n--- DTX Metrics ---")
        print(f"False Alarm requirement: < {cls.FALSE_ALARM_REQUIREMENT*100}%")
        print(
            f"Missed Detection requirement: < {cls.MISSED_DETECTION_REQUIREMENT*100}%")

        print(f"\n--- Output ---")
        print(f"Results: {cls.RESULTS_DIR}")
        print(f"Models: {cls.MODELS_DIR}")
        print(f"Plots: {cls.PLOTS_DIR}")

        print("=" * 70 + "\n")

    @classmethod
    def validate_config(cls) -> bool:
        """Validate configuration."""
        errors = []

        if cls.TRAIN_SNR not in cls.SNR_VALUES:
            errors.append(f"TRAIN_SNR ({cls.TRAIN_SNR}) not in SNR_VALUES")

        if abs(cls.TRAIN_RATIO + cls.VALIDATION_RATIO - 1.0) > 0.001:
            errors.append("TRAIN_RATIO + VALIDATION_RATIO should equal 1.0")

        if cls.INPUT_SIZE != cls.NUM_FEATURES:
            errors.append(
                f"INPUT_SIZE ({cls.INPUT_SIZE}) != NUM_FEATURES ({cls.NUM_FEATURES})")

        if cls.OUTPUT_SIZE != cls.NUM_CLASSES:
            errors.append(
                f"OUTPUT_SIZE ({cls.OUTPUT_SIZE}) != NUM_CLASSES ({cls.NUM_CLASSES})")

        if cls.NUM_CLASSES != cls.NUM_UCI_CLASSES + 1:
            errors.append(f"NUM_CLASSES should be NUM_UCI_CLASSES + 1")

        if cls.DTX_CLASS != cls.NUM_UCI_CLASSES:
            errors.append(f"DTX_CLASS should equal NUM_UCI_CLASSES")

        if len(cls.CLASS_LABELS) != cls.NUM_CLASSES:
            errors.append(
                f"CLASS_LABELS has {len(cls.CLASS_LABELS)} entries, expected {cls.NUM_CLASSES}")

        if not 0.0 <= cls.DROPOUT_RATE <= 1.0:
            errors.append(f"DROPOUT_RATE ({cls.DROPOUT_RATE}) out of range")

        if cls.LEARNING_RATE <= 0:
            errors.append(
                f"LEARNING_RATE ({cls.LEARNING_RATE}) must be positive")

        if errors:
            print("\n" + "!" * 70)
            print("CONFIGURATION VALIDATION ERRORS")
            print("!" * 70)
            for error in errors:
                print(f"  - {error}")
            print("!" * 70 + "\n")
            return False
        else:
            print("Configuration validation: PASSED")
            return True

    @classmethod
    def print_comparison_with_base(cls):
        """Print comparison between base (4-class) and DTX (5-class) configs."""
        from config import config as base_config

        print("\n" + "=" * 70)
        print("COMPARISON: Base (4-class) vs DTX (5-class)")
        print("=" * 70)

        comparisons = [
            ("NUM_CLASSES", base_config.NUM_CLASSES, cls.NUM_CLASSES),
            ("OUTPUT_SIZE", base_config.OUTPUT_SIZE, cls.OUTPUT_SIZE),
            ("RESULTS_DIR", base_config.RESULTS_DIR, cls.RESULTS_DIR),
            ("MODELS_DIR", base_config.MODELS_DIR, cls.MODELS_DIR),
            ("MODEL_FILENAME", base_config.MODEL_FILENAME, cls.MODEL_FILENAME),
        ]

        print(f"\n{'Parameter':<20}{'Base (4-class)':<20}{'DTX (5-class)':<20}")
        print("-" * 60)
        for param, base_val, dtx_val in comparisons:
            changed = " ← CHANGED" if base_val != dtx_val else ""
            print(f"{param:<20}{str(base_val):<20}{str(dtx_val):<20}{changed}")

        print("\nNew in DTX config:")
        print(f"  DTX_CLASS = {cls.DTX_CLASS}")
        print(f"  NUM_UCI_CLASSES = {cls.NUM_UCI_CLASSES}")
        print(f"  FALSE_ALARM_REQUIREMENT = {cls.FALSE_ALARM_REQUIREMENT}")
        print(
            f"  MISSED_DETECTION_REQUIREMENT = {cls.MISSED_DETECTION_REQUIREMENT}")
        print(f"  UCI_FILE_PATTERN = {cls.UCI_FILE_PATTERN}")
        print(f"  DTX_FILE_PATTERN = {cls.DTX_FILE_PATTERN}")

        print("=" * 70 + "\n")


# Create configuration instance
config_dtx = ConfigDTX()


# =============================================================================
# Self-test
# =============================================================================

if __name__ == "__main__":
    print("\nRunning config_dtx.py self-test...\n")

    config_dtx.print_config()

    is_valid = config_dtx.validate_config()

    print("\n--- Testing Utility Methods ---")
    print(f"Run 0 seed: {config_dtx.get_run_seed(0)}")
    print(f"UCI file for SNR=10: {config_dtx.get_uci_filepath(10)}")
    print(f"DTX file for SNR=10: {config_dtx.get_dtx_filepath(10)}")
    print(f"Model file: {config_dtx.get_model_filepath()}")

    print("\n--- Comparison with Base Config ---")
    config_dtx.print_comparison_with_base()

    print("\n--- Creating Directories ---")
    config_dtx.create_directories()

    print("\nSelf-test complete!")
