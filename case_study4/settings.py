from pathlib import Path


N_TRAINING_BATCHES = 256
BATCH_SIZE = 128
EPOCHS = 500
N_TRIALS = 30
N_SUBJECTS = 100
N_SAMPLES = 100
N_TEST = 100
METHOD = 'two_step_adaptive'
STEPS = "adaptive"
MAX_STEP = 1_000
BASE = Path(__file__).resolve().parent

