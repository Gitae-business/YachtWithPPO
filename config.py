# Configuration for Yacht PPO project

# Training Parameters
NUM_EPISODES = int(1e7)
MAX_STEPS_PER_EPISODE = 100 # Increased to allow more steps per episode
UPDATE_TIMESTEPS = 4096 # Collect experience for N steps before updating

# PPO Agent Parameters
LR = 2e-4
GAMMA = 0.99
CLIP_EPSILON = 0.2
GAE_LAMBDA = 0.95
PPO_EPOCHS = 10
MINI_BATCH_SIZE = 128

# Logging
LOG_FILE = "training_log.csv"

# Checkpointing
CHECKPOINT_DIR = "checkpoints"
CHECKPOINT_INTERVAL = 1000 # Save checkpoint every N episodes

# Environment Parameters
NUM_ENVS = 4 # Number of parallel environments