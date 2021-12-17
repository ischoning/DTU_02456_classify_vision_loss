# Data processing parameters
SAMPLE_WINDOW = 2     # in seconds
STEP_SIZE = 0.2

# Network parameters
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-2
VALIDATION_SIZE = 0.1 # percent of train to allocate to validation

# Pupil Invisible Specs
SAMPLE_RATE = 66      #Hz

# Data parameters
FEATS = ['x_deg', 'y_deg', 'isd', 'isv']
TEST_FEATS = ['steps', 'sec', 'motor', 'process']
NUM_FEATS = len(FEATS)
SEQ_LENGTH = int(SAMPLE_WINDOW * SAMPLE_RATE)
NUM_CLASSES = 2       # binary for PVL vs CVL
IN_CHANNELS = 1
BATCH_SIZE = 20
HIDDEN_SIZE = 100

FEATS2 = ['x_mu', 'x_std', 'y_mu', 'y_std']
TEST_FEATS2 = ['d_mu', 'd_std', 'v_mu', 'v_std']
NUM_FEATS2 = len(FEATS2)+len(TEST_FEATS2)
SEQ_LENGTH2 = int(1)
