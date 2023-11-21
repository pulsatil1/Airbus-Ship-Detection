# Model Parameters
BATCH_SIZE = 48
EDGE_CROP = 16
GAUSSIAN_NOISE = 0.1
UPSAMPLE_MODE = 'SIMPLE'
# downsampling inside the network
NET_SCALING = (1, 1)
# downsampling in preprocessing
IMG_SCALING = (3, 3)
# number of validation images to use
VALID_IMG_COUNT = 900
# maximum number of steps_per_epoch in training
MAX_TRAIN_STEPS = 10
MAX_TRAIN_EPOCHS = 100
AUGMENT_BRIGHTNESS = False
SAMPLES_PER_GROUP = 4000

BASE_DIR = ''
TRAIN_DIR = BASE_DIR + 'train_v2/'
TEST_DIR = BASE_DIR + 'test_v2/'
