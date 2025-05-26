# 설정값 (하이퍼파라미터 등) 구현 예정 

import torch

# --- Paths ---
DATA_DIR = "data/"
MSCOCO_DIR = DATA_DIR + "mscoco/"
KARPATHY_SPLIT_DIR = DATA_DIR + "karpathy_split/"
VOCAB_PATH = DATA_DIR + "vocab.pkl"
MODEL_SAVE_DIR = "checkpoints/"

# Training data paths
TRAIN_IMAGE_DIR = MSCOCO_DIR + "train2014"
TRAIN_CAPTION_JSON = KARPATHY_SPLIT_DIR + "dataset_coco.json"

# Validation data paths
VAL_IMAGE_DIR = MSCOCO_DIR + "val2014"
VAL_CAPTION_JSON = KARPATHY_SPLIT_DIR + "dataset_coco.json"

# Test data paths (Karpathy split uses val2014 images for testing)
TEST_IMAGE_DIR = MSCOCO_DIR + "val2014"
TEST_CAPTION_JSON = KARPATHY_SPLIT_DIR + "dataset_coco.json"

# --- Data Preprocessing ---
MIN_WORD_FREQ = 5  # Minimum word frequency for vocabulary
MAX_SEQ_LEN = 40   # Max sequence length (including [BOS], [EOS])

# --- Model Hyperparameters ---
# General Transformer Settings (as per paper)
NUM_LAYERS = 2
HIDDEN_SIZE = 384
FEED_FORWARD_SIZE = 2048
DROPOUT_PROB = 0.2
NUM_ATTENTION_HEADS = 8 # Common choice, can be adjusted

# Image Encoder
IMAGE_EMBED_SIZE = HIDDEN_SIZE # Output size of image encoder
ENCODER_MODEL_NAME = "resnet50"  # 더 가벼운 모델
PRETRAINED_ENCODER = True
FINE_TUNE_CNN = True  # Whether to fine-tune the CNN backbone

# Uncertainty Estimator
UE_NUM_LAYERS = 4
UE_HIDDEN_SIZE = 512
UE_FEED_FORWARD_SIZE = 2048
UE_DROPOUT = 0.1
UE_BATCH_NORM = True
UE_GRADIENT_CLIP = 1.0

# Insertion Transformer
IT_NUM_LAYERS = NUM_LAYERS # Can be different if needed

# --- Training Hyperparameters ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 32
ACCUMULATION_STEPS = 4
NUM_EPOCHS = 30
LEARNING_RATE = 1e-4
LR_DECAY_FACTOR = 0.9
LR_DECAY_EPOCHS = 5
WARMUP_STEPS = 4000
CLIP_GRAD_NORM = 1.0
LR_SCHEDULER_PATIENCE = 2
LR_SCHEDULER_FACTOR = 0.5

# 데이터 로딩 최적화
NUM_WORKERS = 8  # 워커 수 증가
PIN_MEMORY = True

# --- Beam Search ---
DEFAULT_BEAM_SIZE = 3 # For standard beam search if UA-BeamSearch is not used
# UA-BeamSearch specific (as per paper)
# B_k = 3 + int(4 * max(-0.5, min(0.5, (u_avg - u_k) / u_avg)))

# --- Special Tokens ---
PAD_TOKEN = "[PAD]"
BOS_TOKEN = "[BOS]"
EOS_TOKEN = "[EOS]"
UNK_TOKEN = "[UNK]"
NONE_TOKEN = "[NONE]" # For insertion transformer
MASK_TOKEN = "[MASK]" # Placeholder for words to be predicted in S_k-1

# Special token indices (these should match the vocabulary)
PAD_IDX = 0
BOS_IDX = 1
EOS_IDX = 2
UNK_IDX = 3
NONE_IDX = 4
MASK_IDX = 5

# --- Logging and Saving ---
LOG_FREQ = 100 # Log every N batches
SAVE_FREQ_EPOCHS = 1 # Save model every N epochs

# --- Evaluation ---
EVAL_BATCH_SIZE = 32
EVAL_BEAM_SIZE = 5  # 평가시 사용할 빔 크기
LOG_STEP = 100  # 평가 로그 출력 간격
BEAM_SIZE = DEFAULT_BEAM_SIZE  # 기본 빔 크기 설정

# 메모리 최적화 설정
USE_FP16 = True  # AMP 활성화
USE_CUDA = True
CUDNN_BENCHMARK = True

# 데이터셋 크기 제한 (선택사항)
MAX_DATASET_SIZE = 50000  # 데이터셋 크기 제한 

# Loss 관련 설정
WORD_LOSS_WEIGHT = 1.0
POS_LOSS_WEIGHT = 1.0
UNCERTAINTY_LOSS_WEIGHT = 0.5
SPECIAL_TOKEN_WEIGHT = 0.1  # 특수 토큰에 대한 낮은 가중치 