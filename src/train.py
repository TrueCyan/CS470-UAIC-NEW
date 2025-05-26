# 훈련 스크립트 구현 예정 

import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
import os
import time
import math # For math.isinf
from tqdm import tqdm
from tqdm.notebook import tqdm as tqdm_notebook
from torch.amp import autocast
from torch.cuda.amp import GradScaler
import torch.backends.cudnn as cudnn

# Robust import for project modules
try:
    from . import config
    from .utils.data_loader import get_loader, CocoDataset # Assuming CocoDataset might be needed for vocab path
    from .utils.vocabulary import Vocabulary
    from .utils.masking import generate_insertion_training_samples
    from .models.image_encoder import ImageEncoder
    from .models.uncertainty_estimator import UncertaintyEstimator
    from .models.insertion_transformer import InsertionTransformer
    from .losses import compute_insertion_losses
    # from .evaluate import evaluate_model # For validation during training
except ImportError:
    import sys
    # Add project root to path to allow for direct execution and relative imports
    current_dir = os.path.dirname(os.path.abspath(__file__)) # src/
    project_root = os.path.dirname(current_dir) # UAIC_NEW/
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    
    import config # type: ignore
    from utils.data_loader import get_loader, CocoDataset # type: ignore
    from utils.vocabulary import Vocabulary # type: ignore
    from utils.masking import generate_insertion_training_samples # type: ignore
    from models.image_encoder import ImageEncoder # type: ignore
    from models.uncertainty_estimator import UncertaintyEstimator # type: ignore
    from models.insertion_transformer import InsertionTransformer # type: ignore
    from losses import compute_insertion_losses # type: ignore
    # from evaluate import evaluate_model # type: ignore

def main():
    """Main training loop."""
    # ---- Setup ----
    device = torch.device("cuda" if torch.cuda.is_available() and config.USE_CUDA else "cpu")
    print(f"Using device: {device}")

    # CUDA 최적화 설정
    if torch.cuda.is_available():
        cudnn.benchmark = True  # 입력 크기가 일정할 때 속도 향상
        torch.cuda.empty_cache()  # CUDA 캐시 정리
    
    # AMP 설정
    scaler = torch.amp.GradScaler('cuda') if torch.cuda.is_available() else None

    # Create model directory if it doesn't exist
    if not os.path.exists(config.MODEL_SAVE_DIR):
        os.makedirs(config.MODEL_SAVE_DIR)

    # ---- Data Loading ----
    print("Loading vocabulary...")
    vocab = Vocabulary()
    # Attempt to load vocab; if not found, it implies it needs to be built first (e.g. by data_loader's main)
    # For training, we assume vocab_path points to an existing vocab file.
    try:
        vocab = Vocabulary.load_vocab(config.VOCAB_PATH)  # vocab.load_vocab() -> Vocabulary.load_vocab()
        print(f"Vocabulary loaded from {config.VOCAB_PATH}, size: {len(vocab)}")
    except FileNotFoundError:
        print(f"Error: Vocabulary file not found at {config.VOCAB_PATH}.")
        print("Please ensure the vocabulary is built and VOCAB_PATH in config.py is correct.")
        return # Exit if vocab can't be loaded or built
    
    pad_idx = vocab(config.PAD_TOKEN)
    bos_idx = vocab(config.BOS_TOKEN)
    eos_idx = vocab(config.EOS_TOKEN)
    none_idx = vocab(config.NONE_TOKEN)

    print("Loading training data...")
    train_loader = get_loader(
        image_dir=config.TRAIN_IMAGE_DIR,
        json_path=config.TRAIN_CAPTION_JSON,
        vocab=vocab,
        batch_size=config.BATCH_SIZE,
        split='train',
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY
    )
    # print("Loading validation data...") # Optional
    # val_loader = get_loader(
    #     image_dir=config.VAL_IMAGE_DIR,
    #     caption_json=config.VAL_CAPTION_JSON,
    #     vocab=vocab,
    #     transform_type='val',
    #     batch_size=config.BATCH_SIZE,
    #     shuffle=False,
    #     num_workers=config.NUM_WORKERS
    # )

    # ---- Model Initialization ----
    print("Initializing models...")
    image_encoder = ImageEncoder(
        model_name=config.ENCODER_MODEL_NAME,
        pretrained=config.PRETRAINED_ENCODER,
        output_embed_size=config.IMAGE_EMBED_SIZE,
        fine_tune_cnn=config.FINE_TUNE_CNN
    ).to(device)

    uncertainty_estimator = UncertaintyEstimator(
        vocab_size=len(vocab),  # vocab_size를 첫 번째 매개변수로 이동
        image_feature_dim=config.IMAGE_EMBED_SIZE, 
        d_model=config.UE_HIDDEN_SIZE, # Using specific UE hidden size
        num_layers=config.UE_NUM_LAYERS,
        num_heads=config.NUM_ATTENTION_HEADS, # Can share or have UE specific
        d_ff=config.UE_FEED_FORWARD_SIZE,    # Can share or have UE specific
        dropout=config.DROPOUT_PROB
    ).to(device)

    insertion_transformer = InsertionTransformer(
        vocab_size=len(vocab),
        d_model=config.HIDDEN_SIZE,
        num_layers=config.IT_NUM_LAYERS,
        num_heads=config.NUM_ATTENTION_HEADS,
        d_ff=config.FEED_FORWARD_SIZE,
        dropout=config.DROPOUT_PROB,
        image_feature_dim=config.IMAGE_EMBED_SIZE,
        max_seq_len=config.MAX_SEQ_LEN,
        pad_idx=pad_idx
    ).to(device)

    # PyTorch 2.0+ 모델 컴파일 (선택 사항, 호환성 확인 필요)
    # if hasattr(torch, 'compile') and tuple(map(int, torch.__version__.split('.')[:2])) >= (2, 0):
    #     print("Compiling models with torch.compile()...")
    #     try:
    #         image_encoder = torch.compile(image_encoder, mode='reduce-overhead')
    #         uncertainty_estimator = torch.compile(uncertainty_estimator, mode='reduce-overhead')
    #         insertion_transformer = torch.compile(insertion_transformer, mode='reduce-overhead')
    #         print("Models compiled successfully with reduce-overhead mode.")
    #     except Exception as e:
    #         print(f"Warning: Model compilation failed. Training will proceed without it. Error: {e}")
    #         # 컴파일 실패 시 원본 모델 사용
    #         pass
    # else:
    #     print("torch.compile() not available or PyTorch version < 2.0. Skipping model compilation.")

    # ---- Optimizer and Loss ----
    # Parameters from all models that need training
    params = (
        list(filter(lambda p: p.requires_grad, image_encoder.parameters())) +
        list(uncertainty_estimator.parameters()) +
        list(insertion_transformer.parameters())
    )
    optimizer = optim.Adam(params, lr=config.LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min',
        factor=config.LR_SCHEDULER_FACTOR,
        patience=config.LR_SCHEDULER_PATIENCE,
        min_lr=1e-6,
        cooldown=1
    )

    print(f"Models initialized. Total trainable parameters: {sum(p.numel() for p in params if p.requires_grad):,}")

    # ---- Training Loop ----
    print("\n=== 학습 시작 ===")
    torch.cuda.empty_cache()  # 학습 시작 전 메모리 정리
    
    epoch_pbar = tqdm(range(config.NUM_EPOCHS), desc="Epochs", position=0)
    
    for epoch in epoch_pbar:
        epoch_start_time = time.time()
        total_epoch_loss = 0
        total_word_loss_epoch = 0
        total_pos_loss_epoch = 0
        num_insertion_samples_epoch = 0

        image_encoder.train()
        uncertainty_estimator.train()
        insertion_transformer.train()

        batch_pbar = tqdm(enumerate(train_loader), total=len(train_loader), 
                         desc=f"Epoch {epoch+1}", position=1, leave=False)
        
        for i, (images, captions_batch, lengths_batch) in batch_pbar:
            # 데이터를 non_blocking=True로 GPU로 전송
            images = images.to(device, non_blocking=True)
            captions_batch = captions_batch.to(device, non_blocking=True)
            lengths_batch = lengths_batch.to(device, non_blocking=True)
            current_batch_size = images.size(0)

            # 그래디언트 누적을 위한 조건
            if i % config.ACCUMULATION_STEPS == 0:
                optimizer.zero_grad(set_to_none=True)

            # 메모리 최적화를 위한 torch.cuda.amp 사용
            with torch.amp.autocast('cuda', enabled=True):
                # Forward passes - 배치 병렬 처리
                img_features_global = image_encoder(images)
                pi_values = uncertainty_estimator(img_features_global)
                
                # 배치 전체를 한 번에 처리
                training_samples = []
                for b_idx in range(current_batch_size):
                    caption_single_ids = captions_batch[b_idx, :lengths_batch[b_idx]].tolist()
                    
                    # 불확실성 점수 검증 및 로깅 추가
                    current_pi_values = pi_values[b_idx]
                    if torch.isnan(current_pi_values).any() or torch.isinf(current_pi_values).any():
                        print(f"Warning: Found invalid uncertainty scores in batch {i}, sample {b_idx}")
                        continue
                        
                    # 불확실성 점수 범위 확인
                    pi_min, pi_max = current_pi_values.min().item(), current_pi_values.max().item()
                    if pi_min < -1 or pi_max > 0:
                        print(f"Warning: Uncertainty scores out of range [-1, 0]: min={pi_min}, max={pi_max}")
                        current_pi_values = torch.clamp(current_pi_values, -1, 0)
                    
                    samples = generate_insertion_training_samples(
                        true_caption_indices=caption_single_ids,
                        uncertainty_scores=current_pi_values,
                        vocab=vocab,
                        eos_id=eos_idx,
                        bos_id=bos_idx,
                        pad_id=pad_idx,
                        mask_id=vocab(config.MASK_TOKEN),
                        none_id=none_idx,
                        max_len=config.MAX_SEQ_LEN
                    )
                    
                    if samples:
                        training_samples.extend((b_idx, s) for s in samples)
                    
                    # 주기적으로 학습 상태 로깅
                    if i % config.LOG_STEP == 0 and b_idx == 0:
                        print(f"\nSample training sequence (batch {i}, sample 0):")
                        for s_idx, (seq, word, pos) in enumerate(samples[:3]):  # 처음 3개 샘플만 출력
                            seq_text = " ".join([vocab.get_word(idx) for idx in seq if idx != pad_idx])
                            word_text = vocab.get_word(word)
                            print(f"Step {s_idx}: Input='{seq_text}', Insert '{word_text}' at position {pos}")

                if not training_samples:
                    continue

                batch_losses = []
                sub_batch_size = 32  # 서브 배치 크기 설정
                
                for j in range(0, len(training_samples), sub_batch_size):
                    sub_batch = training_samples[j:j + sub_batch_size]
                    b_indices, samples = zip(*sub_batch)
                    
                    s_k_minus_1_padded, target_word_id, target_pos_idx = zip(*samples)
                    
                    partial_cap_tensor = torch.tensor(s_k_minus_1_padded, dtype=torch.long, device=device)
                    target_word_tensor = torch.tensor(target_word_id, dtype=torch.long, device=device)
                    target_pos_tensor = torch.tensor(target_pos_idx, dtype=torch.long, device=device)
                    batch_img_features = img_features_global[list(b_indices)]

                    it_word_logits, it_pos_logits = insertion_transformer(batch_img_features, partial_cap_tensor)
                    
                    word_loss, pos_loss, total_loss = compute_insertion_losses(
                        it_word_logits,
                        it_pos_logits,
                        target_word_tensor,
                        target_pos_tensor,
                        pad_idx
                    )
                    batch_losses.append((total_loss, word_loss, pos_loss, len(sub_batch)))

                # 배치의 총 손실 계산
                batch_total_loss = sum(loss * count for loss, _, _, count in batch_losses)
                batch_word_loss = sum(w_loss * count for _, w_loss, _, count in batch_losses)
                batch_pos_loss = sum(p_loss * count for _, _, p_loss, count in batch_losses)
                num_insertion_samples_batch = sum(count for _, _, _, count in batch_losses)

            if num_insertion_samples_batch > 0:
                avg_batch_loss = batch_total_loss / num_insertion_samples_batch
                
                # 그래디언트 누적을 위한 스케일링
                scaled_loss = avg_batch_loss / config.ACCUMULATION_STEPS
                scaler.scale(scaled_loss).backward()
                
                # 누적된 그래디언트로 옵티마이저 스텝
                if (i + 1) % config.ACCUMULATION_STEPS == 0:
                    # 그래디언트 클리핑
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(params, config.CLIP_GRAD_NORM)
                    
                    scaler.step(optimizer)
                    scaler.update()

                total_epoch_loss += avg_batch_loss.item() * num_insertion_samples_batch
                total_word_loss_epoch += (batch_word_loss.item() / num_insertion_samples_batch) * num_insertion_samples_batch
                total_pos_loss_epoch += (batch_pos_loss.item() / num_insertion_samples_batch) * num_insertion_samples_batch
                num_insertion_samples_epoch += num_insertion_samples_batch

                # Update batch progress bar
                avg_loss = total_epoch_loss / num_insertion_samples_epoch if num_insertion_samples_epoch > 0 else 0
                perplexity = math.exp(avg_loss) if avg_loss < 700 else float('inf')
                batch_pbar.set_postfix({
                    'loss': f'{avg_loss:.4f}',
                    'ppl': f'{perplexity:.2f}'
                })

        # End of epoch
        epoch_time = time.time() - epoch_start_time
        avg_epoch_total_loss = total_epoch_loss / num_insertion_samples_epoch if num_insertion_samples_epoch > 0 else 0
        avg_epoch_word_loss = total_word_loss_epoch / num_insertion_samples_epoch if num_insertion_samples_epoch > 0 else 0
        avg_epoch_pos_loss = total_pos_loss_epoch / num_insertion_samples_epoch if num_insertion_samples_epoch > 0 else 0
        perplexity = math.exp(avg_epoch_total_loss) if avg_epoch_total_loss < 700 else float('inf')

        # Update epoch progress bar
        epoch_pbar.set_postfix({
            'loss': f'{avg_epoch_total_loss:.4f}',
            'ppl': f'{perplexity:.2f}',
            'time': f'{epoch_time:.1f}s'
        })

        # 에포크 종료 후 학습률 조정
        scheduler.step(avg_epoch_total_loss)

        # Save model checkpoint
        checkpoint_path = os.path.join(config.MODEL_SAVE_DIR, f'model_epoch_{epoch+1}.pth')
        # config 모듈에서 필요한 설정값만 추출
        config_dict = {
            'HIDDEN_SIZE': config.HIDDEN_SIZE,
            'IT_NUM_LAYERS': config.IT_NUM_LAYERS,
            'NUM_ATTENTION_HEADS': config.NUM_ATTENTION_HEADS,
            'FEED_FORWARD_SIZE': config.FEED_FORWARD_SIZE,
            'DROPOUT_PROB': config.DROPOUT_PROB,
            'IMAGE_EMBED_SIZE': config.IMAGE_EMBED_SIZE,
            'MAX_SEQ_LEN': config.MAX_SEQ_LEN,
            'UE_HIDDEN_SIZE': config.UE_HIDDEN_SIZE,
            'UE_NUM_LAYERS': config.UE_NUM_LAYERS,
            'UE_FEED_FORWARD_SIZE': config.UE_FEED_FORWARD_SIZE
        }
        
        model_state = {
            'epoch': epoch + 1,
            'image_encoder_state_dict': image_encoder.state_dict(),
            'uncertainty_estimator_state_dict': uncertainty_estimator.state_dict(),
            'insertion_transformer_state_dict': insertion_transformer.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scaler_state_dict': scaler.state_dict() if scaler is not None else None,
            'vocab_path': config.VOCAB_PATH,
            'config': config_dict,
            'losses': {
                'total_loss': avg_epoch_total_loss,
                'word_loss': avg_epoch_word_loss,
                'pos_loss': avg_epoch_pos_loss,
                'perplexity': perplexity
            }
        }
        torch.save(model_state, checkpoint_path)

    print("\n학습이 모두 완료되었습니다!")

if __name__ == '__main__':
    main() 