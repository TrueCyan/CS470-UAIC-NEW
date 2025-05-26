# 평가 스크립트 구현 예정 

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import os
import json
from tqdm import tqdm
import time
from collections import defaultdict
import gc

# Robust import for project modules
try:
    from . import config
    from .utils.data_loader import get_loader
    from .utils.vocabulary import Vocabulary
    from .utils.beam_search import beam_search_inserter
    from .models.image_encoder import ImageEncoder
    # Uncertainty Estimator might not be directly used in beam search eval unless guiding it
    # from .models.uncertainty_estimator import UncertaintyEstimator 
    from .models.insertion_transformer import InsertionTransformer
    # For COCO evaluation metrics
    from pycocoevalcap.bleu.bleu import Bleu
    from pycocoevalcap.rouge.rouge import Rouge
    from pycocoevalcap.cider.cider import Cider
    from pycocoevalcap.meteor.meteor import Meteor
    from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
except ImportError:
    import sys
    current_dir = os.path.dirname(os.path.abspath(__file__)) # src/
    project_root = os.path.dirname(current_dir) # UAIC_NEW/
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    
    import config # type: ignore
    from utils.data_loader import get_loader # type: ignore
    from utils.vocabulary import Vocabulary # type: ignore
    from utils.beam_search import beam_search_inserter # type: ignore
    from models.image_encoder import ImageEncoder # type: ignore
    # from models.uncertainty_estimator import UncertaintyEstimator # type: ignore
    from models.insertion_transformer import InsertionTransformer # type: ignore
    from pycocoevalcap.bleu.bleu import Bleu # type: ignore
    from pycocoevalcap.rouge.rouge import Rouge # type: ignore
    from pycocoevalcap.cider.cider import Cider # type: ignore
    from pycocoevalcap.meteor.meteor import Meteor # type: ignore
    from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer # type: ignore

def load_models(model_path, vocab, device):
    """모델 로드 및 설정"""
    print(f"\n모델 로드 중... ({model_path})")
    checkpoint = torch.load(model_path, map_location=device)
    
    # 체크포인트에서 설정 로드
    loaded_config = checkpoint['config']
    for key, value in loaded_config.items():
        setattr(config, key, value)
    
    # 이미지 인코더 초기화 및 로드
    image_encoder = ImageEncoder(
        model_name=config.ENCODER_MODEL_NAME,
        pretrained=config.PRETRAINED_ENCODER,
        output_embed_size=config.IMAGE_EMBED_SIZE,
        fine_tune_cnn=config.FINE_TUNE_CNN
    ).to(device)
    image_encoder.load_state_dict(checkpoint['image_encoder_state_dict'])
    
    # 삽입 트랜스포머 초기화 및 로드
    insertion_transformer = InsertionTransformer(
        vocab_size=len(vocab),
        d_model=config.HIDDEN_SIZE,
        num_layers=config.IT_NUM_LAYERS,
        num_heads=config.NUM_ATTENTION_HEADS,
        d_ff=config.FEED_FORWARD_SIZE,
        dropout=0.0,  # 평가시에는 드롭아웃 비활성화
        image_feature_dim=config.IMAGE_EMBED_SIZE,
        max_seq_len=config.MAX_SEQ_LEN,
        pad_idx=vocab(config.PAD_TOKEN)
    ).to(device)
    insertion_transformer.load_state_dict(checkpoint['insertion_transformer_state_dict'])
    
    # 평가 모드로 설정
    image_encoder.eval()
    insertion_transformer.eval()
    
    # 불필요한 메모리 정리
    del checkpoint
    torch.cuda.empty_cache()
    gc.collect()
    
    print("모델 로드 완료")
    return image_encoder, insertion_transformer

def calculate_batch_metrics(gts, res):
    """단일 배치에 대한 평가 지표 계산"""
    tokenizer = PTBTokenizer()
    gts = tokenizer.tokenize(gts)
    res = tokenizer.tokenize(res)
    
    bleu_scorer = Bleu(4)
    bleu_scores, _ = bleu_scorer.compute_score(gts, res)
    
    cider_scorer = Cider()
    cider_score, _ = cider_scorer.compute_score(gts, res)
    
    return {
        'Bleu_1': bleu_scores[0],
        'Bleu_4': bleu_scores[3],
        'CIDEr': cider_score
    }

def evaluate_model(model_path: str):
    """모델 평가 실행"""
    start_time = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() and config.USE_CUDA else "cpu")
    print(f"평가 장치: {device}")
    
    # CUDA 최적화 설정
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    
    # 어휘 사전 로드
    print("\n어휘 사전 로드 중...")
    try:
        vocab = Vocabulary.load_vocab(config.VOCAB_PATH)
        print(f"어휘 사전 로드 완료 (크기: {len(vocab)})")
    except FileNotFoundError:
        print(f"오류: 어휘 사전을 찾을 수 없습니다 ({config.VOCAB_PATH})")
        return
    
    # 모델 로드
    image_encoder, insertion_transformer = load_models(model_path, vocab, device)
    
    # 테스트 데이터 로더 설정
    print("\n테스트 데이터 로드 중...")
    test_loader = get_loader(
        image_dir=config.TEST_IMAGE_DIR,
        json_path=config.TEST_CAPTION_JSON,
        vocab=vocab,
        batch_size=1,
        split='test',
        shuffle=False,
        num_workers=min(4, os.cpu_count() or 1),
        pin_memory=True
    )
    
    # 평가 결과 저장용 딕셔너리
    results_gts = {}
    results_res = {}
    running_scores = defaultdict(float)
    num_samples = 0
    
    # 진행 상황 표시
    print("\n평가 시작...")
    pbar = tqdm(test_loader, desc="평가 중")
    
    # AMP 설정
    scaler = torch.cuda.amp.GradScaler()
    
    for i, (images, captions, lengths) in enumerate(pbar):
        images = images.to(device, non_blocking=True)
        
        try:
            # 이미지 특징 추출 및 캡션 생성
            with torch.no_grad(), torch.cuda.amp.autocast():
                # 이미지 특징 추출
                img_features = image_encoder(images)
                
                # 캡션 생성 (빔 서치)
                generated_tokens = beam_search_inserter(
                    image_features=img_features,
                    model=insertion_transformer,
                    vocab=vocab,
                    beam_size=config.EVAL_BEAM_SIZE,
                    max_iterations=config.MAX_SEQ_LEN,
                    max_len=config.MAX_SEQ_LEN,
                    device=device
                )[0]  # 최상위 빔만 사용
            
            # 생성된 캡션과 실제 캡션을 텍스트로 변환
            generated_text = " ".join([vocab.get_word(idx) for idx in generated_tokens])
            real_caption = " ".join([vocab.get_word(idx.item()) 
                                   for idx in captions[0, :lengths[0]]])
            
            # 결과 저장
            image_id = str(i)
            results_res[image_id] = [generated_text]
            results_gts[image_id] = [real_caption]
            
            # 매 스텝마다 캡션 출력
            print(f"\n{'='*100}")
            print(f"샘플 {i+1}/{len(test_loader)}:")
            print(f"생성된 캡션: {generated_text}")
            print(f"실제 캡션:   {real_caption}")
            
            # 현재 배치의 점수 계산 (LOG_STEP 마다)
            if (i + 1) % config.LOG_STEP == 0:
                batch_scores = calculate_batch_metrics(
                    {image_id: results_gts[image_id]},
                    {image_id: results_res[image_id]}
                )
                
                # 점수 누적 및 평균 계산
                for metric, score in batch_scores.items():
                    running_scores[metric] += score
                num_samples += 1
                
                avg_scores = {k: v/num_samples for k, v in running_scores.items()}
                
                # 진행 상황 표시 업데이트
                pbar.set_postfix({
                    'BLEU-1': f"{avg_scores['Bleu_1']:.3f}",
                    'BLEU-4': f"{avg_scores['Bleu_4']:.3f}",
                    'CIDEr': f"{avg_scores['CIDEr']:.3f}"
                })
                
                print(f"현재 BLEU-1: {batch_scores['Bleu_1']:.3f}, BLEU-4: {batch_scores['Bleu_4']:.3f}, CIDEr: {batch_scores['CIDEr']:.3f}")
            print(f"{'='*100}")
            
            # 메모리 정리
            if (i + 1) % 100 == 0:
                torch.cuda.empty_cache()
                gc.collect()
                
        except Exception as e:
            print(f"\n이미지 {i} 처리 중 오류 발생: {str(e)}")
            continue
    
    # 최종 평가 지표 계산
    print("\n최종 평가 지표 계산 중...")
    tokenizer = PTBTokenizer()
    gts = tokenizer.tokenize(results_gts)
    res = tokenizer.tokenize(results_res)
    
    scorers = [
        (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
        (Meteor(), "METEOR"),
        (Rouge(), "ROUGE_L"),
        (Cider(), "CIDEr")
    ]
    
    final_scores = {}
    for scorer, method in scorers:
        score, _ = scorer.compute_score(gts, res)
        if isinstance(method, list):
            for sc, meth in zip(score, method):
                final_scores[meth] = sc
        else:
            final_scores[method] = score
    
    # 결과 저장
    results = {
        'scores': final_scores,
        'examples': {
            'generated': dict(list(results_res.items())[:5]),
            'ground_truth': dict(list(results_gts.items())[:5])
        },
        'config': loaded_config
    }
    
    output_path = os.path.join(
        config.MODEL_SAVE_DIR, 
        f'eval_results_{os.path.basename(model_path).replace(".pth","")}.json'
    )
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    # 최종 결과 출력
    print(f"\n평가 완료 (소요 시간: {time.time() - start_time:.1f}초)")
    print("\n최종 평가 결과:")
    for metric, score in final_scores.items():
        print(f"{metric:8s}: {score:.4f}")
    print(f"\n상세 결과가 저장된 경로: {output_path}")
    
    return final_scores

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='이미지 캡셔닝 모델 평가')
    parser.add_argument('--model_path', type=str, required=True, 
                      help='평가할 모델 체크포인트 경로 (.pth 파일)')
    args = parser.parse_args()
    
    if not os.path.exists(args.model_path):
        print(f"오류: 모델 파일을 찾을 수 없습니다 ({args.model_path})")
    else:
        evaluate_model(args.model_path) 