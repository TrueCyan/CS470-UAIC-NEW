# 데이터 로딩 및 전처리 구현 예정 

import os
import json
import nltk # For tokenization in case Vocabulary doesn't do it or for raw captions
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from typing import List, Tuple, Any, Callable

# Robust import for config and Vocabulary
try:
    from .vocabulary import Vocabulary
    from .. import config
except ImportError:
    import sys
    # Add the project root to sys.path if running this script directly for testing
    # This assumes the script is in src/utils/
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir) # This should be src/
    project_root = os.path.dirname(parent_dir) # This should be UAIC_NEW/
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    if parent_dir not in sys.path:
         sys.path.insert(0, parent_dir) # for `from utils import ...` if in src/

    from utils.vocabulary import Vocabulary # if running from src/ or project_root/
    import config

class CocoDataset(Dataset):
    """COCO Custom Dataset compatible with torch.utils.data.Dataset."""
    def __init__(self, image_dir: str, json_path: str, vocab: Vocabulary, transform: Callable, split: str = 'train'):
        """Set the path for images, captions and vocabulary wrapper.
        Args:
            image_dir: Directory with all COCO images.
            json_path: Path to Karpathy-split COCO annotation JSON file.
            vocab: Vocabulary wrapper.
            transform:  Transform to be applied to images.
            split: 'train', 'val', or 'test'.
        """
        self.image_dir = image_dir
        self.vocab = vocab
        self.transform = transform
        self.split = split
        self.annotations = []

        with open(json_path, 'r') as f:
            data = json.load(f)

        for item in data['images']:
            if item['split'] == self.split:
                # Each image can have multiple captions
                for sentence_data in item['sentences']:
                    self.annotations.append({
                        'image_path': os.path.join(self.image_dir, item['filename']),
                        'caption': sentence_data['raw'].lower(),
                        'tokens': sentence_data['tokens'] # Assuming pre-tokenized by Karpathy split
                    })
            # For Flick_val and Flick_test, the Karpathy format might be slightly different or not present
            # This loader is primarily designed for Karpathy COCO splits.
            # If 'restval' images are used for training as is common, json_path should point to such a combined file.

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """Returns one data pair (image and caption)."""
        annotation = self.annotations[index]
        image_path = annotation['image_path']
        caption_str = annotation['caption']

        try:
            image = Image.open(image_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
        except FileNotFoundError:
            print(f"Warning: Image file not found {image_path}. Skipping or returning placeholder.")
            # Return a placeholder or handle appropriately
            # For now, let's assume files are always present or raise an error if critical
            # Fallback to a black image if not found, to prevent crashing, but this is not ideal.
            image = torch.zeros((3, 256, 256)) # Assuming a common size after transform
            if self.transform: # Try to apply some part of transform if possible or ensure tensor
                 # This part is tricky, as transform might expect PIL image.
                 # Best is to ensure data integrity.
                 pass 

        # Convert caption (string) to word ids.
        # Using pre-tokenized tokens from Karpathy if available and Vocabulary handles them.
        # Otherwise, tokenize with NLTK as in Vocabulary.add_sentence
        # For consistency, we can use vocab.sentence_to_indices which uses NLTK
        
        tokens = nltk.tokenize.word_tokenize(str(caption_str).lower())
        caption_indices = []
        caption_indices.append(self.vocab(config.BOS_TOKEN))
        caption_indices.extend([self.vocab(token) for token in tokens])
        caption_indices.append(self.vocab(config.EOS_TOKEN))
        
        # Truncate or pad to MAX_SEQ_LEN (padding will be done in collate_fn)
        # Here we just truncate if it's too long, considering BOS and EOS
        if len(caption_indices) > config.MAX_SEQ_LEN:
            caption_indices = caption_indices[:config.MAX_SEQ_LEN-1] + [self.vocab(config.EOS_TOKEN)]
        
        caption_len = len(caption_indices)
        caption_tensor = torch.tensor(caption_indices, dtype=torch.long)

        return image, caption_tensor, caption_len

    def __len__(self) -> int:
        return len(self.annotations)

def collate_fn(data: List[Tuple[torch.Tensor, torch.Tensor, int]]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Creates mini-batch tensors from the list of tuples (image, caption, length)."""
    # Sort a data list by caption length (descending order).
    # This is not strictly necessary for Transformer if using padding masks correctly,
    # but can be useful for PackedSequence with RNNs.
    # data.sort(key=lambda x: x[2], reverse=True)
    images, captions, lengths = zip(*data)

    # Merge images (from tuple of 3D tensor to 4D tensor).
    images_batch = torch.stack(images, 0)

    # Merge captions (from tuple of 1D tensor to 2D tensor).
    # Pad all captions to MAX_SEQ_LEN, not just max_len in batch, for consistency
    padded_captions = torch.full((len(captions), config.MAX_SEQ_LEN), config.PAD_IDX, dtype=torch.long)
    for i, cap in enumerate(captions):
        end = lengths[i]
        padded_captions[i, :end] = cap[:end]
    
    lengths_batch = torch.tensor(lengths, dtype=torch.long)
    return images_batch, padded_captions, lengths_batch

def get_loader(image_dir: str, 
               json_path: str, 
               vocab: Vocabulary, 
               batch_size: int, 
               split: str = 'train', 
               shuffle: bool = True, 
               num_workers: int = 0, # Set to 0 for debugging on Windows, >0 for performance
               pin_memory: bool = False) -> DataLoader:
    """Returns torch.utils.data.DataLoader for COCO dataset."""
    # COCO dataset image normalization parameters
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    if split == 'train':
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ])
    else: # 'val' or 'test'
        transform = transforms.Compose([
            transforms.Resize(224), # Ensure same input size as training
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize
        ])

    dataset = CocoDataset(image_dir=image_dir,
                          json_path=json_path,
                          vocab=vocab,
                          transform=transform,
                          split=split)
    
    data_loader = DataLoader(dataset=dataset,
                               batch_size=batch_size,
                               shuffle=shuffle if split == 'train' else False, # No shuffle for val/test
                               num_workers=num_workers,
                               collate_fn=collate_fn,
                               pin_memory=pin_memory)
    return data_loader

if __name__ == '__main__':
    # This is a basic test. 
    # For it to run, you need:
    # 1. A vocabulary file (e.g., vocab.pkl) created by vocabulary.py
    # 2. COCO images in a directory structure like: data/mscoco/train2014/COCO_train2014_...jpg
    # 3. Karpathy split JSON file (e.g., data/karpathy_split/dataset_coco.json)
    # Adjust paths in config.py and below accordingly.

    print("Starting DataLoader test...")

    # Ensure NLTK data is downloaded (run this once if you haven't)
    try:
        nltk.data.find('tokenizers/punkt')
    except nltk.downloader.DownloadError:
        print("NLTK 'punkt' tokenizer not found. Downloading...")
        nltk.download('punkt')
        print("Downloaded 'punkt'. Please re-run the script if it failed initially.")

    # Create a dummy vocab for testing if not available
    if not os.path.exists(config.VOCAB_PATH):
        print(f"Vocabulary file not found at {config.VOCAB_PATH}. Creating a dummy one for testing.")
        # Create dummy captions to build a vocab
        dummy_captions_for_vocab = [
            "a man is riding a horse", 
            "a woman is walking on the beach"
        ]
        vocab = Vocabulary()
        vocab.build_vocab(dummy_captions_for_vocab, min_word_freq=1)
        vocab.save_vocab(config.VOCAB_PATH)
        print(f"Dummy vocabulary saved to {config.VOCAB_PATH}")
    else:
        vocab = Vocabulary.load_vocab(config.VOCAB_PATH)
        print(f"Loaded vocabulary from {config.VOCAB_PATH}")

    print(f"Vocabulary size: {len(vocab)}")

    # --- Adjust these paths according to your setup --- 
    # Assuming your images are in data/mscoco/ and Karpathy JSON in data/karpathy_split/
    # The CocoDataset expects image_dir to be the root of train2014, val2014 etc.
    # Karpathy JSON has fields like item['filepath'] which is 'train2014' or 'val2014'
    # So image_dir should be config.MSCOCO_DIR (e.g. "data/mscoco/")
    
    # Create dummy annotation file if it doesn't exist
    dummy_json_path = os.path.join(config.KARPATHY_SPLIT_DIR, "dummy_dataset_coco.json")
    if not os.path.exists(dummy_json_path):
        print(f"Dummy annotation file not found at {dummy_json_path}. Creating one.")
        os.makedirs(config.KARPATHY_SPLIT_DIR, exist_ok=True)
        dummy_data = {"images": [
            {
                "sentids": [0, 1],
                "imgid": 0,
                "sentences": [
                    {"tokens": ["a", "cat", "sitting", "on", "a", "mat"], "raw": "A cat sitting on a mat.", "imgid": 0, "sentid": 0},
                    {"tokens": ["the", "cat", "is", "black"], "raw": "The cat is black.", "imgid": 0, "sentid": 1}
                ],
                "split": "train",
                "filename": "dummy_image_1.jpg", # You'd need dummy image files too for full test
                "filepath": "train_dummy_data"
            },
            {
                "sentids": [2],
                "imgid": 1,
                "sentences": [
                    {"tokens": ["a", "dog", "running", "in", "the", "park"], "raw": "A dog running in the park.", "imgid": 1, "sentid": 2}
                ],
                "split": "train",
                "filename": "dummy_image_2.jpg",
                "filepath": "val_dummy_data"
            }
        ]}
        with open(dummy_json_path, 'w') as f:
            json.dump(dummy_data, f)
        print(f"Dummy annotation file created at {dummy_json_path}")
        
        # Create dummy image files and directories
        os.makedirs(os.path.join(config.MSCOCO_DIR, "train_dummy_data"), exist_ok=True)
        os.makedirs(os.path.join(config.MSCOCO_DIR, "val_dummy_data"), exist_ok=True)
        try:
            Image.new('RGB', (100, 100), color = 'red').save(os.path.join(config.MSCOCO_DIR, "train_dummy_data", "dummy_image_1.jpg"))
            Image.new('RGB', (100, 100), color = 'blue').save(os.path.join(config.MSCOCO_DIR, "val_dummy_data", "dummy_image_2.jpg"))
            print("Dummy image files created.")
        except Exception as e:
            print(f"Could not create dummy image files: {e}")


    print(f"Using MSCOCO_DIR: {config.MSCOCO_DIR}")
    print(f"Using dummy Karpathy JSON: {dummy_json_path}")

    # Get data loader
    try:
        train_loader = get_loader(
            image_dir=config.MSCOCO_DIR, # This should be the root, e.g., data/mscoco/
            json_path=dummy_json_path, # Path to Karpathy split JSON
            vocab=vocab,
            batch_size=2, # Small batch for testing
            split='train',
            shuffle=True,
            num_workers=0 # Use 0 for easier debugging
        )
        print("Train DataLoader created successfully.")

        # Fetch a batch
        for i, (images, captions, lengths) in enumerate(train_loader):
            print(f"--- Batch {i+1} ---")
            print("Images shape:", images.shape)
            print("Captions shape:", captions.shape)
            print("Lengths:", lengths)
            print("Sample caption (indices):", captions[0])
            print("Sample caption (text):", vocab.indices_to_sentence(captions[0].tolist()))
            if i == 0: # Just test one batch
                break 
        print("Successfully fetched and processed a batch.")

    except Exception as e:
        print(f"Error during DataLoader test: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Clean up dummy files and directories
        if os.path.exists(dummy_json_path):
            #os.remove(dummy_json_path)
            print(f"Kept dummy json: {dummy_json_path}") # For inspection
        if os.path.exists(os.path.join(config.MSCOCO_DIR, "train_dummy_data", "dummy_image_1.jpg")):
            #os.remove(os.path.join(config.MSCOCO_DIR, "train_dummy_data", "dummy_image_1.jpg"))
            pass # Keep for inspection
        if os.path.exists(os.path.join(config.MSCOCO_DIR, "val_dummy_data", "dummy_image_2.jpg")):
            #os.remove(os.path.join(config.MSCOCO_DIR, "val_dummy_data", "dummy_image_2.jpg"))
            pass # Keep for inspection
        # Potentially remove dummy dirs if they were created and are empty, but be cautious
        # if os.path.exists(config.VOCAB_PATH) and "dummy_captions_for_vocab" in open(config.VOCAB_PATH, 'rb').read().decode(errors='ignore')):
            # os.remove(config.VOCAB_PATH) # Clean dummy vocab if it was created by this script

    print("DataLoader test finished.") 