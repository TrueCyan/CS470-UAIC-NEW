# 이미지 인코더 구현 예정 

import torch
import torch.nn as nn
import torchvision.models as models

# Robust import for config
try:
    from .. import config
except ImportError:
    import sys
    import os
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    project_root = os.path.dirname(parent_dir)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    if parent_dir not in sys.path:
         sys.path.insert(0, parent_dir)
    import config

class ImageEncoder(nn.Module):
    def __init__(self, model_name: str = config.ENCODER_MODEL_NAME, 
                 pretrained: bool = config.PRETRAINED_ENCODER,
                 output_embed_size: int = config.IMAGE_EMBED_SIZE,
                 fine_tune_cnn: bool = False): # Set to True to fine-tune earlier layers
        super(ImageEncoder, self).__init__()
        self.model_name = model_name
        self.output_embed_size = output_embed_size

        # Load a pretrained CNN and remove the final classification layer
        if model_name.startswith('resnet'):
            cnn = getattr(models, model_name)(pretrained=pretrained)
            modules = list(cnn.children())[:-1]  # Remove the last fc layer
            self.cnn = nn.Sequential(*modules)
            # The output feature size of ResNet (before fc layer) is cnn.fc.in_features
            cnn_feature_size = cnn.fc.in_features 
        elif model_name.startswith('efficientnet'):
            cnn = getattr(models, model_name)(pretrained=pretrained)
            # EfficientNet's features are in cnn.features, and classifier is cnn.classifier
            self.cnn = cnn.features
            cnn_feature_size = cnn.classifier[1].in_features # Access the in_features of the classifier fc layer
        else:
            raise ValueError(f"Unsupported encoder model: {model_name}. Choose from resnet* or efficientnet*")

        # Optional: Linear layer to transform CNN output to desired embedding size
        # This is useful if IMAGE_EMBED_SIZE (transformer hidden size) is different from CNN output feature size
        # Or if we want a learnable projection.
        # For many architectures (like ResNet after avgpool), the output is (batch, cnn_feature_size, 1, 1)
        # We typically squeeze this to (batch, cnn_feature_size)
        self.fc = nn.Linear(cnn_feature_size, output_embed_size) 
        # No, for ResNet this adaptive_pool makes it BxCx1x1, so squeeze is needed.
        # Let's refine the fc layer based on typical usage pattern for captioning where image features are flattened.
        # The output of self.cnn (after avgpool for ResNet) will be [batch_size, cnn_feature_size, 1, 1]
        # We need to flatten it to [batch_size, cnn_feature_size] before passing to fc.
        # So, the fc layer is correct. An adaptive pooling layer is usually part of the CNN model itself for ResNets (avgpool).

        if not fine_tune_cnn:
            for param in self.cnn.parameters():
                param.requires_grad = False
        else: # Option to fine-tune later layers, e.g., last few blocks of ResNet
            # Example: fine-tune layers from layer4 onwards for ResNet
            if model_name.startswith('resnet') and hasattr(cnn, 'layer4'):
                # Freeze layers before layer4 for ResNet
                for name, child in cnn.named_children():
                    if name not in ['layer4', 'avgpool', 'fc']:
                        for param in child.parameters():
                            param.requires_grad = False
                    else:
                        print(f"Fine-tuning CNN layer: {name}")
            # For EfficientNet, fine-tuning specific blocks might need more granular control.
            # For simplicity, fine_tune_cnn=True will make all cnn params trainable if not ResNet.
            pass # By default, if fine_tune_cnn is True and not a special ResNet case, all cnn params are trainable.

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Extract feature vectors from input images.
        Args:
            images: A tensor of shape (batch_size, 3, image_height, image_width).
        Returns:
            A tensor of shape (batch_size, cnn_feature_size) or (batch_size, output_embed_size) 
            if an fc layer is used. For captioning, usually (batch_size, num_pixels, embed_size) 
            or (batch_size, embed_size) for global features.
            This implementation provides global features: (batch_size, output_embed_size).
        """
        features = self.cnn(images) # (batch_size, cnn_feature_size, H', W') or (batch_size, cnn_feature_size, 1, 1)
        
        # For models like ResNet ending with AdaptiveAvgPool2d, features are (batch, cnn_feature_size, 1, 1)
        # For EfficientNet.features, output is (batch, cnn_feature_size, H', W')
        # We need to adapt this for the fc layer.
        # A common approach for global features is to use an adaptive average pooling layer if not already present,
        # then flatten, then fc.

        if self.model_name.startswith('efficientnet'):
            # EfficientNet's features output shape is (batch, channels, H, W)
            # We apply adaptive average pooling to get (batch, channels, 1, 1)
            features = F.adaptive_avg_pool2d(features, 1)

        # Flatten the features: (batch_size, cnn_feature_size, H', W') -> (batch_size, cnn_feature_size * H' * W')
        # Or if (batch_size, cnn_feature_size, 1, 1) -> (batch_size, cnn_feature_size)
        features = features.view(features.size(0), -1)
        
        # Linear projection to output_embed_size
        output_features = self.fc(features) # (batch_size, output_embed_size)
        
        return output_features

if __name__ == '__main__':
    # Example Usage
    print(f"Testing ImageEncoder with model: {config.ENCODER_MODEL_NAME}, pretrained: {config.PRETRAINED_ENCODER}")
    print(f"Output embedding size: {config.IMAGE_EMBED_SIZE}")

    # Test with ResNet
    try:
        encoder_resnet = ImageEncoder(model_name='resnet50', pretrained=True, output_embed_size=config.HIDDEN_SIZE, fine_tune_cnn=False)
        encoder_resnet.eval() # Set to evaluation mode
        dummy_image_batch_resnet = torch.randn(2, 3, 224, 224) # Batch of 2 images, 224x224
        features_resnet = encoder_resnet(dummy_image_batch_resnet)
        print(f"ResNet Encoder - Input shape: {dummy_image_batch_resnet.shape}, Output features shape: {features_resnet.shape}")
        assert features_resnet.shape == (2, config.HIDDEN_SIZE)
        print("ResNet ImageEncoder tested successfully.")
    except Exception as e:
        print(f"Error testing ResNet ImageEncoder: {e}")

    # Test with EfficientNet
    try:
        encoder_effnet = ImageEncoder(model_name='efficientnet_b0', pretrained=True, output_embed_size=config.HIDDEN_SIZE, fine_tune_cnn=False)
        encoder_effnet.eval()
        dummy_image_batch_effnet = torch.randn(2, 3, 224, 224) # EfficientNet B0 also typically takes 224x224
        features_effnet = encoder_effnet(dummy_image_batch_effnet)
        print(f"EfficientNet Encoder - Input shape: {dummy_image_batch_effnet.shape}, Output features shape: {features_effnet.shape}")
        assert features_effnet.shape == (2, config.HIDDEN_SIZE)
        print("EfficientNet ImageEncoder tested successfully.")
    except Exception as e:
        print(f"Error testing EfficientNet ImageEncoder: {e}")

    # Test fine-tuning (just checks if parameters are trainable)
    try:
        encoder_finetune = ImageEncoder(model_name='resnet18', pretrained=True, output_embed_size=config.HIDDEN_SIZE, fine_tune_cnn=True)
        trainable_params = sum(p.numel() for p in encoder_finetune.parameters() if p.requires_grad)
        print(f"ResNet18 Encoder (fine_tune=True) - Trainable parameters: {trainable_params}")
        assert trainable_params > 0
        # Check if some early layers are frozen for resnet* if fine_tune_cnn=True specified deeper layer fine-tuning
        # This specific check depends on the fine_tune_cnn logic, current logic fine-tunes layer4 onwards for ResNet
        if hasattr(encoder_finetune.cnn, 'layer3'):
            layer3_params_frozen = all(not p.requires_grad for p in encoder_finetune.cnn.layer3.parameters())
            print(f"ResNet18 Encoder (fine_tune=True) - Layer 3 params frozen: {layer3_params_frozen}")
            assert layer3_params_frozen, "Layer 3 should be frozen when fine_tune_cnn=True for ResNet implies layer4+ fine-tuning"
        print("ImageEncoder fine-tuning option tested successfully.")
    except Exception as e:
        print(f"Error testing ImageEncoder fine-tuning: {e}") 