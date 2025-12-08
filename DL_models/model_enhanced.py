"""
Enhanced Deep Learning Models with Improved Architecture
Features:
- Better initialization
- Attention mechanisms
- Residual connections
- Label smoothing
- Better regularization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models import efficientnet_b3, EfficientNet_B3_Weights

class SelfAttention(nn.Module):
    """Self-attention mechanism for feature refinement"""
    def __init__(self, in_dim):
        super(SelfAttention, self).__init__()
        self.query = nn.Linear(in_dim, in_dim // 8)
        self.key = nn.Linear(in_dim, in_dim // 8)
        self.value = nn.Linear(in_dim, in_dim)
        self.gamma = nn.Parameter(torch.zeros(1))
        
    def forward(self, x):
        batch_size, dim = x.size()
        
        q = self.query(x).view(batch_size, -1, 1)
        k = self.key(x).view(batch_size, -1, 1)
        v = self.value(x).view(batch_size, -1, 1)
        
        attention = torch.softmax(torch.bmm(q.transpose(1, 2), k), dim=-1)
        out = torch.bmm(v, attention.transpose(1, 2)).view(batch_size, dim)
        
        return self.gamma * out + x

class EnhancedFusionAttention(nn.Module):
    """Enhanced fusion with multi-head attention and residual connections"""
    def __init__(self, image_dim, feature_dim, fusion_dim, num_heads=4):
        super(EnhancedFusionAttention, self).__init__()
        self.num_heads = num_heads
        self.fusion_dim = fusion_dim
        
        # Projection layers
        self.proj_img = nn.Sequential(
            nn.Linear(image_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        self.proj_feat = nn.Sequential(
            nn.Linear(feature_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Multi-head attention
        self.multihead_attn = nn.MultiheadAttention(fusion_dim, num_heads, dropout=0.1)
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(fusion_dim * 2, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.ReLU()
        )
        
    def forward(self, image_features, extra_features):
        # Project features
        proj_img = self.proj_img(image_features)
        proj_feat = self.proj_feat(extra_features)
        
        # Prepare for multi-head attention: (seq_len, batch, embed_dim)
        proj_img_exp = proj_img.unsqueeze(0)
        proj_feat_exp = proj_feat.unsqueeze(0)
        
        # Apply cross-attention
        attn_out, _ = self.multihead_attn(proj_img_exp, proj_feat_exp, proj_feat_exp)
        attn_out = attn_out.squeeze(0)
        
        # Concatenate and project
        combined = torch.cat([attn_out, proj_img], dim=1)
        fused = self.output_proj(combined)
        
        return fused

class EnhancedResNet_IF(nn.Module):
    """Enhanced ResNet50 for image-only features"""
    def __init__(self, num_classes=3, dropout_rate=0.5):
        super(EnhancedResNet_IF, self).__init__()
        
        # Use ResNet50 for better feature extraction
        try:
            self.backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        except AttributeError:
            # Fallback for older torchvision versions
            self.backbone = resnet50(pretrained=True)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        
        # Enhanced classifier with attention
        self.attention = SelfAttention(in_features)
        
        self.classifier = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout_rate / 2),
            
            nn.Linear(256, num_classes)
        )
        
        # Initialize weights
        self._initialize_weights()
        
    def _initialize_weights(self):
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
    def forward(self, images):
        features = self.backbone(images)
        features = self.attention(features)
        output = self.classifier(features)
        return output

class EnhancedResNet_IFOF(nn.Module):
    """Enhanced ResNet50 with image and additional features fusion"""
    def __init__(self, image_input_dim=2048, feature_input_dim=709, num_classes=3, 
                 fusion_dim=512, dropout_rate=0.5):
        super(EnhancedResNet_IFOF, self).__init__()
        
        # Use ResNet50 for better feature extraction
        try:
            self.backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        except AttributeError:
            # Fallback for older torchvision versions
            self.backbone = resnet50(pretrained=True)
        self.backbone.fc = nn.Identity()
        
        # Enhanced fusion module
        self.fusion = EnhancedFusionAttention(
            image_dim=image_input_dim,
            feature_dim=feature_input_dim,
            fusion_dim=fusion_dim,
            num_heads=8
        )
        
        # Self-attention on fused features
        self.attention = SelfAttention(fusion_dim)
        
        # Enhanced classifier
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout_rate / 2),
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout_rate / 4),
            
            nn.Linear(128, num_classes)
        )
        
        self._initialize_weights()
        
    def _initialize_weights(self):
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
    def forward(self, images, features):
        # Handle sequence of images
        batch_size, seq_len, c, h, w = images.size()
        images = images.view(batch_size * seq_len, c, h, w)
        image_features = self.backbone(images)
        image_features = image_features.view(batch_size, seq_len, -1)
        image_features = image_features[:, -1, :]  # Use last frame
        
        # Flatten extra features
        features = features.view(batch_size, -1)
        
        # Fuse features
        fused_features = self.fusion(image_features, features)
        fused_features = self.attention(fused_features)
        
        # Classify
        output = self.classifier(fused_features)
        return output

class EnhancedEfficientNet_IF(nn.Module):
    """Enhanced EfficientNet-B3 for image-only features"""
    def __init__(self, num_classes=3, dropout_rate=0.5):
        super(EnhancedEfficientNet_IF, self).__init__()
        
        # Use EfficientNet-B3 for better accuracy
        try:
            self.backbone = efficientnet_b3(weights=EfficientNet_B3_Weights.IMAGENET1K_V1)
        except AttributeError:
            # Fallback for older torchvision versions
            self.backbone = efficientnet_b3(pretrained=True)
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Identity()
        
        # Enhanced classifier with attention
        self.attention = SelfAttention(in_features)
        
        self.classifier = nn.Sequential(
            nn.Linear(in_features, 768),
            nn.BatchNorm1d(768),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(768, 384),
            nn.BatchNorm1d(384),
            nn.ReLU(),
            nn.Dropout(dropout_rate / 2),
            
            nn.Linear(384, num_classes)
        )
        
        self._initialize_weights()
        
    def _initialize_weights(self):
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
    def forward(self, images):
        features = self.backbone(images)
        features = self.attention(features)
        output = self.classifier(features)
        return output

class EnhancedEfficientNet_IFOF(nn.Module):
    """Enhanced EfficientNet-B3 with image and additional features fusion"""
    def __init__(self, image_input_dim=1536, feature_input_dim=709, num_classes=3, 
                 fusion_dim=512, dropout_rate=0.5):
        super(EnhancedEfficientNet_IFOF, self).__init__()
        
        # Use EfficientNet-B3 for better feature extraction
        try:
            self.backbone = efficientnet_b3(weights=EfficientNet_B3_Weights.IMAGENET1K_V1)
        except AttributeError:
            # Fallback for older torchvision versions
            self.backbone = efficientnet_b3(pretrained=True)
        self.backbone.classifier = nn.Identity()
        
        # Enhanced fusion module
        self.fusion = EnhancedFusionAttention(
            image_dim=image_input_dim,
            feature_dim=feature_input_dim,
            fusion_dim=fusion_dim,
            num_heads=8
        )
        
        # Self-attention on fused features
        self.attention = SelfAttention(fusion_dim)
        
        # Enhanced classifier
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout_rate / 2),
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout_rate / 4),
            
            nn.Linear(128, num_classes)
        )
        
        self._initialize_weights()
        
    def _initialize_weights(self):
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
    def forward(self, images, features):
        # Handle sequence of images
        batch_size, seq_len, c, h, w = images.size()
        images = images.view(batch_size * seq_len, c, h, w)
        image_features = self.backbone(images)
        image_features = image_features.view(batch_size, seq_len, -1)
        image_features = image_features[:, -1, :]  # Use last frame
        
        # Flatten extra features
        features = features.view(batch_size, -1)
        
        # Fuse features
        fused_features = self.fusion(image_features, features)
        fused_features = self.attention(fused_features)
        
        # Classify
        output = self.classifier(fused_features)
        return output

class LabelSmoothingLoss(nn.Module):
    """Label smoothing for better generalization"""
    def __init__(self, classes=3, smoothing=0.1):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.classes = classes
        
    def forward(self, pred, target):
        pred = pred.log_softmax(dim=-1)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.classes - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=-1))
