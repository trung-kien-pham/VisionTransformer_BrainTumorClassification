import torch
import torch.nn as nn
from model.ResNet50 import ResNet50V2
from model.ViT import TransformerBlock

class R50ViT(nn.Module):
    def __init__(
        self,
        num_classes=4,
        img_size=224,
        downsample_ratio=16,
        embed_dim=768,
        num_heads=12,
        mlp_dim=3072,
        transformer_units=12,
        dropout=0.1
    ):
        super().__init__()

        assert downsample_ratio in [16, 32], "downsample_ratio must be 16 or 32"

        self.backbone = ResNet50V2(output_stride=downsample_ratio)

        feature_channels = self.backbone.out_channels
        feature_size = img_size // downsample_ratio
        num_tokens = feature_size * feature_size

        self.patch_projection = nn.Linear(feature_channels, embed_dim)

        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))

        self.pos_embed = nn.Parameter(
            torch.randn(1, num_tokens + 1, embed_dim)
        )

        self.dropout = nn.Dropout(dropout)

        self.transformer_layers = nn.Sequential(
            *[
                TransformerBlock(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    mlp_dim=mlp_dim,
                    dropout=dropout
                )
                for _ in range(transformer_units)
            ]
        )

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, num_classes)
        )

    def forward(self, x):
        x = self.backbone(x)

        B, C, H, W = x.shape

        x = x.flatten(2)
        x = x.transpose(1, 2)

        x = self.patch_projection(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        x = x + self.pos_embed
        x = self.dropout(x)

        x = self.transformer_layers(x)

        cls_output = x[:, 0]
        x = self.mlp_head(cls_output)

        return x
    

if __name__ == "__main__":
    model = R50ViT(
        num_classes=4,
        img_size=224,
        downsample_ratio=16
    )

    x = torch.randn(2, 3, 224, 224)
    out = model(x)

    print(out.shape) # torch.Size([2, num_classes])
