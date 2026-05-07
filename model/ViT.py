import torch
import torch.nn as nn


class PatchEmbedding(nn.Module):
    def __init__(self, num_channels: int, embed_dim: int, patch_size: int):
        super().__init__()
        self.patch_embed = nn.Conv2d(
            in_channels=num_channels,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

    def forward(self, x):
        x = self.patch_embed(x)      # [B, embed_dim, H/P, W/P]
        x = x.flatten(2)             # [B, embed_dim, num_patches]
        x = x.transpose(1, 2)        # [B, num_patches, embed_dim]
        return x

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, mlp_dim: int, dropout: float = 0.1):
        super().__init__()

        self.layer_norm_1 = nn.LayerNorm(embed_dim)

        self.self_attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        self.layer_norm_2 = nn.LayerNorm(embed_dim)

        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        # Self-attention block
        residual = x
        x_norm = self.layer_norm_1(x)
        attention_output, _ = self.self_attention(x_norm, x_norm, x_norm)
        x = residual + attention_output

        # MLP block
        residual = x
        x_norm = self.layer_norm_2(x)
        mlp_output = self.mlp(x_norm)
        x = residual + mlp_output

        return x

class VisionTransformer(nn.Module):
    def __init__(
        self,
        num_channels: int,
        embed_dim: int,
        patch_size: int,
        img_size: int,
        num_heads: int,
        mlp_dim: int,
        transformer_units: int,
        num_classes: int,
        dropout: float = 0.1
    ):
        super().__init__()

        assert img_size % patch_size == 0, "img_size must be divisible by patch_size"

        num_patches = (img_size // patch_size) ** 2

        self.patch_embedding = PatchEmbedding(
            num_channels=num_channels,
            embed_dim=embed_dim,
            patch_size=patch_size
        )

        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))

        self.pos_embed = nn.Parameter(
            torch.randn(1, num_patches + 1, embed_dim)
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
        x = self.patch_embedding(x)  # [B, num_patches, embed_dim]

        B = x.size(0)

        cls_tokens = self.cls_token.expand(B, -1, -1)

        x = torch.cat((cls_tokens, x), dim=1)  # [B, num_patches + 1, embed_dim]

        x = x + self.pos_embed
        x = self.dropout(x)

        x = self.transformer_layers(x)

        cls_output = x[:, 0]  # Get cls token

        x = self.mlp_head(cls_output)

        return x


if __name__ == "__main__":
    model = VisionTransformer(
        num_channels=3,
        embed_dim=768,
        patch_size=16,
        img_size=224,
        num_heads=12,
        mlp_dim=3072,
        transformer_units=12,
        num_classes=1000
    )

    x = torch.randn(2, 3, 224, 224)
    out = model(x)

    print(out.shape)  # Expected output: [2, 1000]