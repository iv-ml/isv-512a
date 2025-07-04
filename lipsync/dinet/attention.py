import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadSpatialAttention(nn.Module):
    def __init__(self, channels, num_heads=4, reduction_ratio=8):
        super(MultiHeadSpatialAttention, self).__init__()

        self.num_heads = num_heads
        self.channels = channels
        self.reduction_ratio = reduction_ratio

        # Ensure channels are divisible by num_heads
        assert channels % num_heads == 0, "channels must be divisible by num_heads"
        assert (channels // reduction_ratio) % num_heads == 0, "reduced channels must be divisible by num_heads"

        # Combined projection for Q, K, V for all heads
        # For each head: (C/r)/h + (C/r)/h + C/h channels
        head_dim = channels // num_heads
        reduced_head_dim = (channels // reduction_ratio) // num_heads
        total_qkv_dim = (reduced_head_dim * 2 + head_dim) * num_heads

        self.qkv_conv = nn.Conv2d(channels, total_qkv_dim, kernel_size=1)

        # Output projection to combine heads
        self.output_conv = nn.Conv2d(channels, channels, kernel_size=1)

        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        """
        Args:
            x: input tensor of shape (batch_size, channels, height, width)
        Returns:
            out: multi-head self-attention value + input feature
        """
        batch_size, channels, height, width = x.size()

        # Project and split Q, K, V for all heads
        qkv = self.qkv_conv(x)

        # Split channels for Q, K, V and heads
        head_dim = channels // self.num_heads
        reduced_head_dim = (channels // self.reduction_ratio) // self.num_heads

        # Calculate total dimensions for each component
        q_dim = reduced_head_dim * self.num_heads
        k_dim = reduced_head_dim * self.num_heads
        v_dim = head_dim * self.num_heads

        # Split into Q, K, V
        query, key, value = torch.split(qkv, [q_dim, k_dim, v_dim], dim=1)

        # Reshape for multi-head attention
        # Shape: (batch, num_heads, head_dim, height * width)
        query = query.view(batch_size, self.num_heads, reduced_head_dim, height * width)
        key = key.view(batch_size, self.num_heads, reduced_head_dim, height * width)
        value = value.view(batch_size, self.num_heads, head_dim, height * width)

        # Transpose for attention computation
        # Shape: (batch, num_heads, height * width, head_dim)
        query = query.transpose(-2, -1)

        # Calculate attention scores for each head
        # Shape: (batch, num_heads, height * width, height * width)
        attention = torch.matmul(query, key)
        attention = attention / (reduced_head_dim**0.5)  # Scale dot products
        attention = F.softmax(attention, dim=-1)

        # Apply attention to values
        # Shape: (batch, num_heads, height * width, head_dim)
        out = torch.matmul(attention, value.transpose(-2, -1))

        # Reshape and combine heads
        out = out.transpose(-2, -1)  # (batch, num_heads, head_dim, height * width)
        out = out.reshape(batch_size, channels, height, width)

        # Final output projection
        out = self.output_conv(out)

        # Add residual connection
        out = self.gamma * out + x

        return out


# Example usage:
if __name__ == "__main__":
    # Create a sample input tensor
    batch_size, channels, height, width = 2, 256, 64, 64
    x = torch.randn(batch_size, channels, height, width).cuda()

    # Initialize the multi-head attention layer
    attention_layer = MultiHeadSpatialAttention(channels=channels, num_heads=4).cuda()

    # Apply attention
    output = attention_layer(x)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    # number of parameters
    print(f"Number of parameters: {sum(p.numel() for p in attention_layer.parameters())}")
    # mean input and output
    print(f"Mean input: {x.mean()}")
    print(f"Mean output: {output.mean()}")
