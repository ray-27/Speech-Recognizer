import torch
import torch.nn as nn
import torch.nn.functional as F

class SpeechTransformer(nn.Module):
    def __init__(self, num_features, num_classes, dim_model=512, num_heads=8, num_encoder_layers=6, dropout=0.1):
        super(SpeechTransformer, self).__init__()
        self.embedding = nn.Linear(num_features, dim_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=dim_model, nhead=num_heads, dropout=dropout)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        self.output_layer = nn.Linear(dim_model, num_classes)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        # Embedding layer that expands features to model dimensions
        src = self.embedding(src) * math.sqrt(self.embedding.out_features)
        src = src.permute(1, 0, 2)  # Transformer expects seq_len, batch, feature

        # Transformer Encoder
        output = self.encoder(src, mask=src_mask, src_key_padding_mask=src_key_padding_mask)
        output = output.permute(1, 0, 2)  # Convert back to batch, seq_len, feature for linear layer

        # Output layer
        output = self.output_layer(output)
        return F.log_softmax(output, dim=-1)

# Helper function to create masks and padding masks
def create_masks(input_seq, device='cpu'):
    # Create padding mask for sequences padded with zero
    pad_mask = (input_seq == 0).transpose(0, 1).to(device)
    return pad_mask

# Example usage:
num_features = 128  # Number of input features, e.g., size of the spectrogram slice
num_classes = 29  # For example, English alphabets plus space, blank, and apostrophe
model = SpeechTransformer(num_features=num_features, num_classes=num_classes)

# Assuming input is batch_size x seq_len x feature_size
input_seq = torch.rand(10, 100, num_features)  # Example input
pad_mask = create_masks(input_seq)

output = model(input_seq, src_key_padding_mask=pad_mask)
print(output.shape)  # Should be [batch_size, seq_len, num_classes]
