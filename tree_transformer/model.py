import torch
import torch.nn as nn

class SimpleTransformerModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2):
        super(SimpleTransformerModel, self).__init__()
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=input_dim, nhead=1, dim_feedforward=hidden_dim),
            num_layers=num_layers
        )
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        # x shape: (seq_length, batch_size, input_dim)
        # print(f"x.shape: {x.shape}")
        x = self.transformer(x)
        # Take the output of the last node in the sequence
        # print(f"x.shape after transformer: {x.shape}")
        output = x[-1]
        # print(f"output.shape: {output.shape}")
        output = self.fc(output)
        # print(f"output.shape after fc: {output.shape}")
        return output


