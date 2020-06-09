"""ANN models."""

from torch import nn

class TactMlpGru(nn.Module):
    def __init__(self, output_size):
        super(TactMlpGru, self).__init__()
        self.input_size = input_size  # Tactile has 156 taxels
        self.hidden_dim = args.hidden_size

        self.gru = nn.GRU(self.input_size, self.hidden_dim, 1)
        self.fc = nn.Linear(self.hidden_dim, output_size)

    def forward(self, in_tac):
        in_tac.squeeze().permute(2, 0, 1)
        gru_out, _ = self.gru(in_tact)
        y_pred = self.fc(gru_out[-1, :, :])
        return y_pred

class VisMlpGru(nn.Module):
    def __init__(self, output_size):
        super(VisMlpGru, self).__init__()
        self.input_size = 1000
        self.hidden_dim = args.hidden_size
        self.batch_size = args.batch_size

        # Define the LSTM layer
        self.gru = nn.GRU(self.input_size, self.hidden_dim, 1)
        self.fc_vis = nn.Linear(63 * 50 * 2, self.input_size)
        self.fc = nn.Linear(self.hidden_dim, output_size)

    def forward(self, in_vis):
        in_vis = in_vis.reshape([in_vis.shape[0], in_vis.shape[-1], 50 * 63 * 2])
        embeddings = self.fc_vis(in_vis).permute(1, 0, 2)
        out, hidden = self.gru(embeddings)
        out = out.permute(1, 0, 2)

        y_pred = self.fc(out[:, -1, :])

        return y_pred

class MultiMlpGru(nn.Module):
    def __init__(self, output_size):
        super(MultiMLP_GRU, self).__init__()
        self.vis_input_size = 1000
        self.tac_input_size = 156
        self.input_size = self.vis_input_size + self.tac_input_size
        self.hidden_dim = args.hidden_size
        self.batch_size = args.batch_size

        self.gru = nn.GRU(self.input_size, self.hidden_dim, 1)
        self.fc = nn.Linear(self.hidden_dim, output_size)
        self.fc_vis = nn.Linear(63 * 50 * 2, self.vis_input_size)

    def forward(self, in_tact, in_vis):
        in_tac = in_tac.squeeze().permute(2, 0, 1)
        in_vis = in_vis.reshape([in_vis.shape[0], in_vis.shape[-1], 50 * 63 * 2])
        viz_embeddings = self.fc_vis(in_vis).permute(1, 0, 2)
        embeddings = torch.cat([viz_embeddings, in_tact], dim=2)
        out, hidden = self.gru(embeddings)
        out = out.permute(1, 0, 2)
        y_pred = self.fc(out[:, -1, :])

        return y_pred
