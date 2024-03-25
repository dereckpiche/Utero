
import torch.nn as nn
import torch



class recurrent_transfomer(nn.Module):
    def __init__(
            self,
            embed_size,
            nb_layers, 
            vocab_size,
            hidden_length,
            hidden_size,
            symbolic_length,
            symbolic_size
    ):
        super(recurrent_transfomer, self).__init__()
        decoder_layer = nn.TransformerDecoderLayer(d_model=embed_size, nhead=4)
        self.transformer = nn.TransformerDecoder(decoder_layer, nb_layers)


        self.hidden_context = torch.rand(hidden_length, hidden_size) #
        self.hidden_counts = torch.zeros(hidden_length)
        self.symbolic_context = torch.zeros(symbolic_length, symbolic_size)

        self.symbolic_out = nn.Linear(symbolic_size*symbolic_length, vocab_size)
        self.hidden_out = nn.Linear(symbolic_size*symbolic_length, hidden_size)

    def autoregress(self, sequence, item, dim):
        sequence = sequence[:, 1:, :]
        sequence = torch.cat(sequence, item, dim=dim)

    def forward(self, symbol):
        self.autoregress(self.symbolic_context, symbol, 1)
        out = self.transformer(self.symbolic_context)
        probs, hidden = (self.symbolic_out(out), self.hidden_out(out))
        self.autoregress(self.hidden_context, hidden, 1) # add tought to sequence
        self.autoregress(self.hidden_counts, torch.tensor([0]), 0)

        # make gradient cuts past the gradient horizons
        for pos in range(self.hidden_counts.shape[0]):
            if self.hidden_counts[pos] > self.gradient_horizon:
                # prevent further comp graph connexions to this token
                hidden = self.hidden_context[:, pos, :]
                hidden = hidden.clone().detach()
                self.self.hidden_context[:, pos, :]


        return probs


