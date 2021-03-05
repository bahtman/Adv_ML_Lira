import torch.nn as nn
class Encoder(nn.Module):
    def __init__(self, seq_len, n_features, embedding_dim = 64):
        super(Encoder, self).__init__()
        self.embedding_dim, self.hidden_dim = embedding_dim, 2 * embedding_dim
        
        self.rnn1 = nn.LSTM(
        input_size=n_features,
        hidden_size=self.hidden_dim,
        num_layers=1,
        batch_first=True
        )
        self.rnn2 = nn.LSTM(
        input_size=self.hidden_dim,
        hidden_size=embedding_dim,
        num_layers=1,
        batch_first=True
        )
    def forward(self, x):
        rnn_features, = self.rnn1(x)
        rnn_features, h = self.rnn2(x)
        return rnn_features, h

class Decoder(nn.Module):
    def __init__(self, seq_len, n_features, input_dim = 64):
        super(Decoder, self).__init__()
        self.hidden_dim, self.n_features = 2 * input_dim, n_features

        self.rnn1 = nn.LSTM(
            input_size=input_dim,
            hidden_size=self.hidden_dim,
            num_layers=1,
            batch_first=True
        ) 
        self.rnn1 = nn.LSTM(
            input_size=input_dim,
            hidden_size=self.hidden_dim,
            num_layers=1,
            batch_first=True
        ) 

        self.out = nn.Linear(
            in_features = self.hidden_dim,
            out_features = n_features,
            bias = False
        )
    def forward(self, x):
        x, h = self.rnn1(x)
        x = self.out(x)
        return x

class RecurrentAutoencoder(nn.Module):
    def __init__(self, seq_len, n_features, embedding_dim=64):
        super(RecurrentAutoencoder, self).__init__()

        self.encoder = Encoder(seq_len, n_features, embedding_dim)#.to(device)
        self.decoder = Decoder(seq_len, n_features, embedding_dim)#.to(device)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

