class Encoder(nn.Module):
2
3  def __init__(self, seq_len, n_features, embedding_dim=64):
4    super(Encoder, self).__init__()
5
6    self.seq_len, self.n_features = seq_len, n_features
7    self.embedding_dim, self.hidden_dim = embedding_dim, 2 * embedding_dim
8
9    self.rnn1 = nn.LSTM(
10      input_size=n_features,
11      hidden_size=self.hidden_dim,
12      num_layers=1,
13      batch_first=True
14    )
15
16    self.rnn2 = nn.LSTM(
17      input_size=self.hidden_dim,
18      hidden_size=embedding_dim,
19      num_layers=1,
20      batch_first=True
21    )
22
23  def forward(self, x):
24    x = x.reshape((1, self.seq_len, self.n_features))
25
26    x, (_, _) = self.rnn1(x)
27    x, (hidden_n, _) = self.rnn2(x)
28
29    return hidden_n.reshape((self.n_features, self.embedding_dim))

class Decoder(nn.Module):
2
3  def __init__(self, seq_len, input_dim=64, n_features=1):
4    super(Decoder, self).__init__()
5
6    self.seq_len, self.input_dim = seq_len, input_dim
7    self.hidden_dim, self.n_features = 2 * input_dim, n_features
8
9    self.rnn1 = nn.LSTM(
10      input_size=input_dim,
11      hidden_size=input_dim,
12      num_layers=1,
13      batch_first=True
14    )
15
16    self.rnn2 = nn.LSTM(
17      input_size=input_dim,
18      hidden_size=self.hidden_dim,
19      num_layers=1,
20      batch_first=True
21    )
22
23    self.output_layer = nn.Linear(self.hidden_dim, n_features)
24
25  def forward(self, x):
26    x = x.repeat(self.seq_len, self.n_features)
27    x = x.reshape((self.n_features, self.seq_len, self.input_dim))
28
29    x, (hidden_n, cell_n) = self.rnn1(x)
30    x, (hidden_n, cell_n) = self.rnn2(x)
31    x = x.reshape((self.seq_len, self.hidden_dim))
32
33    return self.output_layer(x)

class RecurrentAutoencoder(nn.Module):
2
3  def __init__(self, seq_len, n_features, embedding_dim=64):
4    super(RecurrentAutoencoder, self).__init__()
5
6    self.encoder = Encoder(seq_len, n_features, embedding_dim).to(device)
7    self.decoder = Decoder(seq_len, embedding_dim, n_features).to(device)
8
9  def forward(self, x):
10    x = self.encoder(x)
11    x = self.decoder(x)
12
13    return x

model = RecurrentAutoencoder(seq_len, n_features, 128)