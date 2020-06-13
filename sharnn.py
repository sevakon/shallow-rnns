from typing import List

import torch
import torch.nn as nn


class ShallowRNN(nn.Module):
    """
    Shallow RNN:
        first layer splits the input sequence and runs several independent RNNs.
        The second layer consumes the output of the first layer using a second
        RNN, thus capturing long dependencies.
    """

    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 cell_type: str,
                 hidden_dims: List[int],
                 dropouts: List[float]):
        """
        :param input_dim: feature dimension of input sequence
        :param output_dim: feature dimension of output sequence
        :param cell_type: one of LSTM or GRU
        :param hidden_dims: list of size two of hidden feature dimensions
        :param dropouts: list of size two specifying DropOut probabilities
         after the lower and upper ShaRNN layers
        """

        super(ShallowRNN, self).__init__()

        supported_cells = ['LSTM', 'GRU']
        assert cell_type in supported_cells, \
            'Only %r are supported' % supported_cells
        cls_mapping = dict(LSTM=nn.LSTM, GRU=nn.GRU)

        self.hidden_dims = hidden_dims

        rnn_type = cls_mapping[cell_type]
        self.first_layer = rnn_type(
            input_size=input_dim, hidden_size=hidden_dims[0])
        self.second_layer = rnn_type(
            input_size=hidden_dims[0], hidden_size=hidden_dims[1])

        self.first_dropout = nn.Dropout(p=dropouts[0])
        self.second_dropout = nn.Dropout(p=dropouts[1])

        self.fc = nn.Linear(hidden_dims[1], output_dim)
        # Default initialization of fc layer is Kaiming Uniform
        # Try Normal Distribition N(0, 1)?

    def forward(self, x: torch.Tensor, k: int):
        """
        :param x: Tensor of shape [seq length, batch size, input dimension]
        :param k: int specifying brick size/ stride in sliding window
        :return:
        """

        bricks = self.split_by_bricks(x, k)
        num_bricks, brick_size, batch_size, input_dim = bricks.shape

        bricks = bricks.permute(1, 0, 2, 3).reshape(k, -1, input_dim)
        # bricks shape: [brick size, num bricks * batch size, input dim]

        first, _ = self.first_layer(bricks)
        first = self.first_dropout(first)
        first = torch.squeeze(first[-1]).view(num_bricks, batch_size, -1)
        # first_out shape: [num bricks, batch size, hidden dim[0]]

        second, _ = self.second_layer(first)
        second = self.second_dropout(second)
        second = torch.squeeze(second[-1])
        # second shape: [batch size, hidden dim[1]]

        out = self.fc(second)
        # out shape: [batch size, output dim]

        return out

    @staticmethod
    def split_by_bricks(sequence: torch.Tensor, brick_size: int):
        """
        Splits an incoming sequence into bricks
        :param sequence: Tensor of shape [seq length, batch size, input dim]
        :param brick_size: int specifying brick size
        :return split_sequence: Tensor of shape
         [num bricks, brick size, batch size, feature dim]
        """

        sequence_len, batch_size, feature_dim = sequence.shape
        num_bricks = sequence_len // brick_size
        total_len = brick_size * num_bricks

        splits = torch.split(sequence[:total_len], num_bricks, dim=0)
        split_sequence = torch.stack(splits, dim=1)

        return split_sequence


if __name__ == '__main__':
    """ Simple Test """
    rnn = ShallowRNN(128, 128, 'LSTM', [512, 512], [.0, .0])
    inp = torch.randn(120, 16, 128)
    k = 12
    print(inp.shape)
    output = ShallowRNN.split_by_bricks(inp, k)
    print(output.shape)
    output = rnn(inp, k)
    print(output.shape)
