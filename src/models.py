import torch
import torch.nn as nn
from STConvLSTM import STConvLSTMCell


class Generator(nn.Module):
    """
    Generator model with Spatio-Temporal Convolutional LSTMs.
    """

    def __init__(self, cfg, device):
        super(Generator, self).__init__()

        self.input_size = cfg["input_size"]
        self.hidden_dim = cfg["hidden_dim"]
        self.input_dim = cfg["input_dim"]
        self.kernel_size = tuple(cfg["kernel_size"])

        self.height, self.width = self.input_size
        self.device = device

        self.STConvLSTM_Cell_1 = STConvLSTMCell(
            input_size=self.input_size,
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            kernel_size=self.kernel_size,
            bias=True,
        )

        self.STConvLSTM_Cell_2 = STConvLSTMCell(
            input_size=self.input_size,
            input_dim=self.hidden_dim,
            hidden_dim=self.hidden_dim,
            kernel_size=self.kernel_size,
            bias=True,
        )

        self.STConvLSTM_Cell_3 = STConvLSTMCell(
            input_size=self.input_size,
            input_dim=self.hidden_dim,
            hidden_dim=self.hidden_dim,
            kernel_size=self.kernel_size,
            bias=True,
        )

        self.STConvLSTM_Cell_4 = STConvLSTMCell(
            input_size=self.input_size,
            input_dim=self.hidden_dim,
            hidden_dim=self.hidden_dim,
            kernel_size=self.kernel_size,
            bias=True,
        )

        self.head = nn.Conv2d(in_channels=self.hidden_dim, out_channels=self.input_dim, kernel_size=(1, 1))

    def forward(self, input_sequence, future=10):
        batch_size = input_sequence.size(0)

        hidden_initializer = [torch.zeros(batch_size, self.hidden_dim, self.height, self.width).to(self.device)] * 3

        h_t1, c_t1, m_t1 = hidden_initializer.copy()
        h_t2, c_t2, _ = hidden_initializer.copy()
        h_t3, c_t3, _ = hidden_initializer.copy()
        h_t4, c_t4, _ = hidden_initializer.copy()

        outputs = []
        seq_len = input_sequence.size(1)

        for time in range(seq_len):
            if time:
                m_t1 = m_t4

            h_t1, c_t1, m_t1 = self.STConvLSTM_Cell_1(
                input_tensor=input_sequence[:, time, :, :, :], cur_state=[h_t1, c_t1, m_t1]
            )
            h_t2, c_t2, m_t2 = self.STConvLSTM_Cell_2(input_tensor=h_t1, cur_state=[h_t2, c_t2, m_t1])
            h_t3, c_t3, m_t3 = self.STConvLSTM_Cell_3(input_tensor=h_t2, cur_state=[h_t3, c_t3, m_t2])
            h_t4, c_t4, m_t4 = self.STConvLSTM_Cell_4(input_tensor=h_t3, cur_state=[h_t4, c_t4, m_t3])

            output = self.head(h_t4)
            output = torch.sigmoid(output)
            outputs += [output]

        for t in range(future):
            m_t1 = m_t4

            h_t1, c_t1, m_t1 = self.STConvLSTM_Cell_1(input_tensor=outputs[-1], cur_state=[h_t1, c_t1, m_t1])
            h_t2, c_t2, m_t2 = self.STConvLSTM_Cell_2(input_tensor=h_t1, cur_state=[h_t2, c_t2, m_t1])
            h_t3, c_t3, m_t3 = self.STConvLSTM_Cell_3(input_tensor=h_t2, cur_state=[h_t3, c_t3, m_t2])
            h_t4, c_t4, m_t4 = self.STConvLSTM_Cell_4(input_tensor=h_t3, cur_state=[h_t4, c_t4, m_t3])

            output = self.head(h_t4)
            output = torch.sigmoid(output)
            outputs += [output]

        outputs = torch.stack(outputs, 1)

        return outputs


class Discriminator(nn.Module):
    """
    Discriminator model.
    """

    def __init__(self, cfg):
        super(Discriminator, self).__init__()

        self.input_size = cfg["input_size"]
        self.hidden_dim = cfg["hidden_dim"]
        self.height, self.width = self.input_size

        self.linear_1 = nn.Linear(self.height * self.width, self.hidden_dim * 4)
        self.linear_2 = nn.Linear(self.hidden_dim * 4, self.hidden_dim * 2)
        self.linear_3 = nn.Linear(self.hidden_dim * 2, self.hidden_dim)
        self.linear_4 = nn.Linear(self.hidden_dim, 1)
        self.dropout = nn.Dropout(0.3)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x.view(-1, self.height * self.width)
        x = self.relu(self.linear_1(x))
        x = self.dropout(x)
        x = self.relu(self.linear_2(x))
        x = self.dropout(x)
        x = self.relu(self.linear_3(x))
        x = self.dropout(x)
        out = self.sigmoid(self.linear_4(x))

        return out
