import torch
import torch.nn as nn


class STConvLSTMCell(nn.Module):
    """
    Spatio-Temporal Convolutional LSTM Cell Implementation.
    """

    def __init__(self, input_size, input_dim, hidden_dim, kernel_size, bias, forget_bias=1.0, layer_norm=True):
        super(STConvLSTMCell, self).__init__()

        self.height, self.width = input_size
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias
        self.forget_bias = forget_bias
        self.layer_norm = layer_norm

        self.conv_wx = nn.Conv2d(
            in_channels=self.input_dim,
            out_channels=7 * self.hidden_dim,
            kernel_size=self.kernel_size,
            padding=self.padding,
            bias=self.bias,
        )

        self.conv_wht_1 = nn.Conv2d(
            in_channels=self.hidden_dim,
            out_channels=4 * self.hidden_dim,
            kernel_size=self.kernel_size,
            padding=self.padding,
            bias=self.bias,
        )

        self.conv_wml_1 = nn.Conv2d(
            in_channels=self.hidden_dim,
            out_channels=3 * self.hidden_dim,
            kernel_size=self.kernel_size,
            padding=self.padding,
            bias=self.bias,
        )

        self.conv_wml = nn.Conv2d(
            in_channels=self.hidden_dim,
            out_channels=self.hidden_dim,
            kernel_size=self.kernel_size,
            padding=self.padding,
            bias=self.bias,
        )

        self.conv_wcl = nn.Conv2d(
            in_channels=self.hidden_dim,
            out_channels=self.hidden_dim,
            kernel_size=self.kernel_size,
            padding=self.padding,
            bias=self.bias,
        )

        self.conv_h = nn.Conv2d(
            in_channels=self.hidden_dim + self.hidden_dim,
            out_channels=self.hidden_dim,
            kernel_size=(1, 1),
            padding=0,
            bias=self.bias,
        )

        nn.init.orthogonal_(self.conv_wx.weight)
        nn.init.orthogonal_(self.conv_wht_1.weight)
        nn.init.orthogonal_(self.conv_wml_1.weight)
        nn.init.orthogonal_(self.conv_wml.weight)
        nn.init.orthogonal_(self.conv_wcl.weight)
        nn.init.orthogonal_(self.conv_h.weight)

        if self.layer_norm:
            self.conv_wx_norm = nn.BatchNorm2d(7 * self.hidden_dim)
            self.conv_wht_1_norm = nn.BatchNorm2d(4 * self.hidden_dim)
            self.conv_wml_1_norm = nn.BatchNorm2d(3 * self.hidden_dim)
            self.conv_wml_norm = nn.BatchNorm2d(self.hidden_dim)
            self.conv_wcl_norm = nn.BatchNorm2d(self.hidden_dim)
            self.conv_h_norm = nn.BatchNorm2d(self.hidden_dim)

        self.forget_bias_h = torch.nn.Parameter(torch.tensor(self.forget_bias))
        self.forget_bias_m = torch.nn.Parameter(torch.tensor(self.forget_bias))

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur, m_cur = cur_state

        conved_wx = self.conv_wx(input_tensor)
        conved_wht_1 = self.conv_wht_1(h_cur)
        conved_wml_1 = self.conv_wml_1(m_cur)

        if self.layer_norm:
            conved_wx = self.conv_wx_norm(conved_wx)
            conved_wht_1 = self.conv_wht_1_norm(conved_wht_1)
            conved_wml_1 = self.conv_wml_1_norm(conved_wml_1)

        wxg, wxi, wxf, wxg_, wxi_, wxf_, wxo = torch.split(conved_wx, self.hidden_dim, dim=1)
        whg, whi, whf, who = torch.split(conved_wht_1, self.hidden_dim, dim=1)
        wmg, wmi, wmf = torch.split(conved_wml_1, self.hidden_dim, dim=1)

        g_t = torch.tanh(wxg + whg)
        i_t = torch.sigmoid(wxi + whi)
        f_t = torch.sigmoid(wxf + whf + self.forget_bias_h)
        c_next = f_t * c_cur + i_t * g_t

        g_t_ = torch.tanh(wxg_ + wmg)
        i_t_ = torch.sigmoid(wxi_ + wmi)
        f_t_ = torch.sigmoid(wxf_ + wmf + self.forget_bias_m)
        m_next = f_t_ * m_cur + i_t_ * g_t_

        wco = self.conv_wcl(c_next)
        wmo = self.conv_wml(m_next)

        if self.layer_norm:
            wco = self.conv_wcl_norm(wco)
            wmo = self.conv_wml_norm(wmo)

        o_t = torch.sigmoid(wxo + who + wco + wmo)

        combined_cmn = torch.cat([c_next, m_next], dim=1)
        h_next = o_t * torch.tanh(self.conv_h(combined_cmn))

        return h_next, c_next, m_next
