# Adapted for DCSN from https://github.com/Hank0626/DDN/blob/72b8d9c595ca81e70500919689f8715ed133e6d2/models/FEDformer.py
# Modified: paper configuration and local imports. See LICENSES/README.txt.
import torch
from torch import nn
from torch.nn import functional as F
from ._embedding import DataEmbedding_wo_pos
from ._autocorrelation import AutoCorrelationLayer
from ._fourier import FourierBlock, FourierCrossAttention
from ._former_layers import Encoder, Decoder, EncoderLayer, DecoderLayer, my_Layernorm, series_decomp


class Model(nn.Module):
    """
    FEDformer performs the attention mechanism on frequency domain and achieved O(N) complexity
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.version = configs.version
        self.mode_select = configs.mode_select
        self.modes = configs.modes
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention
        kernel_size = configs.moving_avg
        self.decomp = series_decomp(kernel_size)
        self.enc_embedding = DataEmbedding_wo_pos(configs.enc_in, configs.d_model, configs.embed, configs.freq, configs.dropout)
        self.dec_embedding = DataEmbedding_wo_pos(configs.dec_in, configs.d_model, configs.embed, configs.freq, configs.dropout)
        encoder_self_att = FourierBlock(in_channels=configs.d_model, out_channels=configs.d_model, seq_len=self.seq_len, modes=configs.modes, mode_select_method=configs.mode_select)
        decoder_self_att = FourierBlock(in_channels=configs.d_model, out_channels=configs.d_model, seq_len=self.seq_len // 2 + self.pred_len, modes=configs.modes, mode_select_method=configs.mode_select)
        decoder_cross_att = FourierCrossAttention(in_channels=configs.d_model, out_channels=configs.d_model, seq_len_q=self.seq_len // 2 + self.pred_len, seq_len_kv=self.seq_len, modes=configs.modes, mode_select_method=configs.mode_select)
        self.encoder = Encoder([EncoderLayer(AutoCorrelationLayer(encoder_self_att, configs.d_model, configs.n_heads), configs.d_model, configs.d_ff, moving_avg=configs.moving_avg, dropout=configs.dropout, activation=configs.activation) for l in range(configs.e_layers)], norm_layer=my_Layernorm(configs.d_model))
        self.decoder = Decoder([DecoderLayer(AutoCorrelationLayer(decoder_self_att, configs.d_model, configs.n_heads), AutoCorrelationLayer(decoder_cross_att, configs.d_model, configs.n_heads), configs.d_model, configs.c_out, configs.d_ff, moving_avg=configs.moving_avg, dropout=configs.dropout, activation=configs.activation) for l in range(configs.d_layers)], norm_layer=my_Layernorm(configs.d_model), projection=nn.Linear(configs.d_model, configs.c_out, bias=True))

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None, seq_mean=None, seq_std=None):
        mean = torch.mean(x_enc, dim=1).unsqueeze(1).repeat(1, self.pred_len, 1)
        seasonal_init, trend_init = self.decomp(x_enc)
        trend_init = torch.cat([trend_init[:, -self.label_len:, :], mean], dim=1)
        seasonal_init = F.pad(seasonal_init[:, -self.label_len:, :], (0, 0, 0, self.pred_len))
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask, seq_mean=seq_mean, seq_std=seq_std)
        dec_out = self.dec_embedding(seasonal_init, x_mark_dec)
        seasonal_part, trend_part = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask, trend=trend_init, seq_mean=seq_mean, seq_std=seq_std)
        dec_out = trend_part + seasonal_part
        if self.output_attention:
            return (dec_out[:, -self.pred_len:, :], attns)
        else:
            return dec_out[:, -self.pred_len:, :]
