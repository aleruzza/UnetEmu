import torch.nn as nn
import torch

class MSEandFFT(nn.Module):
    def __init__(self, wmse=1, wfft=1, c_kea=1, gamma_kea=0.1):
        super(MSEandFFT, self).__init__()
        self.wmse = wmse
        self.wfft = wfft
        self.gamma_kea = gamma_kea
        self.c_kea = c_kea

    def forward(self, inputs, targets):
        mse = (inputs-targets)**2
        mse = mse.mean()
        in_fft = torch.fft.rfft(inputs, axis=2)
        tar_fft = torch.abs(torch.fft.rfft(targets, axis=2))
        fftaw = self.c_kea*torch.exp(-self.gamma_kea*torch.range(0, tar_fft.shape[2]-1))
        fft = (((in_fft-tar_fft)**2).mean(axis=-1)*fftaw).mean()
        return mse*self.wfft + fft*self.wfft
