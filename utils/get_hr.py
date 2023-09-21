import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.fft import fft
from scipy.signal import welch, butter, lfilter
from scipy.sparse import spdiags

def MAE(pred,label):
    return np.mean(np.abs(pred-label))

def RMSE(pred,label):
    return np.sqrt(np.mean((pred-label)**2))

def MAPE(pred,label):
    return np.mean(np.abs((pred-label)/label)) *100

def corr(pred,label):
    p = np.corrcoef(pred,label)
    p[np.isnan(p)] = 0
    return p

def detrend(signal, Lambda=25):
    def _detrend(signal, Lambda=Lambda):
        signal_length = signal.shape[0]
        H = np.identity(signal_length)
        ones = np.ones(signal_length)
        minus_twos = -2 * np.ones(signal_length)
        diags_data = np.array([ones, minus_twos, ones])
        diags_index = np.array([0, 1, 2])
        D = spdiags(diags_data, diags_index, (signal_length - 2), signal_length).toarray()
        filtered_signal = np.dot((H - np.linalg.inv(H + (Lambda ** 2) * np.dot(D.T, D))), signal)
        return filtered_signal
    rst = np.zeros_like(signal)
    for i in np.arange(0, signal.shape[0], 900):
        if i<=signal.shape[0]-900:
            rst[i:i+900] = _detrend(signal[i:i+900])
        else:
            rst[i:] = _detrend(signal[-900:])[-(rst.shape[0]-i):]
    return rst

def get_hr(y, sr=30, min=30, max=180):
    p, q = welch(y, sr, nfft=1e5/sr, nperseg=np.min((len(y)-1, 256)))
    return p[(p>min/60)&(p<max/60)][np.argmax(q[(p>min/60)&(p<max/60)])]*60

def get_hrv(y, sr=30):
    # Please use videos longer than 5 minutes, such as the OBF dataset.
    p, q = welch(y, sr, nfft=1e6/sr, nperseg=np.min((len(y)-1, 512)))
    RF = p[(p>.04)&(p<.4)][np.argmax(q[(p>.04)&(p<.4)])]
    TP = q[(p>.04)&(p<.4)].sum()
    HF = q[(p>.15)&(p<.4)].sum()/TP
    LF = q[(p>.04)&(p<.15)].sum()/TP
    return RF, LF, HF, LF/HF

def band_pass_filter(pulse, fs, low, high):
    low = low / (0.5 * fs)
    high = high / (0.5 * fs)
    [b, a] = signal.butter(4, [low, high], btype='bandpass')
    return signal.filtfilt(b, a, np.double(pulse))

def calculate_hr(x, fs, low, high):
    N = 1024
    x_fft = np.abs(fft(x, n=N))[:int(N/2)]
    fre_x = np.arange(int(N / 2)) * fs / N
    st = np.argmin(np.abs(fre_x - low))
    ed = np.argmin(np.abs(fre_x - high))
    idx = np.argmax(x_fft[st:ed]) + st
    hr = fre_x[idx] * 60
    return hr

def get_hr_from_signal(x, fs=30, use_bpf=True, low=0.75, high=2.5):
    if use_bpf:
        x = band_pass_filter(x, fs, low, high)

    hr = calculate_hr(x, fs, low, high)
    return hr

def get_hr_from_signal_welch(x, fs=30, use_bpf=True, low=0.75, high=2.5):
    x = detrend(x, 100)
    if use_bpf:
        x = band_pass_filter(x, fs, low, high)

    hr = get_hr(x, fs, low*60, high*60)
    return hr

class EvaluateHR:
    def __init__(self, mode='DIFF', fs=30, metric=['MAE', 'RMSE', 'MAPE', 'Pearson']):
        self.m_pred_hr = []
        self.m_real_hr = []
        self.m_pred_signal = []
        self.m_real_signal = []
        self.m_signal_loss = {
            'Pearson': [],
            'snr': []
        }
        self.m_metric = metric
        self.m_mode = mode
        self.m_fs = fs

    def clear(self):
        self.m_pred_hr.clear()
        self.m_real_hr.clear()

    def add_data(self, pred_signal, real_signal):
        if self.m_mode in ['DIFF', 'ONLY_DIFF']:
            pred_signal = np.cumsum(pred_signal)
            real_signal = np.cumsum(real_signal)
        # pred_hr = get_hr_from_signal_welch(pred_signal)
        # real_hr = get_hr_from_signal_welch(real_signal)
        self.m_pred_signal.append(pred_signal)
        self.m_real_signal.append(real_signal)
        pred_hr = get_hr_from_signal(pred_signal, self.m_fs)
        real_hr = get_hr_from_signal(real_signal, self.m_fs)

        self.m_pred_hr.append(pred_hr)
        self.m_real_hr.append(real_hr)

    def get_result(self):
        return self.m_pred_signal, self.m_real_signal, self.m_pred_hr, self.m_real_hr

    def get_loss(self):
        loss = {}

        if "MAE" in self.m_metric:
            loss["MAE"] = MAE(np.array(self.m_pred_hr), np.array(self.m_real_hr))
        if "RMSE" in self.m_metric:
            loss["RMSE"] = RMSE(np.array(self.m_pred_hr), np.array(self.m_real_hr))
        if "MAPE" in self.m_metric:
            loss["MAPE"] = MAPE(np.array(self.m_pred_hr), np.array(self.m_real_hr))
        if "Pearson" in self.m_metric:
            loss["Pearson"] = corr(np.array(self.m_pred_hr), np.array(self.m_real_hr))[0, 1]
        return loss
