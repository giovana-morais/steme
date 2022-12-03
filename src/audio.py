import random

from scipy import signal
from scipy.interpolate import interp1d
import librosa
import numpy as np

def load_and_resample(audio_file, mono=True):
    desired_sr = 16000

    x, sr = librosa.load(audio_file, sr=desired_sr, mono=mono)

    if not mono:
        # we just want the voice channel
        x = x[1]

    return x, sr

def cqt(x, sr):
    """
    The CQT frontend is parametrized to use Q = 24 bins per octave,
    so as to achieve a resolution equal to one half-semitone per
    bin.
    We set fbase equal to the frequency of the note C1, i.e.,
    fbase ' 32.70 Hz and we compute up to Fmax = 190 CQT
    bins, i.e., to cover the range of frequency up to Nyquist.
    We use a Hann window with hop length set equal to 512 samples,
    i.e., one CQT frame every 32 ms.
    """

    Q = 24
    fbase = 32.70
    n_bins = 190
    window = "hann"
    hop_length = 512

    x_cqt = librosa.cqt(y=x, sr=sr, hop_length=hop_length, fmin=fbase,
                        n_bins=n_bins, bins_per_octave=Q,
                        window=window)
    C = np.abs(x_cqt)

    return C

def get_cqt_slices(C, F=128, slice_idx=None):
    """
    During training, we extract slices of F = 128 CQT bins, setting kmin = 0
    and kmax = 8 (i.e., between 0 and 4 semitones when Q = 24).
    """
    k_min = 0
    k_max = 8

    shift_1 = random.randint(k_min,k_max)
    shift_2 = random.randint(k_min,k_max)

    if slice_idx is None:
        slice_idx = random.randint(0, C.shape[1]-1)

    cqt_sample_1 = C[shift_1:shift_1+F, slice_idx]
    cqt_sample_2 = C[shift_2:shift_2+F, slice_idx]

    # fix shapes for training
    cqt_sample_1 = cqt_sample_1[:, np.newaxis]
    cqt_sample_2 = cqt_sample_2[:, np.newaxis]

    shift_1 = np.array([shift_1])
    shift_2 = np.array([shift_2])

    return cqt_sample_1, shift_1, cqt_sample_2, shift_2, slice_idx

def get_cqt_sample(C, F=128):
    k_min = 0
    k_max = 8

    shift = random.randint(k_min,k_max)
    slice_idx = random.randint(0, C.shape[1]-1)
    cqt_sample = C[shift:shift+F, slice_idx]

    return cqt_sample[:, np.newaxis]

def spectral_flux(x, sr, n_fft=1024, hop_length=256, gamma=100.0, avg_window=10, norm=True):
    """
    Compute the spectral flux of a signal and apply logarithmic compression.

    Parameters
    ---------
        x : np.ndarray
            audio signal
        sr : int
            sampling rate
        n_fft : int, optional
            fft window size
        hop_length : int, optional
            step between fft windows
        gamma : float, optional
            logarithmic compression factor
        avg_window : int, optional
            window size (in samples) to compute local average
        norm : bool, optional
            boolean flag to normalize or not the novelty function
    Return
    ------
        novelty : np.ndarray
            the novelty function
        sr_novelty : float
            the sampling rate of the novelty function. defined as (sampling
            rate)/hop length
    """
    X = librosa.stft(x, n_fft=n_fft, hop_length=hop_length, win_length=n_fft, window="hanning")
    sr_novelty = sr/hop_length

    Y = np.log(1 + gamma * np.abs(X))

    Y_diff = np.diff(Y)
    Y_diff[Y_diff < 0] = 0

    novelty = np.sum(Y_diff, axis=0)
    novelty = np.concatenate((novelty, np.array([0.0])))

    # subtract local avg
    if avg_window > 0:
        L = len(novelty)
        local_avg = np.zeros(L)
        for m in range(L):
            init = max(m - avg_window, 0)
            end = min(m + avg_window + 1, L)
            local_avg[m] = np.sum(novelty[init:end]) * (1/(1 + 2 * avg_window))
        novelty = novelty - local_avg
        novelty[novelty < 0] = 0.0

    if norm:
        max_value = max(novelty)
        if max_value > 0:
            novelty /= max_value

    return novelty, sr_novelty

def fourier_tempogram(novelty, sr_novelty, window_size, hop_size, theta):
    """
    Compute Fourier tempogram

    Parameters
    ----------
        novelty : np.ndarray
            novelty function
        sr_novelty : np.float
            sampling rate of the novelty function
        window_size : int
            window size in frames. 1000 corresponds to 10s in a signal sampled
            at 100 Hz
        hop_size : int
            hop size
        theta : np.ndarray
            range of BPM to cover
    """
    window = np.hanning(window_size)
    pad_size = int(window_size//2)

    L = novelty.shape[0] + 2*pad_size

    novelty_pad = np.concatenate((np.zeros(pad_size), novelty, np.zeros(pad_size)))
    t_pad = np.arange(L)

    M = np.int64(np.floor(L - window_size) / hop_size + 1)
    K = len(theta)
    X = np.zeros((K,M), dtype=np.complex_)

    for k in range(K):
        omega = (theta[k]/60)/sr_novelty

        exponential = np.exp(-2 * np.pi * 1j  * omega * t_pad)
        x_exp = novelty_pad * exponential

        for n in range(M):
            t_0 = n * hop_size
            t_1 = t_0 + window_size
            X[k, n] = np.sum(window * x_exp[t_0:t_1])

    times = np.arange(M) * hop_size / sr_novelty
    tempi = theta

    return np.abs(X), times, tempi

def tempogram(x, sr, window_size_seconds, t_type, theta):
    """
        x : np.ndarray
            signal
        sr : float64
            sampling rate
        window_size : int, optional
            size in seconds of the tempogram window. default is 5s.
        type : string, optional
            tempogram type. accepted values are "fourier", "autocorrelation",
            "hybrid"
        theta :  np.arange, optional
            tempi interval (BPM). default is (30,300,1), i.e from 30 to 300, 1
            at a time.
    """

    if not isinstance(theta, np.ndarray):
        raise ValueError(f"theta type incorrect. it should be np.ndarray, but is {type(theta)}")

    # 2. novelty function
    novelty, sr_novelty = spectral_flux(x, sr, n_fft=2048, hop_length=512)

    window_size_frames = int(window_size_seconds*sr_novelty)
    hop_size = 10

    if t_type == "fourier":
        T, t, bpm = fourier_tempogram(novelty, sr_novelty, window_size=window_size_frames, hop_size=hop_size, theta=theta)
    elif t_type == "autocorrelation":
        T, t, bpm, _, _ = autocorrelation_tempogram(novelty, sr_novelty,
                window_size=window_size_frames, hop_size=hop_size, theta=theta)
    elif t_type == "hybrid":
        ft, t, bpm = fourier_tempogram(novelty, sr_novelty,
                window_size=window_size_frames, hop_size=hop_size, theta=theta)
        at, ta, freqsa, _, _ = autocorrelation_tempogram(novelty, sr_novelty,
                window_size=window_size_frames, hop_size=hop_size, theta=theta)

        T = ft*at
    else:
        raise ValueError("tempogram_type incorrect. accepted values are \
                ['fourier', 'autocorrelation', 'hybrid']")

    return T, t, bpm

def click_track(bpm, sr=22050, duration=60):
    """
    Generates a 60 seconds click track with the desired BPM

    Parameters
    ----------
        bpm :  int
            desired tempo
        sr : int, optional
            sampling rate
        duration : int
            duration in seconds
    """

    step = 60 / bpm

    times = np.arange(0, duration, step)

    return librosa.clicks(times=times, sr=sr)

def local_autocorrelation(x, sr, N, H):
    """Compute local autocorrelation [FMP, Section 6.2.3]

    Notebook: C6/C6S2_TempogramAutocorrelation.ipynb

    Args:
        x (np.ndarray): Input signal
        sr (scalar): Sampling rate
        N (int): Window length
        H (int): Hop size

    Returns:
        A (np.ndarray): Time-lag representation
        times (np.ndarray): Time axis (seconds)
        lags (np.ndarray): Lag axis
    """
    L_left = round(N / 2)
    L_right = L_left
    x_pad = np.concatenate((np.zeros(L_left), x, np.zeros(L_right)))
    L_pad = len(x_pad)
    M = int(np.floor(L_pad - N) / H) + 1
    A = np.zeros((N, M))
    win = np.ones(N)
    for n in range(M):
        t_0 = n * H
        t_1 = t_0 + N
        x_local = win * x_pad[t_0:t_1]
        r_xx = np.correlate(x_local, x_local, mode='full')
        r_xx = r_xx[N-1:]
        A[:, n] = r_xx
    sr_A = sr / H
    times = np.arange(A.shape[1]) / sr_A
    lags = np.arange(N) / sr
    return A, times, lags

def autocorrelation_tempogram(novelty, sr_novelty, window_size, hop_size, theta):
    """
    Compute autocorrelation-based tempogram

    Parameters
    ----------
        novelty : np.ndarray
            input novelty function
        sr_novelty : float64
            sampling rate
        window_size :  int
            window length in frames
        hop_size : int
            hop size
        theta : np.ndarray
            array with BPM values we want to interpolate the autocorrelation

    Return
    ------
        tempogram : np.ndarray
            autocorrelation tempogram
        times : np.ndarray
            time axis (seconds)
        bpms : np.ndarray
            tempo axis (BPM)
        A_cut : np.ndarray
            time-lag representation A_cut (cut according to theta)
        lags_cut : np.ndarray
            Lag axis lags_cut
    """
    tempo_min = theta[0]
    tempo_max = theta[-1]
    lag_min = int(np.ceil(sr_novelty * 60 / tempo_max))
    lag_max = int(np.ceil(sr_novelty * 60 / tempo_min))

    A, times, lags = local_autocorrelation(novelty, sr_novelty, window_size, hop_size)
    # getting the min/max lag interval to use in the interpolation
    A_cut = A[lag_min:lag_max+1, :]

    # "cut" the frequencies out of the max/min
    lags_cut = lags[lag_min:lag_max+1]

    # translate to BPM
    bpms_cut = 60 / lags_cut
    bpms = theta

    # interpolate
    axis_interpolation = interp1d(bpms_cut, A_cut, kind='linear',
                         axis=0, fill_value='extrapolate')

    tempogram = axis_interpolation(bpms)
    return tempogram, times, bpms, A_cut, lags_cut
