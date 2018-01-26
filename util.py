import numpy as np
import wave
import sys

C_1 = 8.1757989156


def load_wav(filepath, t_start=0, t_end=sys.maxsize):
    """Load a wave file, which must be 22050Hz and 16bit and must be either
    mono or stereo.
    Inputs:
        filepath: audio file
        t_start, t_end:  (optional) subrange of file to load (in seconds)
    Returns:
        a numpy floating-point array with a range of [-1, 1]
    """

    wf = wave.open(filepath)
    num_channels, sampwidth, fs, end, comptype, compname = wf.getparams()

    # for now, we will only accept 16 bit files at 22k
    assert (sampwidth == 2)
    assert (fs == 22050)

    # start frame, end frame, and duration in frames
    f_start = int(t_start * fs)
    f_end = min(int(t_end * fs), end)
    frames = f_end - f_start

    wf.setpos(f_start)
    raw_bytes = wf.readframes(frames)

    # convert raw data to numpy array, assuming int16 arrangement
    samples = np.fromstring(raw_bytes, dtype=np.int16)

    # convert from integer type to floating point, and scale to [-1, 1]
    samples = samples.astype(np.float)
    samples *= (1 / 32768.0)

    if num_channels == 1:
        return samples

    elif num_channels == 2:
        return 0.5 * (samples[0::2] + samples[1::2])

    else:
        raise Exception("Can't deal with this file.")


def save_wav(channels, fs, filepath) :
    """Interleave channels and write out wave file as 16bit audio.
    Inputs:
        channels: a tuple or list of np.arrays. Or can be a single np.array in which case this will be a mono file.
                  format of np.array is floating [-1, 1]
        fs: sampling rate
        filepath: output filepath
    """

    if type(channels) == tuple or type(channels) == list:
        num_channels = len(channels)
    else:
        num_channels = 1
        channels = [channels]

    length = min ([len(c) for c in channels])
    data = np.empty(length*num_channels, np.float)

    # interleave channels:
    for n in range(num_channels):
        data[n::num_channels] = channels[n][:length]

    data *= 32768.0
    data = data.astype(np.int16)
    data = data.tostring()

    wf = wave.open(filepath, 'w')
    wf.setnchannels(num_channels)
    wf.setsampwidth(2)
    wf.setframerate(fs)
    wf.writeframes(data)


def pitch_to_freq(p):
    semitone_difference = 2**(1.0/12)
    return C_1*(semitone_difference**p)


def freq_to_pitch(f):
    semitone_difference = 2**(1.0/12)
    return np.log(f/C_1)/np.log(semitone_difference)


def pitch_to_spn(s):
    notes = ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"]
    note_without_octave = notes[s%12]
    scale = 24
    octave = 0
    while scale <= s:
        scale += 12
        octave += 1
    return note_without_octave + str(octave)

def freq_to_k(freq, fs, len_audio):
    k = freq * len_audio / fs
    return int(np.round(k))

def STFT(x, fft_len, hop_size):
    """
    STFT with centered Hann window and zeropadding
    """
    L = fft_len
    H = hop_size
    x = np.concatenate((np.zeros(int(L/2)), x))
    N = len(x)
    num_frames = int(1 + ((N - L)/H))
    result = np.empty((num_frames, int(1+(L/2))), dtype=complex)
    for m in range(num_frames):
        data = x[m*H:m*H + L]
        data = data*np.hanning(len(data))
        result[m, :] = np.fft.rfft(data)
    return result.T


def istft(X, hop_size, centered):
    # deal with centering
    # transpose stft
    X = X.T
    eps = 0.001
    N = (X.shape[1] - 1) * 2
    H = hop_size
    hops = X.shape[0]
    L = (hops - 1) * H + N
    win_sum = np.zeros(L)
    stft_sum = np.zeros(L)
    win = np.hanning(N)
    for n in range(hops):
        win_shift = np.zeros(L)
        tmp = np.fft.irfft(X[n])
        tmp_shift = np.zeros(L)
        if centered:
            half_N = int(N/2)
            if n * H - N / 2 >= 0:
                win_shift[n * H - half_N: n * H + half_N] += win
                tmp_shift[n * H - half_N: n * H + half_N] += tmp
            else:
                win_shift[: n * H + half_N] += win[half_N - n * H:]
                tmp_shift[: n * H + half_N] += tmp[half_N - n * H:]
        else:
            win_shift[n * H: n * H + N] += win
            tmp_shift[n * H: n * H + N] += tmp

        win_sum += win_shift
        stft_sum += tmp_shift
    win_sum[win_sum < eps] = eps
    x_r = stft_sum / win_sum
    return x_r


def find_peaks(x, thresh=0.2):
    x0 = x[:-2]  # x
    x1 = x[1:-1]  # x shifted by 1
    x2 = x[2:]  # x shifted by 2

    peak_bools = np.logical_and(x0 < x1, x1 > x2)  # where x1 is higher than surroundings
    values = x1[peak_bools]  # list of all peak values

    # find a threshold relative to the highest peak
    th = np.max(values) * thresh

    # filter out values that are below th
    peak_bools = np.logical_and(peak_bools, x1 > th)

    peaks = np.nonzero(peak_bools)[0] + 1  # get indexes of peaks, shift by 1
    return peaks