from sklearn.decomposition import NMF
from util import *

def make_pitch_spectral_model(fs, fft_len, p):
    alpha = 0.9
    result = np.zeros(int(fft_len/2 + 1))
    f0 = pitch_to_freq(p - 0.5)
    f1 = pitch_to_freq(p + 0.5)
    h = 1
    while True:
        k0 = int(h * f0 * fft_len / fs)
        k1 = int(h * f1 * fft_len / fs)
        if k1 > len(result) - 1:
            break
        result[k0:min(k1, len(result)-1)] = alpha ** (h-1)
        h += 1
    return result


def make_W_from_pitches(fs, fft_len, pitches):
    result = []
    for pitch in pitches:
        result.append(make_pitch_spectral_model(fs, fft_len, pitch))
    return np.array(result).T

def nmf(V, Wi, Hi):
    model = NMF(n_components=14, init='custom')
    print(V.shape)
    print(Wi.shape)
    print(Hi.shape)
    W = model.fit_transform(V, W=Wi, H=Hi)
    H = model.components_
    return (W, H)


def nmf2(V, Wi, Hi):
    num_iter = 1000
    W = Wi
    H = Hi
    eps = 0.001
    for _ in range(num_iter):
        tmp_H = np.divide(np.multiply(H, np.dot(np.transpose(W), V)), np.dot(np.dot(np.transpose(W), W), H) + eps)
        tmp_W = np.divide(np.multiply(W, np.dot(V, np.transpose(H))), np.dot(np.dot(W, H), np.transpose(H)) + eps)
        H = tmp_H
        W = tmp_W
    return (W, H)


def separate(filepath):
    snd = load_wav(filepath)
    fs = 22050

    # get initial stft
    win_len = 2048
    hop_size = 1024
    stft = STFT(snd, win_len, hop_size)
    V = np.abs(stft)
    K, N = V.shape

    # get pitches (which I took from the sheet music-- we'd need a good way of generating this
    # from the audio itself (just a list of all pitches it contains)
    mag_pitch = np.mean(V, axis=1)
    peaks = find_peaks(mag_pitch, thresh=0.01)
    freqs = peaks * float(fs) / win_len
    pitches = [freq_to_pitch(i) for i in freqs]
    #pitches = np.array([53, 57, 60, 63, 58, 62, 55, 65, 69, 72, 70, 79, 77, 76])
    R = len(pitches)

    # get initial values of phi
    Wi = make_W_from_pitches(fs, win_len, pitches)
    Hi = np.ones((R, N))

    # get W, H
    W, H = nmf2(V, Wi, Hi)
    error = np.linalg.norm(V - np.dot(W, H))
    print("Error: ", error)

    # get modified filters swashing values
    H_left = H.copy()
    H_left[7:, :] = 0
    H_right = H.copy()
    H_right[0:7, :] = 0
    eps = 0.001

    # get masks to multiply to STFT
    smask_l = np.divide(np.dot(W, H_left), np.dot(W, H) + eps)
    smask_r = np.divide(np.dot(W, H_right), np.dot(W, H) + eps)

    # get modified STFT
    stft_l = smask_l * stft
    stft_r = smask_r * stft

    # reconstruct and save sounds
    snd_l = istft(stft_l, hop_size, centered=True)
    snd_r = istft(stft_r, hop_size, centered=True)
    save_wav(snd_l, fs, "lefthand_test.wav")
    save_wav(snd_r, fs, "righthand_test.wav")


def halve_stft(filepath):
    snd = load_wav(filepath)
    fs = 22050
    win_len = 2048
    hop_size = 1024
    stft = STFT(snd, win_len, hop_size)
    stft_l = stft.copy()
    stft_r = stft.copy()
    freq_mid_c = pitch_to_freq(3)
    print(freq_mid_c)
    bin = freq_to_k(freq_mid_c, fs, len(snd))
    bin = 48
    print(bin)
    print(len(stft))
    stft_l[0:bin, :] = 0
    stft_r[bin:, :] = 0
    snd_l = istft(stft_l, hop_size, centered=True)
    snd_r = istft(stft_r, hop_size, centered=True)
    save_wav(snd_l, fs, "testright.wav")
    save_wav(snd_r, fs, "testleft.wav")

separate("mozart_sonata_f.wav")