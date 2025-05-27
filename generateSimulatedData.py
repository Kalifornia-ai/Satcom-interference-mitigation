import numpy as np
import random
import csv
import os
import argparse
from scipy.signal import firwin, lfilter, butter, filtfilt
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft, fftfreq


############################
# Utility Functions
############################

def generate_prbs(sequence_length, polynomial_mask=0b11000001, seed=0b1111111):
    """
    Generates a PRBS (Pseudo-Random Binary Sequence) using an LFSR.
    """
    if seed is None:
        seed = np.random.randint(1, 128)
    prbs = []
    lfsr = seed
    for _ in range(sequence_length):
        prbs.append(lfsr & 1)
        feedback = (lfsr & 1) ^ ((lfsr & polynomial_mask) >> 1)
        lfsr = (lfsr >> 1) | (feedback << (lfsr.bit_length() - 1))
    return np.array(prbs, dtype=np.uint8)


def add_awgn(signal, snr_db, is_complex=False):
    nonzeros = signal[signal != 0]
    if len(nonzeros) == 0:
        sig_power = 1e-8
    else:
        sig_power = np.mean(np.abs(nonzeros) ** 2)

    snr_linear = 10 ** (snr_db / 10)
    noise_power = sig_power / snr_linear

    if is_complex:
        # for complex signals
        noise = np.sqrt(noise_power / 2) * (
                np.random.randn(*signal.shape) + 1j * np.random.randn(*signal.shape)
        )
    else:
        # If we want a real
        noise = np.sqrt(noise_power) * np.random.randn(*signal.shape)

    return signal + noise


def bpsk_mod(bits):
    """
    BPSK mapping: 0->-1, 1->+1. Normalized to unit average power.
    """
    symbols = 2 * bits.astype(np.int8) - 1  # 0->-1, 1->+1
    symbols = symbols.astype(np.complex64)
    avg_pwr = np.mean(np.abs(symbols) ** 2)
    symbols /= np.sqrt(avg_pwr)
    return symbols


def psk_mod(bits, M=4):
    k = int(np.log2(M))
    num_bits = len(bits)

    if num_bits % k != 0:
        bits = np.append(bits, np.zeros(k - (num_bits % k), dtype=np.int8))

    bit_groups = bits.reshape(-1, k)

    gray_indices = np.zeros(len(bit_groups), dtype=int)
    for i, b in enumerate(bit_groups):
        binary = int("".join(map(str, b)), 2)
        gray_indices[i] = binary ^ (binary >> 1)  # Gray code conversion

    # Calculate angles with rotation for proper constellation alignment
    angles = 2 * np.pi * gray_indices / M + np.pi / M

    return np.exp(1j * angles)


def qam64_mod(bits):
    """
    64-QAM => 6 bits per symbol, 8x8 constellation.
    """
    num_symbols = len(bits) // 6
    bits = bits[:num_symbols * 6]
    bits_reshaped = bits.reshape((num_symbols, 6))
    sym_index = np.array([
        int("".join(map(str, brow)), 2) for brow in bits_reshaped
    ])

    i_idx = sym_index // 8
    q_idx = sym_index % 8
    levels = np.array([-7, -5, -3, -1, 1, 3, 5, 7])
    real_part = levels[i_idx]
    imag_part = levels[q_idx]
    symbols = real_part + 1j * imag_part
    avg_power = np.mean(np.abs(symbols) ** 2)
    symbols /= np.sqrt(avg_power)
    return symbols.astype(np.complex64)


def oversample(symbols, sps=8):
    """
    Repeat each symbol 'sps' times => simulate oversampling.
    """
    return np.repeat(symbols, sps)


def generate_complex_exponential(length, fs, freq, amp=1.0, phase=0.0, phase_noise_std=0.005):
    """
    Returns s(t) = amp * exp(j(2π freq * t + phase)), length=length, t in [0, length/fs).
    """
    t = np.arange(length) / fs
    phase_noise = np.cumsum(np.random.normal(0, phase_noise_std, length))
    mix = np.exp(1j * (2 * np.pi * freq * t + phase_noise))
    mixed_up = mix * np.exp(1j * phase)
    return amp * mixed_up


def set_power(signal, desired_power_dB=0):
    """
    Scale 'signal' so its power becomes 'desired_power_dB' (dBm).
    Using reference Z=50 => P= mean(|V|^2)/50, then convert to dBm.
    """
    nonzeros = signal[signal != 0]
    if len(nonzeros) == 0:
        return signal
    sig_power_lin = np.mean(np.abs(nonzeros) ** 2) / 50
    sig_power_dBm = 10 * np.log10(sig_power_lin / 1e-3)
    scaling_dB = desired_power_dB - sig_power_dBm
    scaling_lin = 10 ** (scaling_dB / 20)
    return signal * scaling_lin


def scale_signals_for_sdr(signal1, signal2, sdr, Z=50):
    """
    Make signal2 be 'sdr' dB below (or above if negative) signal1 in power.
    """

    def calculate_power(sig):
        if len(sig) == 0:
            return -np.inf
        power = np.mean(np.abs(sig) ** 2) / Z
        return 10 * np.log10(power / 1e-3) if power > 0 else -np.inf

    p1 = calculate_power(signal1)
    p2 = calculate_power(signal2)

    # Handle edge cases
    if p1 == -np.inf or p2 == -np.inf:
        return signal2

    target_p2 = p1 - sdr
    scaling = 10 ** ((target_p2 - p2) / 20)
    return signal2 * scaling


def complex_upconvert(baseband_wave, freq_offset, fs):
    """
    Multiply baseband_wave by exp(j 2π freq_offset t).
    freq_offset in Hz. => final is complex passband near freq_offset, ignoring real passband steps.
    """
    length = len(baseband_wave)
    t = np.arange(length) / fs
    mix = np.exp(1j * (2 * np.pi * freq_offset * t))
    return baseband_wave * mix


def plot_qpsk_constellation(signal, sps, title):
    # Downsample to symbol rate
    symbols = signal[sps // 2::sps]  # Sample at symbol centers
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.scatter(np.real(symbols), np.imag(symbols), alpha=0.5)
    plt.title(title)
    plt.grid(True)
    plt.axhline(0, color='k', linestyle='--')
    plt.axvline(0, color='k', linestyle='--')
    plt.xlabel('In-Phase')
    plt.ylabel('Quadrature')
    plt.axis('equal')
    plt.show()


def apply_rrc_filter(signal, sps=8, roll_off=0.35, taps=101):
    t_idx = np.arange(-taps // 2, taps // 2 + 1) / sps

    h = np.zeros_like(t_idx)

    for i, t in enumerate(t_idx):
        if t == 0:
            h[i] = (1 + roll_off * (4 / np.pi - 1))
        elif abs(t) == 1 / (4 * roll_off):
            h[i] = (roll_off / np.sqrt(2)) * ((1 + 2 / np.pi) * np.sin(np.pi / (4 * roll_off)) +
                                              (1 - 2 / np.pi) * np.cos(np.pi / (4 * roll_off)))
        else:
            h[i] = (np.sin(np.pi * t * (1 - roll_off)) + 4 * roll_off * t * np.cos(np.pi * t * (1 + roll_off))) / \
                   (np.pi * t * (1 - (4 * roll_off * t) ** 2))

    # h -= np.mean(h)  # Remove DC component
    h /= np.sqrt(np.sum(h ** 2))

    filtered_signal = np.convolve(signal, h, mode='full')

    delay = (len(h) - 1) // 2
    return filtered_signal[delay:delay + len(signal)]

def calculate_power_dBm(signal, Z=50):
    """Calculate average power in dBm"""
    if len(signal) == 0:
        return -np.inf
    power = np.mean(np.abs(signal) ** 2) / Z
    return 10 * np.log10(power / 1e-3) if power > 0 else -np.inf



def downconvert_to_baseband(passband_signal, carrier_freq, fs, cutoff_freq=10000):
    """
    Downconvert passband signal to baseband
    """
    t = np.arange(len(passband_signal)) / fs
    mixer = np.exp(-1j * 2 * np.pi * carrier_freq * t)
    baseband_signal = passband_signal * mixer

    baseband_signal_no_dc = baseband_signal - np.mean(baseband_signal)
    return baseband_signal_no_dc


def fix_length(arr, desired_len):
    """
    Truncate or zero-pad arr to 'desired_len'.
    """
    L = len(arr)
    if L > desired_len:
        return arr[:desired_len]
    elif L < desired_len:
        pad_count = desired_len - L
        return np.pad(arr, (0, pad_count), 'constant')
    else:
        return arr


def compute_fft(signal):
    # Compute FFT along each channel
    fft_real = np.fft.fft(np.real(signal))
    fft_imag = np.fft.fft(np.imag(signal))
    fft_mag = np.abs(fft_real) + 1j * np.abs(fft_imag)  # Combine magnitudes
    return fft_mag


def plot_fft(signal, fs=1.0, title="FFT of Signal", Z=50):
    """
    Plot the FFT of a signal.
    """
    if signal.ndim == 2 and signal.shape[1] == 2:
        signal = signal[:, 0] + 1j * signal[:, 1]

    N = len(signal)
    freqs = np.fft.fftfreq(N, d=1 / fs)  # length N, from 0..Fs*(N-1)/N
    freqs_shifted = np.fft.fftshift(freqs)
    S = np.fft.fft(signal)
    S_shifted = np.fft.fftshift(S)  # center at 0 Hz

    fft_mag = np.abs(S_shifted) / N
    power_watts = (fft_mag ** 2) / Z
    mag_dbm = 10 * np.log10(power_watts / 1e-3 + 1e-12)

    plt.plot(freqs_shifted, mag_dbm, label="Mix Real", color='blue')
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.show()




def shift_frequency(signal, shift_freq, sampling_freq):
    t = np.arange(len(signal)) / sampling_freq
    shift = np.exp(-1j * 2 * np.pi * shift_freq * t)
    shifted_signal = signal * shift

    return shifted_signal


def apply_freq_window(signal, sample_rate, center_freq, bandwidth=20e6):
    N = len(signal)
    freqs = fftfreq(N, d=1 / sample_rate)  # Frequency bins

    spectrum = fft(signal)
    mask = (np.abs(freqs - center_freq) <= bandwidth / 2)
    filtered_spectrum = spectrum * mask
    filtered_signal = ifft(filtered_spectrum).real  # Keep real part

    return filtered_signal


def apply_phase_correction(signal, carrier_freq, sampling_freq, SPS=8):
    symbol_rate = sampling_freq / SPS  # Calculate symbol rate
    residual_freq = carrier_freq - np.round(carrier_freq / symbol_rate) * symbol_rate
    t = np.arange(len(signal)) / sampling_freq
    phase_correction = np.exp(-1j * 2 * np.pi * residual_freq * t)
    return signal * phase_correction


############################
# Main Data Generation
############################

#Used for shifting the signals correctly
def freq_shift(x, fs, f0):
    n = np.arange(len(x), dtype=np.float32)
    return x * np.exp(-1j * 2 * np.pi * f0 * n / fs)

def lowpass(x: np.ndarray, fs: float, cutoff_hz: float, num_taps: int = 65) -> np.ndarray:
    taps = firwin(num_taps, cutoff=cutoff_hz, fs=fs)
    real = lfilter(taps, [1.0], x.real)
    imag = lfilter(taps, [1.0], x.imag)
    return real + 1j * imag

def generate_sampled_mixtures(
        num_mixtures=10,
        desired_len=1024,
        snr_db=30.0,
        sampling_freq=40e9,
        sig_freq=5.0002e9,
        sdr=10.0,
        signal_power_level=20.0,
        sigType="sin",
        bandPassBW=20E6,
        SPS=8,
        qpsk_center_freq = 5e9
):
    """
    Generate a dataset of fully-complex IQ signals:
      src1 => 5 GHz complex exponential (non-zero imag).
      src2 => random modded signal (BPSK/PSK/QAM64) upconverted in *complex* domain => also non-zero imag.
    Then combine at specified SDR, add AWGN, store in final list.

    Return: list of dicts: [ 'src1', 'src2', 'mixture', + metadata ]
    """

    # mod_schemes = ["BPSK","PSK","QAM64"]
    mod_schemes = ["PSK"]
    psk_orders = [4]
    src1_seeds = [69, 85, 119, 73, 127]
    dataset = []

    for i in range(num_mixtures):
        # -- Generate src1: 5 GHz complex exponential
        amp1 = random.uniform(0.5, 1.5)
        ph1 = random.uniform(0, 2 * np.pi)
        if sigType == "prbs":
            bits1 = generate_prbs(desired_len, seed=src1_seeds[np.random.randint(0, 5)])
            scheme1 = random.choice(mod_schemes)
            if scheme1 == "BPSK":
                baseband = bpsk_mod(bits1)
            elif scheme1 == "PSK":
                M2 = random.choice(psk_orders)
                baseband = psk_mod(bits1, M2)
            else:  # QAM64
                baseband = qam64_mod(bits1)
            baseband = oversample(baseband, sps=SPS)
            baseband = fix_length(baseband, desired_len)
            bits1L = baseband
            baseband = baseband * amp1 * np.exp(1j * np.pi / 4)
            src1_iq = complex_upconvert(baseband, sig_freq, sampling_freq)
            src1_iq = fix_length(src1_iq, desired_len)

            src1_iq_corrected = apply_phase_correction(src1_iq, sig_freq, sampling_freq, SPS)
            saved_src1_iq = src1_iq_corrected.copy()

        elif sigType == "sin":
            scheme1 = sigType
            saved_src1_iq = generate_complex_exponential(
                length=desired_len, fs=sampling_freq,
                freq=sig_freq, amp=amp1, phase=ph1
            )
            bits1L = np.zeros(desired_len)
            bits2L = np.zeros(desired_len)

        # -- Generate src2: random mod
        bits2 = generate_prbs(desired_len)
        scheme2 = random.choice(mod_schemes)
        if scheme2 == "BPSK":
            baseband = bpsk_mod(bits2)
        elif scheme2 == "PSK":
            M2 = random.choice(psk_orders)
            baseband = psk_mod(bits2, M2)
        else:  # QAM64
            baseband = qam64_mod(bits2)

        # random amplitude & phase offsets
        amp2 = random.uniform(0.5, 1.5)
        ph2 = random.uniform(0, 2 * np.pi)
        baseband = oversample(baseband, sps=SPS)
        bits2L = fix_length(baseband, desired_len)
        baseband = apply_rrc_filter(baseband, sps=8, roll_off=0.15)

        baseband = baseband * amp2 * np.exp(1j * ph2)
        freq_offset = random.uniform(-10e7, 10e7)
        src2_iq = complex_upconvert(baseband, qpsk_center_freq + freq_offset, sampling_freq)
        src2_iq = fix_length(src2_iq, desired_len)

        src2_iq_corrected = apply_phase_correction(src2_iq, qpsk_center_freq + freq_offset, sampling_freq, SPS)
        saved_src2_iq = src2_iq_corrected.copy()


        t = np.arange(desired_len) / sampling_freq
        src1_real = (saved_src1_iq.real * np.cos(2 * np.pi * sig_freq * t) -
                     saved_src1_iq.imag * np.sin(2 * np.pi * sig_freq * t))

        src1_real = set_power(src1_real, desired_power_dB=signal_power_level)
        src2_real = (saved_src2_iq.real * np.cos(2 * np.pi * (qpsk_center_freq + freq_offset) * t) - (
                    saved_src2_iq.imag * np.sin(2 * np.pi * (qpsk_center_freq + freq_offset) * t)))

        # -- Mix signals at desired SDR => src2 relative to src1
        src2_real_scaled = set_power(src2_real, sdr)

        scale1 = np.sqrt(np.mean(np.abs(src1_real) ** 2) / np.mean(np.abs(saved_src1_iq) ** 2))
        saved_src1_iq_scaled = saved_src1_iq * scale1

        scale2 = np.sqrt(np.mean(np.abs(src2_real_scaled) ** 2) / np.mean(np.abs(saved_src2_iq) ** 2))
        saved_src2_iq_scaled = saved_src2_iq * scale2
        # -- Combine
        mixture = src1_real + src2_real_scaled
        mixture_noisy_real = add_awgn(mixture, snr_db)

        baseband_mix = freq_shift(mixture_noisy_real, fs=15e9, f0=5.0002e9)

        baseband_mix = lowpass(baseband_mix, 1e9, 20e6)


        src1_power = calculate_power_dBm(saved_src1_iq_scaled)
        src2_power = calculate_power_dBm(saved_src2_iq_scaled)
        mix_power = calculate_power_dBm(baseband_mix)

        saved_src2_iq_scaled = freq_shift(saved_src2_iq_scaled, fs=15e9, f0=5e9)
        saved_src1_iq_scaled = freq_shift(saved_src1_iq_scaled, fs=15e9, f0=5.0002e9)

        sample = {
            'src1': saved_src1_iq_scaled,
            'src2': saved_src2_iq_scaled,
            'src1_prbs': bits1L,
            'src2_prbs': bits2L,
            'mixture': baseband_mix,
            'scheme': str(M2) + scheme1 if scheme1 == "PSK" else scheme1,
            'amp1': amp1, 'amp2': amp2,
            'phase1': ph1, 'phase2': ph2,
            'freq_offset': freq_offset,
            'snr_db': snr_db,
            'sampling_freq': sampling_freq,
            'SDR': sdr,
            'power_level_dB': signal_power_level,
            'src1_power': src1_power,
            'src2_power': src2_power,
            'mix_power': mix_power
        }
        dataset.append(sample)

    return dataset


def save_mixtures_to_csv(dataset, folder="complex_iq_csv(sine)2"):
    os.makedirs(folder, exist_ok=True)
    for i, sample in enumerate(dataset):
        csv_filename = os.path.join(folder, f"mixture_{i}.csv")
        mixture = sample['mixture']
        src1 = sample['src1']
        src1_prbs = sample['src1_prbs']
        src2 = sample['src2']
        src2_prbs = sample['src2_prbs']
        length = len(mixture)

        with open(csv_filename, "w", newline='') as f:
            writer = csv.writer(f)
            # Metadata line
            meta_row = [
                f"Index={i}",
                f"Scheme={sample['scheme']}",
                f"SRC1_dBm={sample['src1_power']}",
                f"SRC2_dBm={sample['src2_power']}",
                f"mix_dBm={sample['mix_power']}",
                f"Amp1={sample['amp1']:.3f}",
                f"Amp2={sample['amp2']:.3f}",
                f"Phase1={sample['phase1']:.3f}",
                f"Phase2={sample['phase2']:.3f}",
                f"FreqOffset={sample['freq_offset']:.2e}",
                f"SNR_dB={sample['snr_db']:.2f}",
                f"SamplingFreq={sample['sampling_freq']:.2e}",
                f"SDR={sample['SDR']:.2f}",
                f"PowerLevel={sample['power_level_dB']:.2f}",
                f"Length={length}"
            ]
            writer.writerow(meta_row)
            # Header
            writer.writerow(
                ["SampleIndex", "mixRe", "mixIm", "src1Re", "src1Im", "src2Re", "src2Im", "PRBS_src1", "PRBS_src2"])

            # Data lines
            for idx_samp in range(length):
                mix_r = mixture[idx_samp].real
                mix_i = mixture[idx_samp].imag
                s1_r = src1[idx_samp].real
                s1_i = src1[idx_samp].imag
                s2_r = src2[idx_samp].real
                s2_i = src2[idx_samp].imag
                PRBS_src1 = src1_prbs[idx_samp]
                PRBS_src2 = src2_prbs[idx_samp]
                writer.writerow([
                    idx_samp,
                    f"{mix_r:.10f}", f"{mix_i:.10f}",
                    f"{s1_r:.10f}", f"{s1_i:.10f}",
                    f"{s2_r:.10f}", f"{s2_i:.10f}",
                    f"{PRBS_src1:.10f}", f"{PRBS_src2:.10f}"
                ])
    print(f"Created {len(dataset)} CSV files in '{folder}'.")


############################
# CLI Entry Point
############################

if __name__ == "__main__":
    sine_levels = [-10, -20, -25, -30, -35, -40, -45, -50]  # Sine (Power_src)
    qpsk_levels = [-10, -20, -30, -40, -50]  # QPSK (Power_int)
    target_pairs = [(q, s) for q in qpsk_levels for s in sine_levels]

    parser = argparse.ArgumentParser(description="Generate fully-complex IQ data (non-zero Imag).")
    parser.add_argument("--num_mixtures", type=int, default=200,
                        help="Number of mixture files.")
    parser.add_argument("--length", type=int, default=1000,
                        help="Number of samples per signal.")
    parser.add_argument("--sig_freq", type=float, default=5.0002e9, help="Frequency of the sine wave for src1 (Hz).")
    parser.add_argument("--snr_db", type=float, default=20.0,
                        help="AWGN SNR in dB.")
    parser.add_argument("--sampling_freq", type=float, default=15e9,
                        help="Sampling frequency (Hz). Must be >= 2*f if using 5GHz.")
    parser.add_argument("--sdr", type=float, default =-25,
                        help="Desired ratio (dB) src1 to src2.")
    parser.add_argument("--power_level", type=float, default = -35,
                        help="Power level in dBm after pathloss for src1.")
    parser.add_argument("--output_folder", type=str, default="LastFolderPleaseWork", #Changed to be generic
                        help="Folder to store CSVs.")
    parser.add_argument("--sigType", type=str, default="sin",
                        help="determines wheter src1 is a sine or a prbs. args: sin or prbs")
    parser.add_argument("--SPS", type=int, default=16, help="amount of samples per symbol")
    parser.add_argument("--BW", type=int, default=2E6, help="bandwidth of bandpass filter")
    parser.add_argument("--center_frequency", type=float, default=5e9, help="Center frequency for QPSK signal (Hz).")
    args = parser.parse_args()


    for qpsk_power, sine_power in target_pairs:
        folder_name = f"sine50002_{sine_power}dBmQPSK{int(args.center_frequency / 1e6)}{qpsk_power}dbm"
        full_output_path = os.path.join(args.output_folder, folder_name)

        print(f"Generating for SINE = {sine_power} dBm, QPSK = {qpsk_power} dBm -> {folder_name}")

        dataset = generate_sampled_mixtures(
            num_mixtures=args.num_mixtures,
            desired_len=args.length,
            snr_db=args.snr_db,
            sampling_freq=args.sampling_freq,
            sdr= qpsk_power,  # SDR = Power_src - Power_int
            signal_power_level=sine_power,  # Power of sine
            sigType=args.sigType,
            sig_freq=args.sig_freq,
            qpsk_center_freq=args.center_frequency,
            SPS=args.SPS,
            bandPassBW=args.BW
        )

        save_mixtures_to_csv(dataset, folder=full_output_path)

    print("All done!")
