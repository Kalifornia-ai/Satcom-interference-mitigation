#######################
#This script is supposed to simulate a drone flying in a circle around an antenna and measuring the power of a mixture dataset while reducing the power.
#The model required for this setup is the CausalLSTM that needs its path to be changed.

#####################################
import numpy as np
import random
import matplotlib.pyplot as plt
import torch
from matplotlib import cm
from scipy.interpolate import griddata

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
        sig_power = np.mean(np.abs(nonzeros)**2)

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
    symbols = 2*bits.astype(np.int8) - 1  # 0->-1, 1->+1
    symbols = symbols.astype(np.complex64)
    avg_pwr = np.mean(np.abs(symbols)**2)
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
    angles = 2 * np.pi * gray_indices / M + np.pi/M
    
    return np.exp(1j * angles)

def oversample(symbols, sps=8):
    """
    Repeat each symbol 'sps' times => simulate oversampling.
    """
    return np.repeat(symbols, sps)


def generate_complex_exponential(length, fs, freq, amp=1.0, phase=0.0,phase_noise_std=0.02):
    """
    Returns s(t) = amp * exp(j(2π freq * t + phase)), length=length, t in [0, length/fs).
    """
    t = np.arange(length) / fs
    phase_noise = np.cumsum(np.random.normal(0, phase_noise_std, length))
    mix = np.exp(1j*(2*np.pi*freq*t+ phase_noise))
    mixed_up = mix * np.exp(1j*phase)
    return amp*mixed_up

def set_power(signal, desired_power_dB=0):
    """
    Scale 'signal' so its power becomes 'desired_power_dB' (dBm).
    Using reference Z=50 => P= mean(|V|^2)/50, then convert to dBm.
    """
    nonzeros = signal[signal != 0]
    if len(nonzeros) == 0:
        return signal
    sig_power_lin = np.mean(np.abs(nonzeros)**2) / 50
    sig_power_dBm = 10*np.log10(sig_power_lin / 1e-3)
    scaling_dB = desired_power_dB - sig_power_dBm
    scaling_lin = 10**(scaling_dB/20)
    return signal * scaling_lin


def scale_signals_for_sdr(signal1, signal2, sdr, Z=50):
    """
    Make signal2 be 'sdr' dB below (or above if negative) signal1 in power.
    """
    def calculate_power(sig):
        if len(sig) == 0:
            return -np.inf
        power = np.mean(np.abs(sig)**2) / Z
        return 10*np.log10(power/1e-3) if power > 0 else -np.inf
    
    p1 = calculate_power(signal1)
    p2 = calculate_power(signal2)

     # Handle edge cases
    if p1 == -np.inf or p2 == -np.inf:
        return signal2
    
    target_p2 = p1 - sdr
    scaling = 10**((target_p2 - p2)/20)
    return signal2 * scaling


def complex_upconvert(baseband_wave, freq_offset, fs, phase_noise_std=0.000):
    """
    Multiply baseband_wave by exp(j 2π freq_offset t).
    freq_offset in Hz. => final is complex passband near freq_offset, ignoring real passband steps.
    """
    length = len(baseband_wave)
    t = np.arange(length) / fs
    phase_noise = np.cumsum(np.random.normal(0, phase_noise_std, length))
    mix = np.exp(1j*(2*np.pi*freq_offset*t+phase_noise))
    return baseband_wave * mix


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

    #h -= np.mean(h)  # Remove DC component
    h /= np.sqrt(np.sum(h ** 2))  

    filtered_signal = np.convolve(signal, h, mode='full')

    delay = (len(h) - 1) // 2
    return filtered_signal[delay:delay + len(signal)]

def calculate_power_dBm(signal, Z=50):
    """Calculate average power in dBm"""
    if len(signal) == 0:
        return -np.inf
    if signal.shape[-1] == 2:  # I/Q format
            signal = signal[...,0] + 1j*signal[...,1]
    power = np.mean(np.abs(signal)**2) / Z
    return 10*np.log10(power/1e-3) if power > 0 else -np.inf



def downconvert_to_baseband(passband_signal, carrier_freq, fs, cutoff_freq=10):
    """
    Downconvert passband signal to baseband
    """
    t = np.arange(len(passband_signal)) / fs
    mixer = np.exp(-1j*2*np.pi*carrier_freq*t)
    baseband_signal = passband_signal * mixer

    #baseband_signal = dc_block(baseband_signal,fs, cutoff_freq)
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



def apply_phase_correction(signal, carrier_freq, sampling_freq, SPS = 8):
    symbol_rate = sampling_freq / SPS  # Calculate symbol rate
    residual_freq = carrier_freq - np.round(carrier_freq / symbol_rate) * symbol_rate
    t = np.arange(len(signal)) / sampling_freq
    phase_correction = np.exp(-1j * 2 * np.pi * residual_freq * t)
    return signal * phase_correction

############################
# Main Data Generation
############################


def generate_sampled_mixtures(
    num_mixtures=1,
    desired_len=1024,
    snr_db=20.0,
    sampling_freq=40e9,
    sig_freq = 5e9,
    sdr=10.0,
    signal_power_level=20.0,
    sigType = "sin",
    SPS = 8
):

    mod_schemes = ["PSK"]
    psk_orders = [4]
    dataset = []

    for i in range(num_mixtures):
        # -- Generate src1: 5 GHz complex exponential
        amp1  = random.uniform(0.5,1.5)
        ph1   = random.uniform(0,2*np.pi)
        scheme1 = sigType
        saved_src1_iq  = generate_complex_exponential(
            length=desired_len, fs=sampling_freq,
            freq=10e6, amp=amp1, phase=ph1
        )
        # -- Generate src2: random mod
        #bits2 = generate_prbs(desired_len)
        bitstring = '1111111000000100000110000101000111100100010110011101010011111010'
        bits2 = np.array([int(b) for b in bitstring], dtype=np.uint8)
        scheme2 = random.choice(mod_schemes)
        if scheme2=="BPSK":
            baseband = bpsk_mod(bits2)
        else:
            M2 = random.choice(psk_orders)
            baseband = psk_mod(bits2, M2)

        # random amplitude & phase offsets
        amp2  = random.uniform(0.5,1.5)
        ph2   = random.uniform(0,2*np.pi)
        baseband = oversample(baseband, sps=SPS)
        baseband = apply_rrc_filter(baseband, sps=8, roll_off=0.15)
        
        baseband =baseband* amp2* np.exp(1j*ph2)
        freq_offset = random.uniform(-10e7, 10e7)
        src2_iq = complex_upconvert(baseband, freq_offset+sig_freq, sampling_freq)
        src2_iq = fix_length(src2_iq, desired_len)


        src2_iq_corrected = apply_phase_correction(src2_iq, sig_freq+freq_offset, sampling_freq, SPS)
        saved_src2_iq = src2_iq_corrected.copy()

        t = np.arange(desired_len) / sampling_freq
        src1_real = (saved_src1_iq.real * np.cos(2 * np.pi * sig_freq * t) - 
                    saved_src1_iq.imag * np.sin(2 * np.pi * sig_freq * t))

        src1_real = set_power(src1_real, desired_power_dB=signal_power_level)
        src2_real = (saved_src2_iq.real * np.cos(2*np.pi*(sig_freq+freq_offset)*t) - (saved_src2_iq.imag * np.sin(2*np.pi*(sig_freq+freq_offset)*t)))
        
        # -- Mix signals at desired SDR => src2 relative to src1
        src2_real_scaled = scale_signals_for_sdr(src1_real, src2_real, sdr)

        scale1 = np.sqrt(np.mean(np.abs(src1_real)**2)/np.mean(np.abs(saved_src1_iq)**2))
        saved_src1_iq_scaled = saved_src1_iq * scale1

        scale2 = np.sqrt(np.mean(np.abs(src2_real_scaled)**2)/np.mean(np.abs(saved_src2_iq)**2))
        saved_src2_iq_scaled = saved_src2_iq * scale2
        # -- Combine
        mixture = src1_real + src2_real_scaled
        mixture_noisy_real = add_awgn(mixture, snr_db)


        baseband_mix = downconvert_to_baseband(mixture_noisy_real, sig_freq, sampling_freq)

        mix_power = calculate_power_dBm(baseband_mix)

        sample = {
            'src1': saved_src1_iq_scaled,
            'src2': saved_src2_iq_scaled,
            'mixture': baseband_mix,
            'scheme': str(M2)+scheme1 if scheme1=="PSK" else scheme1,
            'amp1': amp1, 'amp2': amp2,
            'phase1': ph1, 'phase2': ph2,
            'freq_offset': freq_offset,
            'snr_db': snr_db,
            'sampling_freq': sampling_freq,
            'SDR': sdr,
            'power_level_dB': signal_power_level,
            'mix_power' : mix_power,
        }
        dataset.append(sample)

    return sample

def plot_estimated_radiation_pattern(
    antennaPos,
    dronePositions,
    estimatedGain,
    num_azimuth_slices=360,
    num_elevation_slices=91
):
    # 1) Vector from antenna to drone, and slant range
    dx = dronePositions[:,0] - antennaPos[0]
    dy = dronePositions[:,1] - antennaPos[1]
    dz = dronePositions[:,2] - antennaPos[2]
    r   = np.sqrt(dx*dx + dy*dy + dz*dz)

    # 2) Spherical angles: azimuth and elevation above xy-plane
    azimuth = np.arctan2(dy, dx)                   # –π … +π
    cos_arg = np.clip(dz / r, -1.0, 1.0)
    elevation = np.arcsin(cos_arg)                 # 0 … π/2 for dz≥0

    # 3) Build regular grid matching your measurement slices
    az_grid = np.linspace(-np.pi, np.pi, num_azimuth_slices, endpoint=False)
    el_grid = np.linspace(0, np.pi/2, num_elevation_slices)
    AZ, EL  = np.meshgrid(az_grid, el_grid)

    # 4) Interpolate measured gains onto that grid
    grid_gain = griddata(
        points = np.column_stack((azimuth, elevation)),
        values = estimatedGain,
        xi     = (AZ.flatten(), EL.flatten()),
        method = 'cubic',
        fill_value = np.min(estimatedGain)
    ).reshape(AZ.shape)

    # 5) Normalize gain → radius factor (0–1)
    max_gain = np.max(estimatedGain)
    min_gain = max(np.min(estimatedGain), max_gain - 40)
    grid_gain = np.clip(grid_gain, min_gain, max_gain)
    r_norm   = 10**((grid_gain - max_gain)/20)

    # 6) Back‑project to Cartesian using the same elevation convention
    scale = 10
    x = scale * r_norm * np.cos(EL) * np.cos(AZ)
    y = scale * r_norm * np.cos(EL) * np.sin(AZ)
    z = scale * r_norm * np.sin(EL)

    # 7) Plot
    fig = plt.figure(figsize=(12,10))
    ax  = fig.add_subplot(111, projection='3d')

    norm = plt.Normalize(min_gain, max_gain)
    surf = ax.plot_surface(
        x, y, z,
        facecolors=plt.cm.jet(norm(grid_gain)),
        rstride=2, cstride=2, alpha=0.8, linewidth=0
    )

    mappable = plt.cm.ScalarMappable(cmap='jet', norm=norm)
    mappable.set_array(grid_gain)
    cbar = fig.colorbar(mappable, ax=ax, shrink=0.6, pad=0.1)
    cbar.set_label('Gain (dBi)', rotation=270, labelpad=20)
    cbar.set_ticks(np.arange(int(min_gain), int(max_gain)+1, 5))

    # Ground plane and axes
    max_range = scale * 1.1
    xx, yy = np.meshgrid([-2,2], [-2, 2])
    ax.plot_surface(xx, yy, np.zeros_like(xx), color='gray', alpha=0.15)
    for vec, col in [((2,0,0),'r'), ((0,2,0),'g'), ((0,0,2),'b')]:
        ax.quiver(0,0,0,*vec, color=col, arrow_length_ratio=0.1, linewidth=2)

    # Peak gain marker
    ax.scatter([0], [0], [1.02*scale], color='gold', s=100,
               label=f'Peak Gain: {max_gain:.1f} dBi')

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title(f'Measured Radiation Pattern (0–90° Elevation, {num_elevation_slices} slices)', pad=20)
    ax.legend(loc='upper right')
    ax.view_init(elev=30, azim=45)
    plt.tight_layout()
    plt.savefig("Estimated3DRadiationPattern.PNG")
    plt.show()


def plot_dronemeasured_gain_patterns(actualPos, antennaPos, gains, max_gain=None):

    dx = actualPos[:,0] - antennaPos[0]
    dy = actualPos[:,1] - antennaPos[1]
    dz = actualPos[:,2] - antennaPos[2]
    
    r = np.linalg.norm(np.stack([dx, dy, dz], axis=1), axis=1)
    azimuth = np.arctan2(dy, dx) % (2 * np.pi)
    elevation = np.arccos(dz / r)
    
    # Convert to Cartesian for 3D plotting (normalized by distance)
    x = r * np.sin(elevation) * np.cos(azimuth)
    y = r * np.sin(elevation) * np.sin(azimuth)
    z = r * np.cos(elevation)
    
    # Normalize color scale
    norm = plt.Normalize(vmin=actual_gain_dBi - 27, vmax=actual_gain_dBi+2)
    
    # Create figure with two subplots
    plt.figure(figsize=(18, 8))
    
    # 1. 3D Scatter Plot
    ax1 = plt.subplot(121, projection='3d')
    sc1 = ax1.scatter(x, y, z, c=gains, cmap='jet', norm=norm, s=20)
    ax1.scatter(0, 0, 0, c='red', s=100, label='Antenna Position')
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_zlabel('Z (m)')
    ax1.set_title('3D Radiation Pattern (Relative to Antenna)')
    ax1.set_box_aspect([1, 1, 1])
    ax1.legend()
    
    # Add colorbar to 3D plot
    cbar1 = plt.colorbar(sc1, ax=ax1, shrink=0.6)
    cbar1.set_label('Gain (dBi)')
    
    # 2. 2D Heatmap (Azimuth vs Elevation)
    ax2 = plt.subplot(122)
    
    # Convert to degrees for plotting
    az_deg = np.degrees(azimuth)
    el_deg = np.degrees(elevation)
    
    # Create grid for heatmap
    az_grid = np.linspace(0, 360, 180)
    el_grid = np.linspace(0, 90, 90)
    az_mesh, el_mesh = np.meshgrid(az_grid, el_grid)
    
    # Interpolate gains onto regular grid
    from scipy.interpolate import griddata
    points = np.column_stack((az_deg, el_deg))
    gain_grid = griddata(points, gains, (az_mesh, el_mesh), method='linear')
    gain_grid = np.clip(gain_grid, actual_gain_dBi - 25, actual_gain_dBi+2)
    
    # Plot heatmap
    heatmap = ax2.contourf(az_mesh, el_mesh, gain_grid, levels=100, cmap='jet', norm=norm)
    ax2.set_xlabel('Azimuth Angle (degrees)')
    ax2.set_ylabel('Elevation Angle (degrees)')
    ax2.set_title('2D Radiation Pattern Heatmap')
    
    # Add colorbar to heatmap
    cbar2 = plt.colorbar(heatmap, ax=ax2)
    cbar2.set_label('Gain (dBi)')
    
    plt.tight_layout()
    plt.show()



def plottingGeneratedPattern(AZ_flat, EL_flat, pattern_dBi_flat, num_az = 360, num_el = 180):

    AZ = AZ_flat.reshape((num_el, num_az))
    EL = EL_flat.reshape((num_el, num_az))
    pattern_dBi = pattern_dBi_flat.reshape((num_el, num_az))

    r = 10**(pattern_dBi/20) * 10  # Scale factor
    x = r * np.sin(EL) * np.cos(AZ)  
    y = r * np.sin(EL) * np.sin(AZ)
    z = r * np.cos(EL)

    # Create 3D plot
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Plot settings
    vmin, vmax = -40, 0
    norm = plt.Normalize(vmin, vmax)
    colors = cm.jet(norm(np.clip(pattern_dBi, vmin, vmax)))

    # Create surface plot with full coverage
    surf = ax.plot_surface(x, y, z, facecolors=colors,
                        rstride=2, cstride=2,
                        linewidth=0, antialiased=True,
                        alpha=0.8)

    # Add reference axes
    ax.quiver(0, 0, 0, 1, 0, 0, color='r', arrow_length_ratio=0.1)  # X
    ax.quiver(0, 0, 0, -1, 0, 0, color='r', arrow_length_ratio=0.1)  # -X
    ax.quiver(0, 0, 0, 0, 1, 0, color='g', arrow_length_ratio=0.1)  # Y
    ax.quiver(0, 0, 0, 0, -1, 0, color='g', arrow_length_ratio=0.1)  # -Y
    ax.quiver(0, 0, 0, 0, 0, 1, color='b', arrow_length_ratio=0.1)  # Z

    # Add colorbar
    mappable = cm.ScalarMappable(norm=norm, cmap='jet')
    mappable.set_array(pattern_dBi)
    cbar = fig.colorbar(mappable, ax=ax, shrink=0.6)
    cbar.set_label('Gain (dBi)')

    # Configure axes
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_ylim(-2,2)
    ax.set_xlim(-2,2)
    ax.set_title('3D Antenna Pattern (Full Hemispherical Coverage)')

    # Set initial view
    ax.view_init(elev=30, azim=45)
    plt.tight_layout()
    plt.savefig("General3DRadiationPattern.PNG")
    plt.show()

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    pattern_dBi_capped = np.clip(pattern_dBi, -40, 40)

    # Create surface using azimuth and elevation directly
    surf = ax.plot_surface(np.degrees(AZ), np.degrees(EL),pattern_dBi,
                        cmap='jet',
                        rstride=5, cstride=5,
                        linewidth=0, antialiased=True)

    ax.set_xlabel('Azimuth (degrees)')
    ax.set_ylabel('Elevation (degrees)')
    ax.set_zlabel('Gain (dBi)')
    ax.set_zlim(-40, 0)
    ax.set_title('3D Antenna Pattern (Azimuth-Elevation Coordinates)')
    plt.colorbar(surf, label='Gain (dBi)')
    plt.show()

def omnidirectional_with_directional_bias(
    num_samples=360, 
    seed=123, 
    main_lobe_dir=np.pi/2, 
    main_lobe_strength=0.8,
    main_lobe_width=0.4,
    num_terms=40, 
    random_power=1.0
):
    np.random.seed(seed)
    theta = np.linspace(0, 2 * np.pi, num_samples, endpoint=False)

    # MAIN LOBE: directional Gaussian
    angle_diff = np.angle(np.exp(1j * (theta - main_lobe_dir)))
    directional_bias = np.exp(-0.5 * (angle_diff / main_lobe_width) ** 2)
    directional_bias = directional_bias / directional_bias.max()  # normalize

    # RANDOM SIDES: richer random harmonics (more chaotic)
    an = np.random.randn(num_terms)
    bn = np.random.randn(num_terms)
    perturb = sum(
        (an[n] * np.cos((n + 1) * theta) + bn[n] * np.sin((n + 1) * theta)) / (n + 1)**random_power
        for n in range(num_terms)
    )
    perturb = (perturb - perturb.min()) / (perturb.max() - perturb.min())  

    # Combined pattern
    pattern = (1 - main_lobe_strength) * perturb + main_lobe_strength * directional_bias
    pattern = (pattern - pattern.min()) / (pattern.max() - pattern.min())

    return theta, pattern

def parabolic_antenna_pattern_3d(
    num_azimuth=360,
    num_elevation=180,
    az_main_lobe_dir=0,         # Azimuth direction (toward -X, optional)
    el_main_lobe_dir=np.pi / 2,     # Point lobe upward along Z-axis
    az_width=0.3,
    el_width=0.2,
    sidelobe_level=-10,              # dB relative to main lobe
    Laying=False
):
    az = np.linspace(0, 2 * np.pi, num_azimuth)
    el = np.linspace(0, np.pi, num_elevation)
    AZ, EL = np.meshgrid(az, el)
    if Laying:
        # When laying flat, the main lobe should point along the x/y plane
        # Calculate angular differences with proper wrapping
        az_diff = np.angle(np.exp(1j * (AZ - az_main_lobe_dir)))
        
        # For the elevation pattern, we need to transform it to be in the x/y plane
        # The "elevation" becomes the angle from the x/y plane (0 = in plane, pi/2 = up/down)
        el_diff = np.abs(EL - np.pi/2)  # How far from the horizon
        
        # Main lobe pattern
        main_lobe = np.exp(-(az_diff/az_width)**2 - (el_diff/el_width)**2)

        # Sidelobes pattern
        sidelobes = 10 ** (sidelobe_level / 10) * np.exp(-(az_diff/(az_width*3))**2) * np.exp(-(el_diff/(el_width*2))**2)

        combined = np.maximum(main_lobe, sidelobes)
        pattern_dBi = 10 * np.log10(combined)
        pattern_dBi -= np.max(pattern_dBi)
    else:
        az_diff = np.zeros_like(AZ) 
        el_diff = EL - el_main_lobe_dir
        
        # Pattern calculation (azimuth-independent)
        main_lobe = np.exp(-(el_diff/el_width)**2)
        sidelobes = 10**(sidelobe_level/10) * np.exp(-(el_diff/(el_width*2))**2)
        combined = np.maximum(main_lobe, sidelobes)
        
        pattern_dBi = 10*np.log10(combined)
        pattern_dBi -= np.max(pattern_dBi)
    

    return AZ.flatten(), EL.flatten(), pattern_dBi.flatten()

def planned_measurement_route_3d(
    radius=100.0,
    num_azimuths=360,
    num_elevations=10,
    center=(0.0, 0.0, 10.0),
    min_elevation_deg=0,
    max_elevation_deg=90
):
    cx, cy, cz = center
    planned_coords = []
    theta_list = []

    elevations = np.linspace(np.radians(min_elevation_deg), np.radians(max_elevation_deg), num_elevations)
    azimuths = np.linspace(0, 2 * np.pi, num_azimuths, endpoint=False)

    for el in elevations:
        for az in azimuths:
            x = cx + radius * np.sin(el) * np.cos(az)
            y = cy + radius * np.sin(el) * np.sin(az)
            z = cz + radius * np.cos(el)
            planned_coords.append([x, y, z])
            theta_list.append(az)  # For plotting or angle association

    return np.array(theta_list), np.array(planned_coords)

def MoveDrone(plannedPos, xy_deviation = 0.2, z_deviation=0.1):
    x_coords = plannedPos[:, 0]
    y_coords = plannedPos[:, 1]
    z_coords = plannedPos[:, 2]

    randomPosErrorX = np.random.normal(x_coords, xy_deviation)
    randomPosErrorY = np.random.normal(y_coords, xy_deviation)
    randomPosZ = np.random.normal(z_coords, z_deviation)

    return np.stack([randomPosErrorX, randomPosErrorY, randomPosZ],axis=1)

def friis_received_power(P_t_dBm, G_t_dBi, G_r_dBi, d_m, f_Hz):
    return P_t_dBm + G_t_dBi + G_r_dBi - 20 * np.log10(d_m) - 20 * np.log10(f_Hz) + 147.55

def estimate_antenna_gain(P_r_dBm, P_t_dBm, G_r_dBi, distances_m, f_Hz):
    P_r_dBm = np.array(P_r_dBm)
    return P_r_dBm - P_t_dBm- G_r_dBi + 20 * np.log10(distances_m) + 20 * np.log10(f_Hz) - 147.55
    
def spherical_to_cartesian(azimuth, elevation, r=1.0):
    x = r * np.sin(elevation) * np.cos(azimuth)
    y = r * np.sin(elevation) * np.sin(azimuth)
    z = r * np.cos(elevation)
    return x, y, z


if __name__=="__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #Antenna parameters.
    actual_gain_dBi = 30 #dBi
    az_main_lobe_dir = np.pi/4
    num_elation_slices = 90
    az_grid, el_grid, pattern = parabolic_antenna_pattern_3d(num_elevation=num_elation_slices, el_main_lobe_dir=0)
    antennaPos=[0.0,0.0,10.0]

    #Section for the "drone" planned coordinates and so on:
    plannedTheta, plannedCoords = planned_measurement_route_3d(radius=100,center=antennaPos, num_elevations=num_elation_slices)
    actualPos = MoveDrone(plannedCoords)
    x_coords = plannedCoords[:,0]
    y_coords = plannedCoords[:,1]
    z_coords = plannedCoords[:,2]
    x_actual = actualPos[:,0]
    y_actual = actualPos[:,1]
    z_actual = actualPos[:,2]
    dx, dy, dz = actualPos[:,0] - antennaPos[0], actualPos[:,1] - antennaPos[1], actualPos[:,2] - antennaPos[2]
    r = np.linalg.norm(np.stack([dx, dy, dz], axis=1), axis=1)
    azimuth = np.arctan2(dy, dx) % (2 * np.pi)
    elevation = np.arccos(dz / r)
    RealDistance = np.linalg.norm(antennaPos - actualPos, axis=1)

    #Finding power and generating the sinusoids with certain power.
    P_t_dBm = 0
    G_r_dBi = 8
    frequency_Hz = 5e9

    pattern_points = np.column_stack((az_grid.flatten(), el_grid.flatten()))
    actual_points = np.column_stack((azimuth, elevation))
    gain_per_point = griddata(pattern_points, pattern, actual_points, method='cubic', fill_value=-20)
    print("gain per point max: ", max(gain_per_point) )
    directional_gain_dBi = gain_per_point + actual_gain_dBi
    print("max pattern gain: ", max(directional_gain_dBi))
    # Compute received power using per-point transmitter gain
    received_power_dBm = friis_received_power(P_t_dBm, directional_gain_dBi, G_r_dBi, RealDistance, frequency_Hz)

    allsamples=[]
    allPowers = []
    for i in received_power_dBm:
        samples = generate_sampled_mixtures(signal_power_level=i, desired_len=1000, sig_freq=frequency_Hz, sampling_freq=15e9, sdr=20)
        mixture_array = np.stack([samples['mixture'].real, samples['mixture'].imag], axis=1).astype(np.float32)
        allPowers.append(samples['mix_power'])
        allsamples.append(mixture_array)

    model = torch.jit.load("./Deployment/Deploy.pth").to(device)
    model.eval()#.to(device)
    predicted_powers = []
    for i in allsamples:
        mixture_tensor = torch.from_numpy(i).unsqueeze(0).to(device)
        #t1 = datetime.now()
        with torch.no_grad():
            power = model(mixture_tensor)
        #t2 = datetime.now()
        #print(t2-t1)
        power_np = power.squeeze().cpu().numpy()
        calculated_power = calculate_power_dBm(power_np)
        predicted_powers.append(calculated_power)

    estimatedGain = estimate_antenna_gain(predicted_powers, P_t_dBm,G_r_dBi=G_r_dBi, distances_m=RealDistance, f_Hz=frequency_Hz)
    print("predicted received power: ",predicted_powers[0])
    print("Generated signal power : ", allPowers[0])
    print("max gain shape: ", max(directional_gain_dBi))
    print("max gain estimated: ", max(estimatedGain))
    #Plotting
    az_rad = azimuth  # already in radians
    el_rad = elevation  # already in radians
   
    plot_dronemeasured_gain_patterns(actualPos, antennaPos, estimatedGain)
    plot_estimated_radiation_pattern(antennaPos, actualPos, estimatedGain, num_elevation_slices=num_elation_slices)

    unique_els = np.unique(el_grid)
    closest_el_idx = np.argmin(np.abs(unique_els - 0))
    slice_indices = np.where(el_grid == unique_els[closest_el_idx])[0]

    theta = az_grid[slice_indices]
    directional_gain_dBi_slice = pattern[slice_indices]+actual_gain_dBi

    rangeBMax = 40
    plt.figure(figsize=(12, 6))
    ax1 = plt.subplot(1, 2, 1, polar=True)
    ax1.plot(theta, directional_gain_dBi_slice, label="Actual Radiation Pattern (dB)")
    ax1.plot(theta, estimatedGain[:len(theta)], label="Estimated Gain pattern")
    ax1.set_rmin(actual_gain_dBi - rangeBMax)
    ax1.set_rmax(actual_gain_dBi+10)
    ax1.set_rticks(np.arange(actual_gain_dBi-rangeBMax, actual_gain_dBi+10 + 1, 5))
    ax1.plot([az_main_lobe_dir, az_main_lobe_dir], [0, actual_gain_dBi+5], 
         'r--', linewidth=1, label='Main Lobe Direction')
    ax1.set_title("Radiation Pattern (Log dB)")
    ax1.legend()

    ax2 = plt.subplot(1, 2, 2)
    ax2.scatter(x_coords, y_coords, label="Measurement Route")
    ax2.scatter(x_actual, y_actual,label="Actual Route")
    ax2.set_aspect('equal', 'box')
    ax2.set_title("Measurement Route (XY Plane)")
    ax2.set_xlabel("X (m)")
    ax2.set_ylabel("Y (m)")
    ax2.grid(True)
    ax2.legend()

    plt.tight_layout()
    plt.show()


    az_laying, el_laying, pattern_laying = parabolic_antenna_pattern_3d(
    num_elevation=num_elation_slices, 
    az_main_lobe_dir=np.pi/2, 
    el_main_lobe_dir=np.pi/2, 
    Laying=True
    )
    
    unique_els = np.unique(el_laying)
    closest_el_idx = np.argmin(np.abs(unique_els - np.pi/2))
    slice_indices = np.where(el_laying == unique_els[closest_el_idx])[0]

    theta = az_laying[slice_indices]
    directional_gain_dBi_slice = pattern[slice_indices]+actual_gain_dBi

    rangeBMax = 40
    plt.figure(figsize=(12, 6))
    ax1 = plt.subplot(1, 1, 1, polar=True)
    ax1.plot(theta, directional_gain_dBi_slice, label="Actual Radiation Pattern (dB)")
    ax1.set_rmin(actual_gain_dBi - rangeBMax)
    ax1.set_rmax(actual_gain_dBi+10)
    ax1.set_rticks(np.arange(actual_gain_dBi-rangeBMax, actual_gain_dBi+10 + 1, 5))
    ax1.plot([az_main_lobe_dir, az_main_lobe_dir], [0, actual_gain_dBi+5], 
         'r--', linewidth=1, label='Main Lobe Direction')
    ax1.set_title("Radiation Pattern (Log dB)")
    ax1.legend()
    plt.show()

    plottingGeneratedPattern(az_grid, el_grid, pattern,num_el=num_elation_slices)





