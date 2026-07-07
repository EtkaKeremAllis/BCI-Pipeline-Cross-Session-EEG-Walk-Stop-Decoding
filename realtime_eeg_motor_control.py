import numpy as np
from scipy import signal
from collections import deque
import threading
import time
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.patches as mpatches

class RealTimeEEGProcessor:
    """
    Real-time EEG processor - synchronous noise reduction and motor command triggering
    - Sample Rate: 256 Hz
    - Line Noise: 50 Hz
    - Electrodes: C3, C4, Cz (sensorimotor cortex)
    """
    
    def __init__(self, sampling_rate=256, buffer_size=256):
        """
        Args:
            sampling_rate: Sampling rate (Hz)
            buffer_size: Buffer size (1 second = 256 samples)
        """
        self.fs = sampling_rate
        self.buffer_size = buffer_size
        self.nyquist = sampling_rate / 2
        
        # Buffers for the electrodes
        self.buffers = {
            'C3': deque(maxlen=buffer_size),
            'C4': deque(maxlen=buffer_size),
            'Cz': deque(maxlen=buffer_size)
        }
        
        # Threshold values (adaptive)
        self.thresholds = {
            'C3': 0.8,
            'C4': 0.8,
            'Cz': 0.7
        }
        
        # Motor command state
        self.motor_command_active = False
        self.command_history = []
        
        # Statistics
        self.stats = {
            'C3': {'mu': [], 'beta': []},
            'C4': {'mu': [], 'beta': []},
            'Cz': {'mu': [], 'beta': []}
        }
    
    def create_bandpass_sos(self, lowcut, highcut, order=3):
        """
        Bandpass filter (low order)
        
        Args:
            lowcut, highcut: Frequency range
            order: Filter order (low: stability)
        
        Returns:
            Filter in SOS format
        """
        low_norm = lowcut / self.nyquist
        high_norm = highcut / self.nyquist
        sos = signal.butter(order, [low_norm, high_norm], btype='band', output='sos')
        return sos
    
    def create_notch_sos(self, freq=50, quality=20):
        """Notch filter (50 Hz line noise)"""
        w0_norm = freq / self.nyquist
        b, a = signal.iirnotch(w0_norm, quality)
        sos = signal.tf2sos(b, a)
        return sos
    
    def process_signal(self, raw_signal, electrode='C3'):
        """
        Process the raw signal (real-time noise reduction)
        
        Args:
            raw_signal: Raw signal array
            electrode: Electrode name (C3, C4, Cz)
        
        Returns:
            Processed signal, Mu band energy, Beta band energy
        """
        # 1. Notch filter (50 Hz)
        notch_sos = self.create_notch_sos(freq=50, quality=20)
        notch_filtered = signal.sosfiltfilt(notch_sos, raw_signal)
        
        # 2. Mu rhythm (8-13 Hz) - sensorimotor activity
        mu_sos = self.create_bandpass_sos(lowcut=8, highcut=13, order=3)
        mu_signal = signal.sosfiltfilt(mu_sos, notch_filtered)
        mu_energy = np.mean(mu_signal ** 2)  # Mu band energy
        
        # 3. Beta rhythm (13-30 Hz) - motor control
        beta_sos = self.create_bandpass_sos(lowcut=13, highcut=30, order=3)
        beta_signal = signal.sosfiltfilt(beta_sos, notch_filtered)
        beta_energy = np.mean(beta_signal ** 2)  # Beta band energy
        
        # Save statistics
        self.stats[electrode]['mu'].append(mu_energy)
        self.stats[electrode]['beta'].append(beta_energy)
        
        return notch_filtered, mu_energy, beta_energy
    
    def detect_motor_command(self, electrode_data):
        """
        Motor command detection (threshold detection)
        
        Args:
            electrode_data: dict in the form {electrode: energy}
        
        Returns:
            Whether a motor command was triggered (boolean)
        """
        command_triggered = False
        triggered_electrodes = []
        
        for electrode, energy in electrode_data.items():
            if energy > self.thresholds[electrode]:
                command_triggered = True
                triggered_electrodes.append(electrode)
        
        if command_triggered:
            self.motor_command_active = True
            self.command_history.append({
                'time': time.time(),
                'electrodes': triggered_electrodes,
                'energy': electrode_data
            })
        
        return command_triggered, triggered_electrodes
    
    def generate_joystick_command(self, triggered_electrodes):
        """
        Generate joystick walk command
        
        Args:
            triggered_electrodes: Triggered electrodes
        
        Returns:
            Joystick command (string)
        """
        if 'C3' in triggered_electrodes or 'C4' in triggered_electrodes:
            return "FORWARD"  # Forward movement
        elif 'Cz' in triggered_electrodes:
            return "STOP"     # Stop
        return None
    
    def apply_joystick_command(self, command):
        """
        Apply joystick command (simulation)
        
        Args:
            command: Joystick command
        """
        if command == "FORWARD":
            print(f"[⚡ MOTOR COMMAND] FORWARD - Walking activation detected!")
            # Real joystick control could be implemented here
            # implementation with pynput or PyGame
            return True
        elif command == "STOP":
            print(f"[⚡ MOTOR COMMAND] STOP - Walking stop signal!")
            return False
        return None


class EEGDataGenerator:
    """
    EEG data generator - Rest vs Walking data
    """
    
    def __init__(self, sampling_rate=256, duration=10):
        """
        Args:
            sampling_rate: Sampling rate
            duration: Data duration (seconds)
        """
        self.fs = sampling_rate
        self.duration = duration
        self.t = np.arange(0, duration, 1/sampling_rate)
    
    def generate_normal_eeg(self):
        """Normal EEG data (resting state) - Mu rhythm dominant"""
        # C3: Left motor cortex - Mu wave dominant
        c3_signal = (
            2.0 * np.sin(2 * np.pi * 10 * self.t) +      # 10 Hz Mu
            0.5 * np.sin(2 * np.pi * 20 * self.t) +      # Less beta
            0.3 * np.random.randn(len(self.t))            # Noise
        )
        
        # C4: Right motor cortex
        c4_signal = (
            2.1 * np.sin(2 * np.pi * 10 * self.t + 0.5) + # 10 Hz Mu (phase difference)
            0.4 * np.sin(2 * np.pi * 20 * self.t) +        # Beta
            0.3 * np.random.randn(len(self.t))
        )
        
        # Cz: Midline (leg control) - Less activity
        cz_signal = (
            1.0 * np.sin(2 * np.pi * 10 * self.t) +        # Weak Mu
            0.3 * np.sin(2 * np.pi * 20 * self.t) +        # Weak Beta
            0.3 * np.random.randn(len(self.t))
        )
        
        # Add 50 Hz line noise
        line_noise = 0.5 * np.sin(2 * np.pi * 50 * self.t)
        
        c3_signal += line_noise
        c4_signal += line_noise
        cz_signal += line_noise
        
        return {
            'C3': c3_signal,
            'C4': c4_signal,
            'Cz': cz_signal,
            'time': self.t
        }
    
    def generate_walking_eeg(self):
        """
        EEG data while walking
        - Mu rhythm suppression (desynchronization): 8-13 Hz decreases
        - Increase in Beta activity: 13-30 Hz increases
        - Leg control (Cz) more dominant
        """
        # Walking signal - 0.5 Hz (gait rhythm)
        walking_phase = 3.0 * np.sin(2 * np.pi * 0.5 * self.t)
        
        # C3: Left motor cortex - Mu suppression, Beta increase
        c3_signal = (
            0.8 * np.sin(2 * np.pi * 10 * self.t) +       # Mu decreased (2.0 -> 0.8)
            1.5 * np.sin(2 * np.pi * 20 * self.t + walking_phase) +  # Beta increased (0.5 -> 1.5)
            0.4 * np.random.randn(len(self.t))
        )
        
        # C4: Right motor cortex - Mu suppression, Beta increase
        c4_signal = (
            0.9 * np.sin(2 * np.pi * 10 * self.t + 0.5) + # Mu decreased
            1.4 * np.sin(2 * np.pi * 20 * self.t + walking_phase + 0.3) +  # Beta increased
            0.4 * np.random.randn(len(self.t))
        )
        
        # Cz: Leg control - More dominant
        cz_signal = (
            0.5 * np.sin(2 * np.pi * 10 * self.t) +        # Mu greatly decreased
            2.0 * np.sin(2 * np.pi * 20 * self.t + walking_phase) +  # Beta greatly increased
            0.4 * np.random.randn(len(self.t))
        )
        
        # 50 Hz line noise
        line_noise = 0.5 * np.sin(2 * np.pi * 50 * self.t)
        
        c3_signal += line_noise
        c4_signal += line_noise
        cz_signal += line_noise
        
        return {
            'C3': c3_signal,
            'C4': c4_signal,
            'Cz': cz_signal,
            'time': self.t
        }


def analyze_eeg_recordings():
    """
    Analyze Normal and Walking EEG recordings
    """
    print("=" * 70)
    print("REAL-TIME EEG PROCESSOR - Normal vs Walking Analysis")
    print("=" * 70)
    
    # Parameters
    sampling_rate = 256
    duration = 10
    
    # Generate data
    generator = EEGDataGenerator(sampling_rate=sampling_rate, duration=duration)
    normal_eeg = generator.generate_normal_eeg()
    walking_eeg = generator.generate_walking_eeg()
    
    # Create processor
    processor = RealTimeEEGProcessor(sampling_rate=sampling_rate)
    
    print("\n[1] NORMAL EEG ANALYSIS (Resting State)")
    print("-" * 70)
    
    normal_results = {}
    for electrode in ['C3', 'C4', 'Cz']:
        raw = normal_eeg[electrode]
        filtered, mu_energy, beta_energy = processor.process_signal(raw, electrode)
        
        normal_results[electrode] = {
            'mu_energy': mu_energy,
            'beta_energy': beta_energy,
            'ratio': mu_energy / (beta_energy + 1e-6)
        }
        
        print(f"  {electrode}:")
        print(f"    Mu Energy (8-13 Hz):   {mu_energy:.6f}")
        print(f"    Beta Energy (13-30 Hz): {beta_energy:.6f}")
        print(f"    Mu/Beta Ratio:         {normal_results[electrode]['ratio']:.4f}")
    
    # Detect motor command - Normal
    electrode_energy_normal = {
        'C3': normal_results['C3']['mu_energy'],
        'C4': normal_results['C4']['mu_energy'],
        'Cz': normal_results['Cz']['mu_energy']
    }
    command_triggered, electrodes = processor.detect_motor_command(electrode_energy_normal)
    print(f"\n  Motor Command Triggered: {command_triggered}")
    
    # Reset processor
    processor = RealTimeEEGProcessor(sampling_rate=sampling_rate)
    
    print("\n[2] WALKING EEG ANALYSIS (Movement State)")
    print("-" * 70)
    
    walking_results = {}
    for electrode in ['C3', 'C4', 'Cz']:
        raw = walking_eeg[electrode]
        filtered, mu_energy, beta_energy = processor.process_signal(raw, electrode)
        
        walking_results[electrode] = {
            'mu_energy': mu_energy,
            'beta_energy': beta_energy,
            'ratio': mu_energy / (beta_energy + 1e-6)
        }
        
        print(f"  {electrode}:")
        print(f"    Mu Energy (8-13 Hz):    {mu_energy:.6f}")
        print(f"    Beta Energy (13-30 Hz):  {beta_energy:.6f}")
        print(f"    Mu/Beta Ratio:          {walking_results[electrode]['ratio']:.4f}")
    
    # Detect motor command - Walking
    electrode_energy_walking = {
        'C3': walking_results['C3']['beta_energy'],  # Look at Beta
        'C4': walking_results['C4']['beta_energy'],
        'Cz': walking_results['Cz']['beta_energy']
    }
    
    # Adjust thresholds based on Beta energy
    processor.thresholds = {
        'C3': 0.8,
        'C4': 0.8,
        'Cz': 0.7
    }
    
    command_triggered, electrodes = processor.detect_motor_command(electrode_energy_walking)
    print(f"\n  Motor Command Triggered: {command_triggered}")
    
    if command_triggered:
        joystick_cmd = processor.generate_joystick_command(electrodes)
        print(f"  Triggered Electrodes: {electrodes}")
        print(f"  Joystick Command: {joystick_cmd}")
        processor.apply_joystick_command(joystick_cmd)
    
    # COMPARISON
    print("\n[3] NORMAL vs WALKING COMPARISON")
    print("-" * 70)
    
    for electrode in ['C3', 'C4', 'Cz']:
        mu_change = ((walking_results[electrode]['mu_energy'] - normal_results[electrode]['mu_energy']) 
                     / (normal_results[electrode]['mu_energy'] + 1e-6)) * 100
        
        beta_change = ((walking_results[electrode]['beta_energy'] - normal_results[electrode]['beta_energy']) 
                       / (normal_results[electrode]['beta_energy'] + 1e-6)) * 100
        
        print(f"\n  {electrode}:")
        print(f"    Mu Change:   {mu_change:+.2f}% (Suppression = negative)")
        print(f"    Beta Change: {beta_change:+.2f}% (Activation = positive)")
    
    # Visualization
    plot_eeg_comparison(normal_eeg, walking_eeg, normal_results, walking_results)


def plot_eeg_comparison(normal_eeg, walking_eeg, normal_results, walking_results):
    """Visualize the EEG recordings"""
    
    t_normal = normal_eeg['time']
    t_walking = walking_eeg['time']
    
    fig = plt.figure(figsize=(16, 12))
    
    # Normal EEG
    for i, electrode in enumerate(['C3', 'C4', 'Cz']):
        ax = plt.subplot(3, 2, i*2 + 1)
        ax.plot(t_normal[:1000], normal_eeg[electrode][:1000], 'b-', linewidth=0.8, alpha=0.7)
        ax.set_ylabel(f'{electrode} (μV)', fontsize=10)
        ax.set_title(f'Normal EEG - {electrode}', fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 4])
        
        # Statistics
        mu_energy = normal_results[electrode]['mu_energy']
        beta_energy = normal_results[electrode]['beta_energy']
        ax.text(0.02, 0.95, f'Mu: {mu_energy:.4f}\nBeta: {beta_energy:.4f}',
                transform=ax.transAxes, fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Walking EEG
    for i, electrode in enumerate(['C3', 'C4', 'Cz']):
        ax = plt.subplot(3, 2, i*2 + 2)
        ax.plot(t_walking[:1000], walking_eeg[electrode][:1000], 'r-', linewidth=0.8, alpha=0.7)
        ax.set_ylabel(f'{electrode} (μV)', fontsize=10)
        ax.set_title(f'Walking EEG - {electrode}', fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 4])
        
        # Statistics
        mu_energy = walking_results[electrode]['mu_energy']
        beta_energy = walking_results[electrode]['beta_energy']
        ax.text(0.02, 0.95, f'Mu: {mu_energy:.4f}\nBeta: {beta_energy:.4f}',
                transform=ax.transAxes, fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig('C:\\EEG_Project\\eeg_normal_vs_walking.png', dpi=150, bbox_inches='tight')
    print("\n[INFO] Chart saved: C:\\EEG_Project\\eeg_normal_vs_walking.png")
    plt.show()


if __name__ == "__main__":
    analyze_eeg_recordings()
