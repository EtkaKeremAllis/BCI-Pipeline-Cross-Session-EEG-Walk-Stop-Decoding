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
    Real-time EEG işlemci - Es zamanli noise reduction ve motor komutu tetikleme
    - Sample Rate: 256 Hz
    - Şebeke Gürültüsü: 50 Hz
    - Elektrotlar: C3, C4, Cz (sensorimotor korteks)
    """
    
    def __init__(self, sampling_rate=256, buffer_size=256):
        """
        Args:
            sampling_rate: Örnekleme hızı (Hz)
            buffer_size: Buffer boyutu (1 saniye = 256 örnek)
        """
        self.fs = sampling_rate
        self.buffer_size = buffer_size
        self.nyquist = sampling_rate / 2
        
        # Elektrotlar için buffer'lar
        self.buffers = {
            'C3': deque(maxlen=buffer_size),
            'C4': deque(maxlen=buffer_size),
            'Cz': deque(maxlen=buffer_size)
        }
        
        # Threshold değerleri (adaptive)
        self.thresholds = {
            'C3': 0.8,
            'C4': 0.8,
            'Cz': 0.7
        }
        
        # Motor komutu state
        self.motor_command_active = False
        self.command_history = []
        
        # İstatistikler
        self.stats = {
            'C3': {'mu': [], 'beta': []},
            'C4': {'mu': [], 'beta': []},
            'Cz': {'mu': [], 'beta': []}
        }
    
    def create_bandpass_sos(self, lowcut, highcut, order=3):
        """
        Bandpass filtresi (düşük order)
        
        Args:
            lowcut, highcut: Frekans aralığı
            order: Filtre sırası (düşük: stabilite)
        
        Returns:
            SOS formatında filtre
        """
        low_norm = lowcut / self.nyquist
        high_norm = highcut / self.nyquist
        sos = signal.butter(order, [low_norm, high_norm], btype='band', output='sos')
        return sos
    
    def create_notch_sos(self, freq=50, quality=20):
        """Notch filtresi (50 Hz şebeke gürültüsü)"""
        w0_norm = freq / self.nyquist
        b, a = signal.iirnotch(w0_norm, quality)
        sos = signal.tf2sos(b, a)
        return sos
    
    def process_signal(self, raw_signal, electrode='C3'):
        """
        Raw sinyali işle (real-time noise reduction)
        
        Args:
            raw_signal: Ham sinyal array
            electrode: Elektrot adı (C3, C4, Cz)
        
        Returns:
            Işlenmiş sinyal, Mu bandı enerji, Beta bandı enerji
        """
        # 1. Notch filtresi (50 Hz)
        notch_sos = self.create_notch_sos(freq=50, quality=20)
        notch_filtered = signal.sosfiltfilt(notch_sos, raw_signal)
        
        # 2. Mu ritmi (8-13 Hz) - sensorimotor aktivite
        mu_sos = self.create_bandpass_sos(lowcut=8, highcut=13, order=3)
        mu_signal = signal.sosfiltfilt(mu_sos, notch_filtered)
        mu_energy = np.mean(mu_signal ** 2)  # Mu bandı enerjisi
        
        # 3. Beta ritmi (13-30 Hz) - motor kontrol
        beta_sos = self.create_bandpass_sos(lowcut=13, highcut=30, order=3)
        beta_signal = signal.sosfiltfilt(beta_sos, notch_filtered)
        beta_energy = np.mean(beta_signal ** 2)  # Beta bandı enerjisi
        
        # İstatistikler kaydet
        self.stats[electrode]['mu'].append(mu_energy)
        self.stats[electrode]['beta'].append(beta_energy)
        
        return notch_filtered, mu_energy, beta_energy
    
    def detect_motor_command(self, electrode_data):
        """
        Motor komutu algılama (treshold detection)
        
        Args:
            electrode_data: {electrode: energy} şeklinde dict
        
        Returns:
            Motor komudu tetiklenmiş mi? (boolean)
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
        Joystick yürüme komudu oluştur
        
        Args:
            triggered_electrodes: Tetiklenen elektrotlar
        
        Returns:
            Joystick komudu (string)
        """
        if 'C3' in triggered_electrodes or 'C4' in triggered_electrodes:
            return "FORWARD"  # İleri hareket
        elif 'Cz' in triggered_electrodes:
            return "STOP"     # Durdur
        return None
    
    def apply_joystick_command(self, command):
        """
        Joystick komudu uygula (simülasyon)
        
        Args:
            command: Joystick komudu
        """
        if command == "FORWARD":
            print(f"[⚡ MOTOR COMMAND] FORWARD - Yürüme aktivasyonu algılandı!")
            # Burada gerçek joystick kontrolü yapılabilir
            # pynput veya PyGame ile implementasyon
            return True
        elif command == "STOP":
            print(f"[⚡ MOTOR COMMAND] STOP - Yürüme durdurma sinyali!")
            return False
        return None


class EEGDataGenerator:
    """
    EEG veri oluşturucu - Normal vs Yürürken verileri
    """
    
    def __init__(self, sampling_rate=256, duration=10):
        """
        Args:
            sampling_rate: Örnekleme hızı
            duration: Veri süresi (saniye)
        """
        self.fs = sampling_rate
        self.duration = duration
        self.t = np.arange(0, duration, 1/sampling_rate)
    
    def generate_normal_eeg(self):
        """Normal EEG verisi (dinlenme durumu) - Mu ritmi baskınlık"""
        # C3: Sol motor korteks - Mu dalgası baskın
        c3_signal = (
            2.0 * np.sin(2 * np.pi * 10 * self.t) +      # 10 Hz Mu
            0.5 * np.sin(2 * np.pi * 20 * self.t) +      # Beta daha az
            0.3 * np.random.randn(len(self.t))            # Gürültü
        )
        
        # C4: Sağ motor korteks
        c4_signal = (
            2.1 * np.sin(2 * np.pi * 10 * self.t + 0.5) + # 10 Hz Mu (faz farkı)
            0.4 * np.sin(2 * np.pi * 20 * self.t) +        # Beta
            0.3 * np.random.randn(len(self.t))
        )
        
        # Cz: Orta hat (bacak kontrolü) - Daha az aktivite
        cz_signal = (
            1.0 * np.sin(2 * np.pi * 10 * self.t) +        # Zayıf Mu
            0.3 * np.sin(2 * np.pi * 20 * self.t) +        # Zayıf Beta
            0.3 * np.random.randn(len(self.t))
        )
        
        # 50 Hz şebeke gürültüsü ekle
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
        Yürürken EEG verisi
        - Mu ritmi baskılanması (desynchronization): 8-13 Hz azalır
        - Beta aktivitesi artışı: 13-30 Hz artar
        - Bacak kontrolü (Cz) daha baskın
        """
        # Yürüme sinyal - 0.5 Hz (yaşamsal ritim)
        walking_phase = 3.0 * np.sin(2 * np.pi * 0.5 * self.t)
        
        # C3: Sol motor korteks - Mu baskılanması, Beta artışı
        c3_signal = (
            0.8 * np.sin(2 * np.pi * 10 * self.t) +       # Mu azaldı (2.0 → 0.8)
            1.5 * np.sin(2 * np.pi * 20 * self.t + walking_phase) +  # Beta arttı (0.5 → 1.5)
            0.4 * np.random.randn(len(self.t))
        )
        
        # C4: Sağ motor korteks - Mu baskılanması, Beta artışı
        c4_signal = (
            0.9 * np.sin(2 * np.pi * 10 * self.t + 0.5) + # Mu azaldı
            1.4 * np.sin(2 * np.pi * 20 * self.t + walking_phase + 0.3) +  # Beta arttı
            0.4 * np.random.randn(len(self.t))
        )
        
        # Cz: Bacak kontrolü - Daha baskın
        cz_signal = (
            0.5 * np.sin(2 * np.pi * 10 * self.t) +        # Mu çok azaldı
            2.0 * np.sin(2 * np.pi * 20 * self.t + walking_phase) +  # Beta çok arttı
            0.4 * np.random.randn(len(self.t))
        )
        
        # 50 Hz şebeke gürültüsü
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
    Normal ve Yürürken EEG kayıtlarını analiz et
    """
    print("=" * 70)
    print("REAL-TIME EEG PROCESSOR - Normal vs Yürürken Analizi")
    print("=" * 70)
    
    # Parametreler
    sampling_rate = 256
    duration = 10
    
    # Veri oluştur
    generator = EEGDataGenerator(sampling_rate=sampling_rate, duration=duration)
    normal_eeg = generator.generate_normal_eeg()
    walking_eeg = generator.generate_walking_eeg()
    
    # Processor oluştur
    processor = RealTimeEEGProcessor(sampling_rate=sampling_rate)
    
    print("\n[1] NORMAL EEG ANALİZİ (Dinlenme Durumu)")
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
        print(f"    Mu Enerji (8-13 Hz):   {mu_energy:.6f}")
        print(f"    Beta Enerji (13-30 Hz): {beta_energy:.6f}")
        print(f"    Mu/Beta Oranı:         {normal_results[electrode]['ratio']:.4f}")
    
    # Motor komutu algıla - Normal
    electrode_energy_normal = {
        'C3': normal_results['C3']['mu_energy'],
        'C4': normal_results['C4']['mu_energy'],
        'Cz': normal_results['Cz']['mu_energy']
    }
    command_triggered, electrodes = processor.detect_motor_command(electrode_energy_normal)
    print(f"\n  Motor Komudu Tetiklendi: {command_triggered}")
    
    # Processor reset
    processor = RealTimeEEGProcessor(sampling_rate=sampling_rate)
    
    print("\n[2] YÜRÜRKEN EEG ANALİZİ (Hareket Durumu)")
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
        print(f"    Mu Enerji (8-13 Hz):    {mu_energy:.6f}")
        print(f"    Beta Enerji (13-30 Hz):  {beta_energy:.6f}")
        print(f"    Mu/Beta Oranı:          {walking_results[electrode]['ratio']:.4f}")
    
    # Motor komutu algıla - Yürürken
    electrode_energy_walking = {
        'C3': walking_results['C3']['beta_energy'],  # Beta'ya bakalım
        'C4': walking_results['C4']['beta_energy'],
        'Cz': walking_results['Cz']['beta_energy']
    }
    
    # Threshold'ları Beta enerjisine göre ayarla
    processor.thresholds = {
        'C3': 0.8,
        'C4': 0.8,
        'Cz': 0.7
    }
    
    command_triggered, electrodes = processor.detect_motor_command(electrode_energy_walking)
    print(f"\n  Motor Komudu Tetiklendi: {command_triggered}")
    
    if command_triggered:
        joystick_cmd = processor.generate_joystick_command(electrodes)
        print(f"  Tetiklenen Elektrotlar: {electrodes}")
        print(f"  Joystick Komudu: {joystick_cmd}")
        processor.apply_joystick_command(joystick_cmd)
    
    # KARŞILAŞTIRMA
    print("\n[3] NORMAL vs YÜRÜRKEN KARŞILAŞTIRMASI")
    print("-" * 70)
    
    for electrode in ['C3', 'C4', 'Cz']:
        mu_change = ((walking_results[electrode]['mu_energy'] - normal_results[electrode]['mu_energy']) 
                     / (normal_results[electrode]['mu_energy'] + 1e-6)) * 100
        
        beta_change = ((walking_results[electrode]['beta_energy'] - normal_results[electrode]['beta_energy']) 
                       / (normal_results[electrode]['beta_energy'] + 1e-6)) * 100
        
        print(f"\n  {electrode}:")
        print(f"    Mu Değişimi:   {mu_change:+.2f}% (Baskılanma = negatif)")
        print(f"    Beta Değişimi: {beta_change:+.2f}% (Aktivasyon = pozitif)")
    
    # Görselleştirme
    plot_eeg_comparison(normal_eeg, walking_eeg, normal_results, walking_results)


def plot_eeg_comparison(normal_eeg, walking_eeg, normal_results, walking_results):
    """EEG kayıtlarını görselleştir"""
    
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
        
        # İstatistik
        mu_energy = normal_results[electrode]['mu_energy']
        beta_energy = normal_results[electrode]['beta_energy']
        ax.text(0.02, 0.95, f'Mu: {mu_energy:.4f}\nBeta: {beta_energy:.4f}',
                transform=ax.transAxes, fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Yürürken EEG
    for i, electrode in enumerate(['C3', 'C4', 'Cz']):
        ax = plt.subplot(3, 2, i*2 + 2)
        ax.plot(t_walking[:1000], walking_eeg[electrode][:1000], 'r-', linewidth=0.8, alpha=0.7)
        ax.set_ylabel(f'{electrode} (μV)', fontsize=10)
        ax.set_title(f'Yürürken EEG - {electrode}', fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 4])
        
        # İstatistik
        mu_energy = walking_results[electrode]['mu_energy']
        beta_energy = walking_results[electrode]['beta_energy']
        ax.text(0.02, 0.95, f'Mu: {mu_energy:.4f}\nBeta: {beta_energy:.4f}',
                transform=ax.transAxes, fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig('C:\\EEG_Project\\eeg_normal_vs_walking.png', dpi=150, bbox_inches='tight')
    print("\n[INFO] Grafik kaydedildi: C:\\EEG_Project\\eeg_normal_vs_walking.png")
    plt.show()


if __name__ == "__main__":
    analyze_eeg_recordings()
