"""
Minimal EDF (European Data Format) reader - no external dependencies.
Follows the standard EDF specification: https://www.edfplus.info/specs/edf.html

Scope: only plain EDF (16-bit samples, no annotation channel) is supported.
BDF (BioSemi, 24-bit samples) and EDF+ (adds an "EDF Annotations"
pseudo-channel that is not a numeric signal) are explicitly rejected rather
than silently mis-parsed - this reader has only been verified against plain
EDF files (OpenNeuro ds007788), and misreading BDF's 24-bit samples as EDF's
16-bit ones, or an EDF+ annotations channel as a regular numeric channel,
would produce corrupted signal data without any error.
"""
import numpy as np


def read_edf(filepath):
    with open(filepath, 'rb') as f:
        header = f.read(256)

        if header[0:1] == b'\xff' and header[1:8] == b'BIOSEMI':
            raise ValueError(
                f"{filepath}: this is a BDF (BioSemi) file, not EDF. BDF uses "
                "24-bit samples; this reader only supports plain EDF's 16-bit "
                "samples and would misread BDF data as corrupted noise."
            )

        version = header[0:8].decode('ascii', errors='replace').strip()
        patient_id = header[8:88].decode('ascii', errors='replace').strip()
        recording_id = header[88:168].decode('ascii', errors='replace').strip()
        start_date = header[168:176].decode('ascii', errors='replace').strip()
        start_time = header[176:184].decode('ascii', errors='replace').strip()
        n_header_bytes = int(header[184:192].decode('ascii').strip())
        reserved = header[192:236].decode('ascii', errors='replace').strip()
        n_records = int(header[236:244].decode('ascii').strip())
        record_duration = float(header[244:252].decode('ascii').strip())
        n_signals = int(header[252:256].decode('ascii').strip())

        if reserved.startswith('EDF+'):
            raise ValueError(
                f"{filepath}: this is an EDF+ file (reserved field={reserved!r}). "
                "EDF+ adds an 'EDF Annotations' pseudo-channel holding variable-"
                "length text, not a numeric signal; this reader would misread it "
                "as regular sample data. Only plain EDF is currently supported."
            )

        # Per-signal header fields
        def read_field(n_bytes):
            return [f.read(n_bytes).decode('ascii', errors='replace').strip()
                    for _ in range(n_signals)]

        labels = read_field(16)

        if 'EDF Annotations' in labels:
            raise ValueError(
                f"{filepath}: found an 'EDF Annotations' channel even though "
                "the reserved header field did not mark this as EDF+. This "
                "reader cannot parse annotation channels as numeric signals - "
                "refusing to continue rather than silently corrupt the data."
            )
        transducer = read_field(80)
        phys_dim = read_field(8)
        phys_min = [float(x) for x in read_field(8)]
        phys_max = [float(x) for x in read_field(8)]
        dig_min = [int(x) for x in read_field(8)]
        dig_max = [int(x) for x in read_field(8)]
        prefiltering = read_field(80)
        n_samples_per_record = [int(x) for x in read_field(8)]
        reserved_sig = read_field(32)

        # Data records
        data = {label: [] for label in labels}
        record_size = sum(n_samples_per_record)

        raw = f.read()
        n_records_actual = len(raw) // (record_size * 2)  # 2 bytes per sample (int16)
        n_records_use = min(n_records, n_records_actual) if n_records > 0 else n_records_actual

        arr = np.frombuffer(raw[:n_records_use * record_size * 2], dtype='<i2')
        arr = arr.reshape(n_records_use, record_size)

        offset = 0
        signals = {}
        fs_per_channel = {}
        for i, label in enumerate(labels):
            ns = n_samples_per_record[i]
            chan_digital = arr[:, offset:offset + ns].flatten()
            offset += ns

            # digital -> physical conversion
            dmin, dmax = dig_min[i], dig_max[i]
            pmin, pmax = phys_min[i], phys_max[i]
            scale = (pmax - pmin) / (dmax - dmin) if dmax != dmin else 1.0
            phys = (chan_digital.astype(np.float64) - dmin) * scale + pmin

            signals[label] = phys
            fs_per_channel[label] = ns / record_duration

    info = {
        'version': version,
        'patient_id': patient_id,
        'recording_id': recording_id,
        'start_date': start_date,
        'start_time': start_time,
        'n_records': n_records_use,
        'record_duration': record_duration,
        'n_signals': n_signals,
        'labels': labels,
        'phys_dim': dict(zip(labels, phys_dim)),
        'sampling_rate': fs_per_channel,
        'prefiltering': dict(zip(labels, prefiltering)),
    }

    return signals, info


if __name__ == '__main__':
    import sys
    signals, info = read_edf(sys.argv[1])
    print("EDF Info:")
    for k, v in info.items():
        if k not in ('sampling_rate', 'phys_dim', 'prefiltering'):
            print(f"  {k}: {v}")
    print("\nChannels & sampling rates:")
    for label in info['labels']:
        print(f"  {label:20s} fs={info['sampling_rate'][label]:.1f}Hz  "
              f"n_samples={len(signals[label])}  unit={info['phys_dim'][label]}")
