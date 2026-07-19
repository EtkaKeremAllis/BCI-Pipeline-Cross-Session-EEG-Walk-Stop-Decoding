# Canli tahmin web arayuzu

Arayuz EDF kaydini gercek EEG hizinda oynatir ve WALK/STOP tahminlerini,
smoothed WALK/STOP komutunu, guven skorunu, EEG zamanini ve gecikme
metriklerini tarayicida canli gosterir.

Varsayilan smoothing penceresi 3 karardir. 50 ms karar adiminda bu, ham model
islem gecikmesine ek olarak sabit 50 ms smoothing gecikmesi getirir:

```text
total_latency_ms = end_to_end_ms + smoothing_wall_delay_ms
```

`smoothing_delay_ms` pencere ve karar adimindan hesaplanan teorik gecikmedir.
`smoothing_wall_delay_ms` ise kaydin gelisinden smoothed karar emit edilene kadar
gecen wall-clock sureden modelin `end_to_end_ms` suresi cikarilarak olculur. Bu
alan Python scheduling, GC ve diger jitter etkilerini de kapsar. Arayuzde ve
CSV'de `total_latency_ms`, bu olculen smoothing gecikmesi dahil gercek komut
gecikmesidir.
`end_to_end_ms` yalnizca feature cikarma ve model karar suresidir.

## Kurulum

```powershell
python -m pip install -r requirements.txt
```

## Calistirma

`latest` klasorunde:

```powershell
python fast_causal_web_ui.py --port 8766
```

Ardindan tarayicida `http://127.0.0.1:8766` adresini acin. Arayuzde EDF dosyasini
ve uygun model klasorunu secip **Baslat** dugmesine basin. Events TSV zorunlu
degildir; verilirse gercek etiket de son tahminler tablosunda gosterilir.

Komut satirindan varsayilan alanlar da verilebilir:

```powershell
python fast_causal_web_ui.py --port 8766 `
  --edf "C:\\data\\recording.edf" `
  --model "models\\ses-01-to-ses-02" `
  --events "C:\\data\\events.tsv"
```

Sunucu guvenlik amaciyla varsayilan olarak yalnizca `127.0.0.1` uzerinde acilir.

CLI replay'de smoothing penceresi `--smoothing-window 1|3|5` ile secilebilir;
varsayilan 3'tur. `1` smoothing gecikmesini sifirlar, `5` ise 50 ms karar
adiminda 100 ms ekler.

## Dogrulama

3 saniyelik gercek-zaman tempolu EDF smoke testinde 41 karar uretildi ve 41
satirin tamaminda `smoothed_prediction` yazildi. Varsayilan window=3 sonucu:

| Metrik | p50 | p95 | p99 | max |
|---|---:|---:|---:|---:|
| `end_to_end_ms` | 5.22 | 11.32 | 12.91 | 13.03 |
| `smoothing_delay_ms` | 50.00 | 50.00 | 50.00 | 50.00 |
| `smoothing_wall_delay_ms` | 49.90 | 61.13 | 64.65 | 65.91 |
| `total_latency_ms` | 56.78 | 66.35 | 71.01 | 71.06 |

Bu arayuz ve smoothing entegrasyonu model agirliklarini degistirmez.
