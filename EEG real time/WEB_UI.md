# Canli tahmin web arayuzu

Arayuz EDF kaydini gercek EEG hizinda oynatir ve WALK/STOP tahminlerini,
guven skorunu, EEG zamanini ve gecikme metriklerini tarayicida canli gosterir.

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

## Dogrulama

3 saniyelik gercek-zaman tempolu EDF smoke testinde HTTP arayuzu 200 dondu,
41 tahmin UI API'sine ulasti ve hata olusmadi. Bu arayuz modeli degistirmez;
son capraz-oturum balanced accuracy degeri `%73.77` olarak kalir.
