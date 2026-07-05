"""
joystick_output.py
==================
prediction -> state machine -> joystick çıktısı

Şu anki katman GÜVENLİ: gerçek bir joystick sürücüsüne (vJoy/ViGEm)
bağlanmıyor, sadece komutu terminale basıyor. İleride gerçek donanım
eklenecekse sadece JoystickOutput.send() içindeki iki dal doldurulur;
prediction_to_command() ve dışarıdaki API değişmez.
"""


def prediction_to_command(prediction: int, confidence: float, threshold: float = 0.6) -> str:
    """
    Ham (prediction, confidence) çiftini WALK / STOP / IDLE komutuna çevirir.

    - confidence eşik altındaysa -> IDLE (kararsız/gürültülü tahmine güvenme)
    - prediction == 1            -> WALK
    - prediction == 0            -> STOP
    """
    if confidence < threshold:
        return "IDLE"

    if prediction == 1:
        return "WALK"

    return "STOP"


class JoystickOutput:
    """
    Şimdilik güvenli çıktı katmanı.
    Gerçek vJoy/ViGEm eklenene kadar komudu terminale basar.

    last_command ile state değişimi takip edilir: aynı komut arka arkaya
    geldiğinde (örn. WALK sırasında her pencere WALK tahmini üretirken)
    tekrar tekrar yazdırma/axis güncellemesi yapılmaz - sadece komut
    DEĞİŞTİĞİNDE (STOP->WALK veya WALK->STOP) bir eylem tetiklenir.
    Sürekli joystick ekseni beslemesi gerekiyorsa (her frame'de forward
    axis'i canlı tutmak gibi) bu davranış send() içinde ayarlanabilir;
    şu an tasarım "state değişiminde tetikle" mantığıyla çalışıyor.
    """

    def __init__(self):
        self.last_command = None

    def send(self, command: str):
        if command == self.last_command:
            return

        self.last_command = command

        if command == "WALK":
            print("[JOYSTICK] FORWARD")
            # Buraya sonra vJoy axis forward gelecek

        elif command == "STOP":
            print("[JOYSTICK] STOP")
            # Buraya sonra joystick neutral gelecek

        else:
            print("[JOYSTICK] IDLE")


# ============================================================================
# Basit kullanım (tek pencere)
# ============================================================================
if __name__ == "__main__":
    joystick = JoystickOutput()

    # Örnek: pipe.predict([current_window]) -> (pred, conf) döndüğünü varsayıyoruz
    # pred, conf = pipe.predict([current_window])
    # command = prediction_to_command(int(pred[0]), float(conf[0]), threshold=0.6)
    # joystick.send(command)

    # Manuel test:
    for pred, conf in [(1, 0.9), (1, 0.85), (0, 0.7), (0, 0.4)]:
        command = prediction_to_command(pred, conf, threshold=0.6)
        joystick.send(command)
