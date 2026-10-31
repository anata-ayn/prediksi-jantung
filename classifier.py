import json
import re
import time

import openai

NAMA_FITUR = {
    "Age": "Usia",
    "Sex": "Jenis kelamin",
    "ChestPainType": "Tipe nyeri dada",
    "RestingBP": "Tekanan darah istirahat",
    "Cholesterol": "Kolesterol",
    "FastingBS": "Gula darah puasa > 120",
    "RestingECG": "EKG istirahat",
    "MaxHR": "Detak jantung maksimum",
    "ExerciseAngina": "Angina akibat olahraga",
    "Oldpeak": "Oldpeak",
    "ST_Slope": "Kemiringan segmen ST",
}
TIDAK_TERUKUR = ("RestingBP", "Cholesterol")


def load_config(path="config.json"):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def format_data(data):
    bagian = []
    for kolom, nama in NAMA_FITUR.items():
        nilai = data[kolom]
        if kolom in TIDAK_TERUKUR and float(nilai) == 0:
            nilai = "tidak terukur"
        bagian.append(f"{nama}: {nilai}")
    return "; ".join(bagian)


def build_messages(cfg, data):
    msgs = [{"role": "system", "content": cfg["system_prompt"]}]
    for ex in cfg.get("few_shot", []):
        msgs.append({"role": "user", "content": format_data(ex["data"])})
        msgs.append({"role": "assistant", "content": ex.get("jawaban", ex["label"])})
    msgs.append({"role": "user", "content": format_data(data)})
    return msgs


def normal(teks):
    return re.sub(r"[^a-z]", "", teks.lower())


def parse_label(output, labels):
    if not output:
        return "invalid"
    peta = {normal(label): label for label in labels}
    pola = "|".join(r"\s+".join(re.escape(kata) for kata in label.split()) for label in labels)
    teks = output.strip()

    cocok = re.findall(rf"status\s*[:=]\s*\**\s*({pola})", teks, flags=re.I)
    if cocok:
        return peta[normal(cocok[-1])]

    if normal(teks) in peta:
        return peta[normal(teks)]

    semua = re.findall(pola, teks, flags=re.I)
    if semua:
        return peta[normal(semua[-1])]
    return "invalid"


def classify(client, cfg, data, max_retry=4):
    for percobaan in range(max_retry):
        try:
            resp = client.chat.completions.create(
                model=cfg["model"],
                temperature=cfg["temperature"],
                max_tokens=cfg["max_tokens"],
                messages=build_messages(cfg, data),
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )
            return parse_label(resp.choices[0].message.content, cfg["labels"])
        except openai.RateLimitError:
            time.sleep(15 * (percobaan + 1))
        except (openai.APIConnectionError, openai.APITimeoutError, openai.InternalServerError):
            time.sleep(5 * (percobaan + 1))
        except openai.APIError:
            return "invalid"
    return "invalid"
