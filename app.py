import json
import os
import time

import joblib
import openai
import pandas as pd
import streamlit as st

from classifier import classify, load_config

BATAS_LLM = 10
KOLOM_BARIS = [
    "Age", "Sex", "ChestPainType", "RestingBP", "Cholesterol", "FastingBS",
    "RestingECG", "MaxHR", "ExerciseAngina", "Oldpeak", "ST_Slope",
]

st.set_page_config(page_title="Prediksi Penyakit Jantung", page_icon="🫀", layout="wide")


@st.cache_resource
def muat_model():
    with open("model_info.json", encoding="utf-8") as f:
        info = json.load(f)
    return joblib.load("model_jantung.pkl"), info


@st.cache_resource
def muat_llm():
    cfg = load_config("config.json")
    try:
        kunci = st.secrets.get("NVIDIA_API_KEY")
    except Exception:
        kunci = None
    kunci = kunci or os.environ.get("NVIDIA_API_KEY")
    if not kunci:
        return cfg, None
    client = openai.OpenAI(base_url="https://integrate.api.nvidia.com/v1", api_key=kunci, timeout=60)
    return cfg, client


def prediksi_library(model, df):
    hasil = df[KOLOM_BARIS].copy()
    hasil["prediksi_library"] = model.predict(hasil)
    proba = model.predict_proba(hasil[KOLOM_BARIS])
    hasil["keyakinan"] = proba.max(axis=1).round(3)
    return hasil


def prediksi_llm(client, cfg, df, bar=None):
    out, waktu = [], []
    for i, (_, baris) in enumerate(df.iterrows()):
        mulai = time.time()
        out.append(classify(client, cfg, baris.to_dict()))
        waktu.append(round(time.time() - mulai, 1))
        if bar:
            bar.progress((i + 1) / len(df))
    hasil = df[KOLOM_BARIS].copy()
    hasil["prediksi_llm"] = out
    hasil["detik"] = waktu
    return hasil


def form_input(info):
    r = info["rentang_fitur"]
    nk = info["nilai_kategori"]
    k1, k2, k3 = st.columns(3)
    with k1:
        umur = st.number_input("Usia (tahun)", 1, 120, 54)
        sex = st.selectbox("Jenis kelamin (M = pria, F = wanita)", nk["Sex"])
        nyeri = st.selectbox("Tipe nyeri dada", nk["ChestPainType"])
        tekanan = st.number_input(
            "Tekanan darah istirahat (mm Hg), 0 jika tidak terukur", 0, 300, 130
        )
    with k2:
        kolesterol = st.number_input("Kolesterol (mg/dl), 0 jika tidak terukur", 0, 700, 240)
        gula = st.selectbox("Gula darah puasa > 120 mg/dl", [0, 1], format_func=lambda v: "Ya" if v else "Tidak")
        ekg = st.selectbox("EKG istirahat", nk["RestingECG"])
        hr = st.number_input("Detak jantung maksimum", 40, 250, 140)
    with k3:
        angina = st.selectbox("Angina akibat olahraga", nk["ExerciseAngina"], format_func=lambda v: "Ya" if v == "Y" else "Tidak")
        oldpeak = st.number_input("Oldpeak", -3.0, 10.0, 1.0, step=0.1)
        slope = st.selectbox("Kemiringan segmen ST", nk["ST_Slope"])
    return pd.DataFrame([{
        "Age": umur, "Sex": sex, "ChestPainType": nyeri, "RestingBP": tekanan,
        "Cholesterol": kolesterol, "FastingBS": gula, "RestingECG": ekg, "MaxHR": hr,
        "ExerciseAngina": angina, "Oldpeak": oldpeak, "ST_Slope": slope,
    }])


def cek_csv(df, info):
    kurang = [k for k in KOLOM_BARIS if k not in df.columns]
    if kurang:
        return f"Kolom berikut tidak ada di file: {', '.join(kurang)}"
    for k, nilai in info["nilai_kategori"].items():
        salah = sorted(set(df[k].astype(str)) - set(nilai))
        if salah:
            return f"Kolom {k} berisi nilai di luar {nilai}: {salah}"
    for k in info["fitur_numerik"] + info["fitur_nol_kosong"]:
        if pd.to_numeric(df[k], errors="coerce").isna().any():
            return f"Kolom {k} harus berisi angka dan tidak boleh kosong"
    return None


model, info = muat_model()
cfg, client = muat_llm()
ev = cfg.get("evaluasi", {})

st.title("Prediksi Penyakit Jantung")
st.caption(
    "Studi kasus Heart Failure Prediction (918 pasien). Hasil ini alat belajar, bukan diagnosis medis."
)

with st.sidebar:
    st.subheader("Model library")
    st.write(info["nama_model"])
    st.metric("F1 macro uji", info["metrik"]["f1_macro"])
    st.metric("Accuracy uji", info["metrik"]["accuracy"])
    st.subheader("Model LLM")
    st.write(cfg["model"])
    if ev:
        st.metric("F1 macro uji", ev.get("f1_macro"))
        st.metric("Accuracy uji", ev.get("accuracy"))
    if client is None:
        st.warning("NVIDIA_API_KEY belum diisi, tab LLM tidak aktif.")
    with open("contoh_data_baru.csv", "rb") as f:
        st.download_button("Unduh contoh data uji (CSV)", f, "contoh_data_baru.csv", "text/csv")

tab1, tab2, tab3 = st.tabs(["Input satu pasien", "Unggah CSV", "Tentang model"])

with tab1:
    with st.form("form_pasien"):
        satu = form_input(info)
        kirim = st.form_submit_button("Prediksi")
    if kirim:
        k1, k2 = st.columns(2)
        with k1:
            st.markdown("**Model library**")
            hasil = prediksi_library(model, satu)
            st.success(hasil.loc[0, "prediksi_library"])
            st.caption(f"Keyakinan {hasil.loc[0, 'keyakinan']:.1%}")
        with k2:
            st.markdown("**Model LLM**")
            if client is None:
                st.info("API key belum diisi.")
            else:
                with st.spinner("Menunggu jawaban LLM..."):
                    h = prediksi_llm(client, cfg, satu)
                st.success(h.loc[0, "prediksi_llm"])
                st.caption(f"Waktu {h.loc[0, 'detik']} detik")

with tab2:
    st.write(
        "Unggah CSV dengan kolom: " + ", ".join(KOLOM_BARIS)
        + ". Kolom lain seperti id_pasien boleh ada. Contoh file ada di sidebar."
    )
    berkas = st.file_uploader("File CSV", type="csv")
    if berkas is not None:
        try:
            data = pd.read_csv(berkas)
        except Exception as e:
            st.error(f"File tidak bisa dibaca: {e}")
            data = None
        if data is not None:
            galat = cek_csv(data, info)
            if galat:
                st.error(galat)
            else:
                st.write(f"{len(data)} baris terbaca")
                st.dataframe(data, use_container_width=True)
                pakai_llm = st.checkbox(
                    f"Sertakan prediksi LLM (maksimal {BATAS_LLM} baris pertama)",
                    value=client is not None, disabled=client is None,
                )
                if st.button("Jalankan prediksi"):
                    gabung = prediksi_library(model, data)
                    if "id_pasien" in data.columns:
                        gabung.insert(0, "id_pasien", data["id_pasien"].values)
                    if pakai_llm and client is not None:
                        bar = st.progress(0.0)
                        h = prediksi_llm(client, cfg, data.head(BATAS_LLM), bar)
                        gabung["prediksi_llm"] = None
                        gabung.loc[h.index, "prediksi_llm"] = h["prediksi_llm"].values
                    st.dataframe(gabung, use_container_width=True)
                    st.download_button(
                        "Unduh hasil (CSV)", gabung.to_csv(index=False).encode("utf-8"),
                        "hasil_prediksi.csv", "text/csv",
                    )

with tab3:
    m = info["metrik"]
    st.write(
        f"Model library: {info['nama_model']}, dilatih pada {m['n_train']} data dan diuji pada {m['n_test']} data. "
        f"F1 macro {m['f1_macro']} dibanding baseline {m['f1_baseline']}."
    )
    st.write(
        "Nilai 0 pada RestingBP dan Cholesterol dianggap tidak terukur dan diisi median data latih."
    )
    if ev:
        st.write(
            f"Model LLM: {cfg['model']}, F1 macro {ev.get('f1_macro')}, accuracy {ev.get('accuracy')}, "
            f"invalid {ev.get('invalid')}, rata-rata {ev.get('waktu_per_prediksi_detik')} detik per prediksi."
        )
    st.json(info["versi"])
