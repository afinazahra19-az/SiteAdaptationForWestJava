# -*- coding: utf-8 -*-
"""
Created on Thu Nov  6 08:28:24 2025

@author: user
"""
# -*- coding: utf-8 -*-
"""
Processes all .nc files in folder_nc using:
  - Semi-Empirical (SE)
  - Regression site adaptation
  - Deep Learning (LSTM) site adaptation
Outputs: single combined CSV per file.
"""

import os
import glob
import numpy as np
import pandas as pd
import netCDF4 as nc
from datetime import datetime, timedelta

# ==========================================================
# CONFIG
# ==========================================================
folder_nc = "H:/0_sinergi/nc"
file_elevasi = "H:/0_sinergi/ELEVASI_3_ZONATS.xlsx"
output_base = "H:/0_sinergi/OUTPUT"
output_combined = os.path.join(output_base, "Output_Combined_nonrl")
os.makedirs(output_combined, exist_ok=True)

# ==========================================================
# DEEP LEARNING MODEL CONFIG
# ==========================================================
models_config = {
    "A": {
        "range": (0, 500),
        "model": r"H:/0_sinergi/DeepLearningModels/lstm02-HAURGEULIS_model.h5",
        "xscaler": r"H:/0_sinergi/DeepLearningModels/scaler02-HAURGEULIS_X.pkl",
        "yscaler": r"H:/0_sinergi/DeepLearningModels/scaler02-HAURGEULIS_y.pkl"
    },
    "B": {
        "range": (500, 1000),
        "model": r"H:/0_sinergi/DeepLearningModels/lstm02-CIKASUNGKA_model.h5",
        "xscaler": r"H:/0_sinergi/DeepLearningModels/scaler02-CIKASUNGKA_X.pkl",
        "yscaler": r"H:/0_sinergi/DeepLearningModels/scaler02-CIKASUNGKA_y.pkl"
    },
    "C": {
        "range": (1000, float('inf')),
        "model": r"H:/0_sinergi/DeepLearningModels/lstm02-PATUHA_model.h5",
        "xscaler": r"H:/0_sinergi/DeepLearningModels/scaler02-PATUHA_X.pkl",
        "yscaler": r"H:/0_sinergi/DeepLearningModels/scaler02-PATUHA_y.pkl"
    }
}

ENABLE_DL = True

# ==========================================================
# HELPER FUNCTIONS
# ==========================================================
def extract_datetime_from_filename(filename):
    parts = filename.split("_")
    yyyymmddhhmm = parts[-1].replace(".nc", "")
    dt_utc = datetime.strptime(yyyymmddhhmm, "%Y%m%d%H%M")
    dt_wib = dt_utc + timedelta(hours=7)
    doy = dt_wib.timetuple().tm_yday
    return dt_utc, dt_wib, doy

def lincol_from_latlon_geos(Resolution, Latitude, Longitude):
    import numpy as np
    degtorad = np.pi / 180.0
    sub_lon = 128.2 * degtorad
    Latitude, Longitude = Latitude * degtorad, Longitude * degtorad
    if Resolution == 0.5:
        COFF, CFAC = 11000.5, 8.170135561335742e7
        LOFF, LFAC = 11000.5, 8.170135561335742e7
    else:
        COFF, CFAC = 5500.5, 4.0850677806678705e7
        LOFF, LFAC = 5500.5, 4.0850677806678705e7
    c_lat = np.arctan(0.993305616 * np.tan(Latitude))
    RL = 6356.7523 / np.sqrt(1.0 - 0.00669438444 * np.cos(c_lat)**2.0)
    R1 = 42164.0 - RL * np.cos(c_lat) * np.cos(Longitude - sub_lon)
    R2 = -RL * np.cos(c_lat) * np.sin(Longitude - sub_lon)
    R3 = RL * np.sin(c_lat)
    Rn = np.sqrt(R1**2.0 + R2**2.0 + R3**2.0)
    x = np.arctan(-R2 / R1) / degtorad
    y = np.arcsin(-R3 / Rn) / degtorad
    ncol = COFF + (x * 2.0**(-16) * CFAC)
    nlin = LOFF + (y * 2.0**(-16) * LFAC)
    return int(np.round(nlin)), int(np.round(ncol))

def calculate_sza(lat, lon, doy, hour_wib):
    delta = np.radians(23.45 * np.sin(np.radians((360/365)*(doy+284))))
    phi = np.radians(lat)
    h = np.radians(15*(hour_wib-12))
    cos_sza = np.sin(phi)*np.sin(delta) + np.cos(phi)*np.cos(delta)*np.cos(h)
    return np.degrees(np.arccos(np.clip(cos_sza, -1, 1)))

def fnGHIc(sun_dist, SZA):
    return 0.74 * 1366 * sun_dist * np.cos(np.radians(SZA)) ** 1.11

def fnGHIConversion(CI, GHIc):
    Ktm = (2.8*(CI**5)) - (6.2*(CI**4)) + (6.22*(CI**3)) - (2.63*(CI**2)) - (0.59*CI) + 1
    GHI = Ktm * GHIc * ((0.00009*Ktm*GHIc) + 0.9)
    return GHI, Ktm

# ==========================================================
# LOAD ELEVASI & STASIUN
# ==========================================================
df_pts = pd.read_excel(file_elevasi)
df_pts.rename(columns=lambda x: x.strip().capitalize(), inplace=True)
print(f"Jumlah titik: {len(df_pts)}")

stations = {
    "ARJAWINANGUN": (-6.657093112, 108.4026088),
    "BANJARSARI": (-7.49796, 108.61577),
    "BOJONGMANGU": (-6.43199, 107.19197),
    "CIANJUR": (-6.773122879, 107.1155292),
    "CIAWI": (-6.657909299, 106.848),
    "CIKARANG": (-6.266, 107.156),
    "CIKASUNGKA": (-6.991210801, 107.824115),
    "CIKUMPAY": (-6.504432536, 107.4937473),
    "CIPEUNDEUY": (-6.868198443, 107.4741616),
    "HAURGEULIS": (-6.436708853, 107.9460629),
    "INDRAMAYU1": (-6.4904, 107.92409),
    "INDRAMAYU2": (-6.34439, 108.34113),
    "JATINANGOR": (-6.92924, 107.76995),
    "KADIPATEN": (-6.79126964, 108.1747922),
    "KARANGNUNGGAL": (-7.63599485, 108.1273794),
    "PATUHA": (-7.181892477, 107.4205164),
    "PELABUHANRATU_ISM": (-6.987111705, 106.5551164),
    "RENGASDENGKLOK": (-6.162700589, 107.3115359),
    "SUMEDANG": (-6.82425, 107.84493),
    "TASIKMALAYA": (-7.352045139, 108.2089832),
    "UJUNGGENTENG": (-7.32476, 106.41298)
}

def find_nearest_station(lat, lon):
    min_dist = float("inf")
    nearest = None
    for name, (st_lat, st_lon) in stations.items():
        dist = np.sqrt((lat - st_lat)**2 + (lon - st_lon)**2)
        if dist < min_dist:
            min_dist = dist
            nearest = name
    return nearest

def GHI_ARJAWINANGUN(GHI_SemiEmpiris, kc, Kt, AM, Solar_Elevation):
    return -36.62217 + (1.11448 * GHI_SemiEmpiris) + (507.53203 * kc) + (-992.44898 * Kt) + (16.49479 * AM) + (1.89513 * Solar_Elevation)

def GHI_BANJARSARI(GHI_SemiEmpiris, kc, Kt, AM, Solar_Elevation):
    return 80.10494 + (1.56440 * GHI_SemiEmpiris) + (1227.80024 * kc) + (-2369.64679 * Kt) + (-6.22983 * AM) + (-2.10678 * Solar_Elevation)

def GHI_BOJONGMANGU(GHI_SemiEmpiris, kc, Kt, AM, Solar_Elevation):
    return 158.78511 + (1.40357 * GHI_SemiEmpiris) + (1591.11634 * kc) + (-2976.99454 * Kt) + (-19.49990 * AM) + (-2.16582 * Solar_Elevation)

def GHI_CIANJUR(GHI_SemiEmpiris, kc, Kt, AM, Solar_Elevation):
    return 97.47555 + (1.48074 * GHI_SemiEmpiris) + (1275.02946 * kc) + (-2564.48109 * Kt) + (7.44655 * AM) + (-0.02656 * Solar_Elevation)

def GHI_CIAWI(GHI_SemiEmpiris, kc, Kt, AM, Solar_Elevation):
    return -8.45564 + (1.43483 * GHI_SemiEmpiris) + (863.74195 * kc) + (-1933.46679 * Kt) + (36.27411 * AM) + (0.78644 * Solar_Elevation)

def GHI_CIKARANG(GHI_SemiEmpiris, kc=None, Kt=None, AM=None, Solar_Elevation=None):
    return 23.85261 + (0.77137 * GHI_SemiEmpiris)

def GHI_CIKASUNGKA(GHI_SemiEmpiris, kc, Kt, AM, Solar_Elevation):
    return -70.59884 + (1.20645 * GHI_SemiEmpiris) + (959.54669 * kc) + (-1801.29998 * Kt) + (18.98368 * AM) + (3.29493 * Solar_Elevation)

def GHI_CIKUMPAY(GHI_SemiEmpiris, kc, Kt, AM, Solar_Elevation):
    return 29.63355 + (1.13110 * GHI_SemiEmpiris) + (431.19181 * kc) + (-1040.26407 * Kt) + (13.33148 * AM) + (1.30169 * Solar_Elevation)

def GHI_CIPEUNDEUY(GHI_SemiEmpiris, kc, Kt, AM, Solar_Elevation):
    return 84.29538 + (1.23106 * GHI_SemiEmpiris) + (-71.85986 * kc) + (-489.00827 * Kt) + (27.84417 * AM) + (0.30807 * Solar_Elevation)

def GHI_HAURGEULIS(GHI_SemiEmpiris, kc, Kt, AM, Solar_Elevation):
    return -6.33312 + (1.10790 * GHI_SemiEmpiris) + (237.95945 * kc) + (-621.78296 * Kt) + (15.94115 * AM) + (1.46857 * Solar_Elevation)

def GHI_INDRAMAYU1(GHI_SemiEmpiris, kc, Kt, AM, Solar_Elevation):
    return 14.18274 + (0.87378 * GHI_SemiEmpiris) + (-486.57276 * kc) + (667.90400 * Kt) + (-1.01923 * AM) + (0.63945 * Solar_Elevation)

def GHI_INDRAMAYU2(GHI_SemiEmpiris, kc, Kt, AM, Solar_Elevation):
    return 158.97302 + (1.29339 * GHI_SemiEmpiris) + (848.21390 * kc) + (-1619.52915 * Kt) + (-20.66651 * AM) + (-2.43843 * Solar_Elevation)

def GHI_JATINANGOR(GHI_SemiEmpiris, kc, Kt, AM, Solar_Elevation):
    return -45.65611 + (1.42746 * GHI_SemiEmpiris) + (-312.74756 * kc) + (-95.30911 * Kt) + (46.61543 * AM) + (-0.50559 * Solar_Elevation)

def GHI_KADIPATEN(GHI_SemiEmpiris, kc, Kt, AM, Solar_Elevation):
    return 25.33371 + (1.30889 * GHI_SemiEmpiris) + (1573.85970 * kc) + (-2826.81325 * Kt) + (9.07592 * AM) + (2.00014 * Solar_Elevation)

def GHI_KARANGNUNGGAL(GHI_SemiEmpiris, kc=None, Kt=None, AM=None, Solar_Elevation=None):
    return -21.33460 + (0.92968 * GHI_SemiEmpiris)

def GHI_PATUHA(GHI_SemiEmpiris, kc=None, Kt=None, AM=None, Solar_Elevation=None):
    return -66.68238 + (0.97993 * GHI_SemiEmpiris)

def GHI_PELABUHANRATU_ISM(GHI_SemiEmpiris, kc, Kt, AM, Solar_Elevation):
    return 73.37474 + (1.38020 * GHI_SemiEmpiris) + (-354.43638 * kc) + (-100.18763 * Kt) + (31.06925 * AM) + (-0.59703 * Solar_Elevation)

def GHI_RENGASDENGKLOK(GHI_SemiEmpiris, kc=None, Kt=None, AM=None, Solar_Elevation=None):
    return 13.44871 + (0.98133 * GHI_SemiEmpiris)

def GHI_SUMEDANG(GHI_SemiEmpiris, kc, Kt, AM, Solar_Elevation):
    return 34.66801 + (0.88245 * GHI_SemiEmpiris) + (-147.94499 * kc) + (287.85626 * Kt) + (-22.34256 * AM) + (-0.95980 * Solar_Elevation)

def GHI_TASIKMALAYA(GHI_SemiEmpiris, kc, Kt, AM, Solar_Elevation):
    return -92.50570 + (1.18411 * GHI_SemiEmpiris) + (1.19899 * kc) + (-386.46355 * Kt) + (43.62204 * AM) + (1.36120 * Solar_Elevation)

def GHI_UJUNGGENTENG(GHI_SemiEmpiris, kc, Kt, AM, Solar_Elevation):
    return -0.57108 + (0.54225 * GHI_SemiEmpiris) + (-2835.74630 * kc) + (4567.23579 * Kt) + (13.39723 * AM) + (-0.39302 * Solar_Elevation)

def apply_siteadapt_regression(row):
    zona = str(row.get("Zona", "")).upper()
    if zona == "BIASA":
        return row["GHI_SemiEmpiris"]
    nearest_station = find_nearest_station(row["Latitude"], row["Longitude"])
    func_name = f"GHI_{nearest_station.upper().replace(' ', '_')}"
    if func_name in globals():
        func = globals()[func_name]
        return func(row["GHI_SemiEmpiris"], row["kc"], row["Kt"], row["AM"], row["Solar_Elevation"])
    else:
        return np.nan

# ==========================================================
# LOAD DEEP LEARNING MODELS
# ==========================================================
dl_available = False
dl_models = {}
if ENABLE_DL:
    try:
        import joblib
        from tensorflow.keras.models import load_model
        for key, cfg in models_config.items():
            mdl = load_model(cfg["model"], compile=False)
            xsc = joblib.load(cfg["xscaler"])
            ysc = joblib.load(cfg["yscaler"])
            dl_models[key] = {"model": mdl, "xscaler": xsc, "yscaler": ysc, "range": cfg["range"]}
        dl_available = True
        print(f"[DL] Loaded {len(dl_models)} models.")
    except Exception as e:
        print(f"[DL] Model load failed: {e}")
        dl_available = False

def create_sequences(X, time_steps=24):
    return np.array([X[i:i+time_steps] for i in range(len(X)-time_steps)])

# ==========================================================
# MAIN OFFLINE PROCESSING
# ==========================================================
nc_files = sorted(glob.glob(os.path.join(folder_nc, "**", "*.nc"), recursive=True))
print(f"Ditemukan {len(nc_files)} file .nc untuk diproses di {folder_nc}")

for f in nc_files:
    print(f"\n=== Memproses {os.path.basename(f)} ===")
    try:
        dt_utc, dt_wib, doy = extract_datetime_from_filename(os.path.basename(f))
        hour_wib = dt_wib.hour
        data_records = []

        with nc.Dataset(f) as ds:
            if "image_pixel_values" not in ds.variables:
                print(f"[SKIP] Tidak ada variabel image_pixel_values di {f}")
                continue
            img = ds["image_pixel_values"][:]
            for _, r in df_pts.iterrows():
                lin, col = lincol_from_latlon_geos(0.5, r["Latitude"], r["Longitude"])
                pixel = float(img[lin, col]) if (0 <= lin < img.shape[0] and 0 <= col < img.shape[1]) else np.nan
                sza = calculate_sza(r["Latitude"], r["Longitude"], doy, hour_wib)
                sun_dist = 1.00011 + 0.034221 * np.cos((2 * np.pi) * (doy - 1) / 365)
                elev = r.get("Elevasi", np.nan)
                alt_angle = 90 - sza
                AM = (1 - (elev / 10000)) / (np.cos(np.radians(sza)) + 0.50572 * (96.07995 - sza) ** -1.6364)
                am_effect = 2.283 * (alt_angle ** (-0.26)) * np.exp(0.004 * alt_angle)
                CCp = pixel * AM * sun_dist / am_effect
                zona = str(r["Zona"]).upper() if "Zona" in r else None

                data_records.append({
                    "Datetime": dt_utc, "Datetime_WIB": dt_wib, "DOY": doy,
                    "Hour": hour_wib, "Latitude": r["Latitude"], "Longitude": r["Longitude"],
                    "Elevasi": elev, "Zona": zona, "Pixel": pixel, "SZA": sza,
                    "Sun_Distance": sun_dist, "AM": AM, "CCp": CCp
                })

        df_raw = pd.DataFrame(data_records)
        df_valid = df_raw[df_raw["SZA"] < 80].copy()
        if df_valid.empty:
            print("⚠️ Tidak ada titik valid (SZA < 80)")
            continue

        Q1, Q3 = np.percentile(df_valid["CCp"].dropna(), [25, 75])
        IQR = Q3 - Q1
        UB, LB = Q3 + 1.5*IQR, max(Q1 - 1.5*IQR, df_valid["CCp"].min())

        rows = []
        for _, r in df_valid.iterrows():
            CI = np.clip((r["CCp"] - LB) / (UB - LB), 0, 1)
            GHIc = fnGHIc(r["Sun_Distance"], r["SZA"])
            GHI, Ktm = fnGHIConversion(CI, GHIc)
            if GHI > 0:
                rows.append({**r, "CI": CI, "GHIc": GHIc, "GHI_SemiEmpiris": GHI, "Ktm": Ktm})

        df_out = pd.DataFrame(rows)
        if df_out.empty:
            print("⚠️ Tidak ada output GHI > 0")
            continue

        # Regression
        df_out["Solar_Elevation"] = 90 - df_out["SZA"]
        df_out["E0n"] = 1367 * (1 + 0.033 * np.cos(2*np.pi*df_out["DOY"]/365))
        df_out["TOA"] = df_out["E0n"] * np.cos(np.radians(df_out["SZA"]))
        df_out["Kt"] = df_out["GHI_SemiEmpiris"] / df_out["TOA"]
        df_out["kc"] = df_out["GHI_SemiEmpiris"] / df_out["GHIc"]
        df_out["Nearest_Station"] = df_out.apply(lambda r: find_nearest_station(r["Latitude"], r["Longitude"]), axis=1)
        df_out["GHI_SiteAdapt"] = df_out.apply(apply_siteadapt_regression, axis=1)

        # Deep Learning
        df_out["GHI_DeepLearning"] = np.nan
        if ENABLE_DL and dl_available:
            for key, mdl in dl_models.items():
                low, high = mdl["range"]
                sel_idx = df_out[(df_out["Elevasi"] >= low) & (df_out["Elevasi"] < high)].index
                if len(sel_idx) == 0:
                    continue
                df_sub = df_out.loc[sel_idx, ['Solar_Elevation', 'AM', 'Kt', 'kc', 'GHI_SemiEmpiris']].dropna()
                if len(df_sub) < 24:
                    continue
                X_scaled = mdl["xscaler"].transform(df_sub)
                X_seq = create_sequences(X_scaled)
                if X_seq.size == 0:
                    continue
                y_scaled_pred = mdl["model"].predict(X_seq, verbose=0)
                y_pred = mdl["yscaler"].inverse_transform(y_scaled_pred).flatten()
                idx_list = list(df_sub.index)
                pred_idx = idx_list[24:24+len(y_pred)]
                minlen = min(len(pred_idx), len(y_pred))
                df_out.loc[pred_idx[:minlen], "GHI_DeepLearning"] = y_pred[:minlen]

      
        timestamp_str = dt_wib.strftime("%Y%m%d%H%M")
        combined_csv = os.path.join(output_combined, f"GHI_COMBINED_{timestamp_str}.csv")
        cols_csv = ["Datetime_WIB", "Latitude", "Longitude", "Zona", "Elevasi", "GHIc", "kc", "Kt", "AM",
                    "Solar_Elevation", "CI", "GHI_SemiEmpiris", "GHI_SiteAdapt", "GHI_DeepLearning", "Nearest_Station"]
        for c in cols_csv:
            if c not in df_out.columns:
                df_out[c] = np.nan
        df_out[cols_csv].to_csv(combined_csv, index=False)
        print(f"✅ Saved: {combined_csv}")

    except Exception as e:
        print(f"❌ Gagal proses {f}: {e}")
