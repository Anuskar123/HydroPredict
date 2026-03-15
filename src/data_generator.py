"""
Generates realistic synthetic weather + hydrology data modeled on Nepal's climate.

Patterns encoded:
- Monsoon season (June-September): heavy rainfall, high river flow
- Winter (December-February): low rainfall, low flow, cold temperatures
- Pre-monsoon (March-May): rising temperatures, sporadic rain
- Post-monsoon (October-November): declining rain, moderate flow

Districts modeled: Kaski (Pokhara), Kathmandu, Chitwan, Sunsari, Kalikot
Each has distinct elevation-based climate profiles.
"""

import numpy as np
import pandas as pd
from pathlib import Path

DISTRICTS = {
    # Koshi (1)
    "Taplejung": {"elevation": 1800, "base_temp": 15, "rain_mult": 1.5, "river": "Tamor / Kabeli"},
    "Sankhuwasabha": {"elevation": 1300, "base_temp": 16, "rain_mult": 1.5, "river": "Arun / Barun"},
    "Solukhumbu": {"elevation": 2500, "base_temp": 10, "rain_mult": 1.3, "river": "Dudhkoshi / Khimti"},
    "Bhojpur": {"elevation": 1600, "base_temp": 17, "rain_mult": 1.3, "river": "Piluwa Khola"},
    "Dhankuta": {"elevation": 1200, "base_temp": 18, "rain_mult": 1.3, "river": "Tamor / Sawa Khola"},
    "Terhathum": {"elevation": 1500, "base_temp": 17, "rain_mult": 1.4, "river": "Mewa Khola"},
    "Panchthar": {"elevation": 1400, "base_temp": 17, "rain_mult": 1.5, "river": "Maikhola / Tamor"},
    "Ilam": {"elevation": 1200, "base_temp": 18, "rain_mult": 1.8, "river": "Jogmai Khola"},
    "Jhapa": {"elevation": 100, "base_temp": 26, "rain_mult": 1.6, "river": "Kankai / Mechi"},
    "Morang": {"elevation": 100, "base_temp": 26, "rain_mult": 1.4, "river": "Koshi"},
    "Sunsari": {"elevation": 100, "base_temp": 26, "rain_mult": 1.4, "river": "Saptakoshi"},
    "Udayapur": {"elevation": 500, "base_temp": 23, "rain_mult": 1.3, "river": "Triyuga"},
    "Khotang": {"elevation": 1500, "base_temp": 17, "rain_mult": 1.3, "river": "Solu / Likhu"},
    "Okhaldhunga": {"elevation": 1600, "base_temp": 16, "rain_mult": 1.2, "river": "Molung Khola"},
    # Madhesh (2)
    "Saptari": {"elevation": 100, "base_temp": 27, "rain_mult": 1.2, "river": "Koshi / Triyuga"},
    "Siraha": {"elevation": 100, "base_temp": 27, "rain_mult": 1.2, "river": "Khando / Kamala"},
    "Dhanusha": {"elevation": 100, "base_temp": 27, "rain_mult": 1.1, "river": "Kamala"},
    "Mahottari": {"elevation": 100, "base_temp": 27, "rain_mult": 1.1, "river": "Bagmati"},
    "Sarlahi": {"elevation": 100, "base_temp": 27, "rain_mult": 1.1, "river": "Bagmati / Lalbakeya"},
    "Rautahat": {"elevation": 100, "base_temp": 27, "rain_mult": 1.1, "river": "Bagmati"},
    "Bara": {"elevation": 100, "base_temp": 27, "rain_mult": 1.1, "river": "Bagmati / Pathraiya"},
    "Parsa": {"elevation": 100, "base_temp": 27, "rain_mult": 1.1, "river": "Pathraiya / Banganga"},
    # Bagmati (3)
    "Sindhuli": {"elevation": 700, "base_temp": 22, "rain_mult": 1.3, "river": "Sunkoshi"},
    "Ramechhap": {"elevation": 1400, "base_temp": 18, "rain_mult": 1.3, "river": "Tamakoshi"},
    "Dolakha": {"elevation": 1600, "base_temp": 17, "rain_mult": 1.3, "river": "Tamakoshi / Bhotekoshi"},
    "Sindhupalchok": {"elevation": 1400, "base_temp": 17, "rain_mult": 1.4, "river": "Bhotekoshi / Indrawati"},
    "Rasuwa": {"elevation": 2000, "base_temp": 14, "rain_mult": 1.3, "river": "Trishuli / Langtang"},
    "Nuwakot": {"elevation": 900, "base_temp": 20, "rain_mult": 1.3, "river": "Trishuli"},
    "Dhading": {"elevation": 1000, "base_temp": 20, "rain_mult": 1.3, "river": "Trishuli / Budhigandaki"},
    "Kathmandu": {"elevation": 1400, "base_temp": 18, "rain_mult": 1.0, "river": "Bagmati / Bishnumati"},
    "Bhaktapur": {"elevation": 1400, "base_temp": 18, "rain_mult": 1.0, "river": "Hanumante Khola"},
    "Lalitpur": {"elevation": 1400, "base_temp": 18, "rain_mult": 1.0, "river": "Bagmati / Nakhu Khola"},
    "Kavrepalanchok": {"elevation": 1600, "base_temp": 17, "rain_mult": 1.2, "river": "Roshi Khola / Sunkoshi"},
    "Makwanpur": {"elevation": 1000, "base_temp": 21, "rain_mult": 1.2, "river": "Kulekhani"},
    "Chitwan": {"elevation": 200, "base_temp": 25, "rain_mult": 1.3, "river": "Narayani / Rapti"},
    # Gandaki (4)
    "Gorkha": {"elevation": 1100, "base_temp": 20, "rain_mult": 1.3, "river": "Budhigandaki"},
    "Manang": {"elevation": 3500, "base_temp": 8, "rain_mult": 0.8, "river": "Marsyangdi"},
    "Mustang": {"elevation": 2800, "base_temp": 10, "rain_mult": 0.7, "river": "Kali Gandaki"},
    "Myagdi": {"elevation": 900, "base_temp": 19, "rain_mult": 1.4, "river": "Myagdi Khola / Kali Gandaki"},
    "Kaski": {"elevation": 827, "base_temp": 20, "rain_mult": 1.6, "river": "Seti / Modi Khola"},
    "Lamjung": {"elevation": 1300, "base_temp": 18, "rain_mult": 1.4, "river": "Marsyangdi"},
    "Tanahun": {"elevation": 900, "base_temp": 21, "rain_mult": 1.4, "river": "Seti / Marsyangdi"},
    "Syangja": {"elevation": 1100, "base_temp": 20, "rain_mult": 1.4, "river": "Andhi Khola"},
    "Palpa": {"elevation": 1200, "base_temp": 20, "rain_mult": 1.3, "river": "Kali Gandaki / Ridi"},
    "Nawalparasi East": {"elevation": 200, "base_temp": 25, "rain_mult": 1.3, "river": "Narayani"},
    "Baglung": {"elevation": 1020, "base_temp": 19, "rain_mult": 1.4, "river": "Kali Gandaki"},
    "Parbat": {"elevation": 1200, "base_temp": 19, "rain_mult": 1.4, "river": "Modi Khola"},
    "Nawalpur": {"elevation": 200, "base_temp": 25, "rain_mult": 1.3, "river": "Narayani"},
    # Lumbini (5)
    "Rukum East": {"elevation": 1500, "base_temp": 18, "rain_mult": 1.2, "river": "Bheri / Jhariganga"},
    "Rolpa": {"elevation": 1600, "base_temp": 18, "rain_mult": 1.2, "river": "Madi Khola / Lungri"},
    "Pyuthan": {"elevation": 1300, "base_temp": 19, "rain_mult": 1.2, "river": "Lungri / West Rapti"},
    "Gulmi": {"elevation": 1500, "base_temp": 18, "rain_mult": 1.2, "river": "Kali Gandaki / Ridi"},
    "Arghakhanchi": {"elevation": 1200, "base_temp": 20, "rain_mult": 1.2, "river": "Arghau / West Rapti"},
    "Kapilvastu": {"elevation": 100, "base_temp": 27, "rain_mult": 1.1, "river": "Banganga / Rohini"},
    "Rupandehi": {"elevation": 100, "base_temp": 27, "rain_mult": 1.1, "river": "Tinau / Rohini"},
    "Nawalparasi West": {"elevation": 200, "base_temp": 25, "rain_mult": 1.3, "river": "Narayani"},
    "Dang": {"elevation": 700, "base_temp": 24, "rain_mult": 1.1, "river": "Rapti / Babai"},
    "Banke": {"elevation": 150, "base_temp": 26, "rain_mult": 1.0, "river": "West Rapti"},
    "Bardiya": {"elevation": 150, "base_temp": 26, "rain_mult": 1.0, "river": "Karnali / Babai"},
    # Karnali (6)
    "Dolpa": {"elevation": 2600, "base_temp": 9, "rain_mult": 0.7, "river": "Thuli Bheri"},
    "Mugu": {"elevation": 3000, "base_temp": 8, "rain_mult": 0.7, "river": "Mugu Karnali"},
    "Humla": {"elevation": 2900, "base_temp": 8, "rain_mult": 0.7, "river": "Humla Karnali"},
    "Jumla": {"elevation": 2400, "base_temp": 10, "rain_mult": 0.8, "river": "Tila / Karnali"},
    "Kalikot": {"elevation": 1220, "base_temp": 16, "rain_mult": 0.8, "river": "Karnali / Neruwa"},
    "Dailekh": {"elevation": 1500, "base_temp": 17, "rain_mult": 0.9, "river": "Kavre Khola"},
    "Jajarkot": {"elevation": 1500, "base_temp": 17, "rain_mult": 0.9, "river": "Bheri / Sani Bheri"},
    "Rukum West": {"elevation": 1800, "base_temp": 16, "rain_mult": 0.9, "river": "Bheri / Lungri"},
    "Salyan": {"elevation": 1500, "base_temp": 18, "rain_mult": 0.9, "river": "Bheri / Bangad"},
    "Surkhet": {"elevation": 720, "base_temp": 22, "rain_mult": 1.0, "river": "Karnali / Bheri"},
    # Sudurpashchim (7)
    "Bajura": {"elevation": 2100, "base_temp": 14, "rain_mult": 1.0, "river": "Seti / Budhiganga"},
    "Bajhang": {"elevation": 1700, "base_temp": 15, "rain_mult": 1.0, "river": "Seti"},
    "Achham": {"elevation": 1200, "base_temp": 18, "rain_mult": 1.0, "river": "Budhiganga / Khadganga"},
    "Doti": {"elevation": 1400, "base_temp": 17, "rain_mult": 1.0, "river": "Seti Nadi"},
    "Kailali": {"elevation": 200, "base_temp": 27, "rain_mult": 1.0, "river": "Karnali / Mahakali"},
    "Kanchanpur": {"elevation": 200, "base_temp": 27, "rain_mult": 1.0, "river": "Mahakali / Sarda"},
    "Dadeldhura": {"elevation": 1500, "base_temp": 17, "rain_mult": 1.0, "river": "Dadeldhura Khola"},
    "Baitadi": {"elevation": 1500, "base_temp": 17, "rain_mult": 1.0, "river": "Mahakali / Seti"},
    "Darchula": {"elevation": 1800, "base_temp": 15, "rain_mult": 1.0, "river": "Mahakali / Chameliya"},
}

# Hydropower plants mapped to river systems (capacity in MW).
# Where MW is missing (— in the table), we assign a small generic
# capacity so the visualization still works.
HYDRO_PLANTS = {
    # Koshi (1)
    "Tamor / Kabeli": {"name": "Kabeli-A HEP", "capacity_mw": 37.6},
    "Arun / Barun": {"name": "Arun-III HEP", "capacity_mw": 900.0},
    "Dudhkoshi / Khimti": {"name": "Khimti-I HEP", "capacity_mw": 60.0},
    "Piluwa Khola": {"name": "Piluwa Khola HEP", "capacity_mw": 3.0},
    "Tamor / Sawa Khola": {"name": "Sawa Khola HEP", "capacity_mw": 24.0},
    "Mewa Khola": {"name": "Mewa Khola HEP", "capacity_mw": 10.0},
    "Maikhola / Tamor": {"name": "Mai Cascade HEP", "capacity_mw": 22.0},
    "Jogmai Khola": {"name": "Jogmai HEP", "capacity_mw": 5.4},
    "Kankai / Mechi": {"name": "Kankai HEP", "capacity_mw": 4.2},
    "Koshi": {"name": "Chatara (Koshi) HEP", "capacity_mw": 20.0},
    "Saptakoshi": {"name": "Saptakoshi Hydro", "capacity_mw": 20.0},
    "Triyuga": {"name": "Triyuga HEP", "capacity_mw": 7.1},
    "Solu / Likhu": {"name": "Solu Corridor HEP", "capacity_mw": 23.5},
    "Molung Khola": {"name": "Molung Khola HEP", "capacity_mw": 45.0},
    # Madhesh (2) – mostly run-of-river or small / planned plants
    "Koshi / Triyuga": {"name": "Koshi–Triyuga Hydro", "capacity_mw": 10.0},
    "Khando / Kamala": {"name": "Khando–Kamala Hydro", "capacity_mw": 10.0},
    "Kamala": {"name": "Kamala Hydro", "capacity_mw": 10.0},
    "Bagmati": {"name": "Bagmati Hydro", "capacity_mw": 10.0},
    "Bagmati / Lalbakeya": {"name": "Bagmati–Lalbakeya Hydro", "capacity_mw": 10.0},
    "Bagmati / Pathraiya": {"name": "Bagmati–Pathraiya Hydro", "capacity_mw": 10.0},
    "Pathraiya / Banganga": {"name": "Pathraiya–Banganga Hydro", "capacity_mw": 10.0},
    # Bagmati (3)
    "Sunkoshi": {"name": "Sunkoshi Minor Plants", "capacity_mw": 20.0},
    "Tamakoshi": {"name": "Upper Tamakoshi HEP", "capacity_mw": 456.0},
    "Tamakoshi / Bhotekoshi": {"name": "Bhotekoshi HEP", "capacity_mw": 45.0},
    "Bhotekoshi / Indrawati": {"name": "Chilime HEP", "capacity_mw": 20.0},
    "Trishuli / Langtang": {"name": "Rasuwagadhi HEP", "capacity_mw": 111.0},
    "Trishuli": {"name": "Trishuli HEP", "capacity_mw": 24.0},
    "Trishuli / Budhigandaki": {"name": "Upper Trishuli-1 HEP", "capacity_mw": 216.0},
    "Bagmati / Bishnumati": {"name": "Sundarijal HEP", "capacity_mw": 0.97},
    "Hanumante Khola": {"name": "Hanumante Hydro", "capacity_mw": 5.0},
    "Bagmati / Nakhu Khola": {"name": "Nakhu Hydro", "capacity_mw": 5.0},
    "Roshi Khola / Sunkoshi": {"name": "Roshi Khola HEP", "capacity_mw": 8.0},
    "Kulekhani": {"name": "Kulekhani I & II", "capacity_mw": 92.0},
    "Narayani / Rapti": {"name": "Chitwan Hydro", "capacity_mw": 20.0},
    # Gandaki (4)
    "Budhigandaki": {"name": "Budhigandaki HEP", "capacity_mw": 1200.0},
    "Marsyangdi": {"name": "Middle Marsyangdi HEP", "capacity_mw": 70.0},
    "Kali Gandaki": {"name": "Kaligandaki-A HEP", "capacity_mw": 144.0},
    "Myagdi Khola / Kali Gandaki": {"name": "Modi HEP", "capacity_mw": 14.8},
    "Seti / Modi Khola": {"name": "Seti–Modi Hydro", "capacity_mw": 22.5},
    "Seti / Marsyangdi": {"name": "Marsyangdi HEP", "capacity_mw": 69.0},
    "Andhi Khola": {"name": "Andhi Khola HEP", "capacity_mw": 5.1},
    "Kali Gandaki / Ridi": {"name": "Ridi Hydro", "capacity_mw": 10.0},
    "Narayani": {"name": "Narayani Hydro", "capacity_mw": 45.0},
    "Myagdi Khola HEP": {"name": "Myagdi Khola HEP", "capacity_mw": 30.0},
    "Modi Khola": {"name": "Modi Khola HEP", "capacity_mw": 14.8},
    # Lumbini (5)
    "Bheri / Jhariganga": {"name": "Jhariganga HEP", "capacity_mw": 12.0},
    "Madi Khola / Lungri": {"name": "Lungri HEP", "capacity_mw": 7.5},
    "Lungri / West Rapti": {"name": "Madi Khola HEP", "capacity_mw": 10.0},
    "Arghau / West Rapti": {"name": "Arghau Hydro", "capacity_mw": 10.0},
    "Banganga / Rohini": {"name": "Banganga–Rohini Hydro", "capacity_mw": 10.0},
    "Tinau / Rohini": {"name": "Tinau HEP", "capacity_mw": 1.0},
    "Rapti / Babai": {"name": "Rapti–Babai Hydro", "capacity_mw": 10.0},
    "West Rapti": {"name": "West Rapti Hydro", "capacity_mw": 10.0},
    "Karnali / Babai": {"name": "Karnali–Babai Hydro", "capacity_mw": 10.0},
    # Karnali (6)
    "Thuli Bheri": {"name": "Thuli Bheri HEP", "capacity_mw": 3.0},
    "Mugu Karnali": {"name": "Mugu Karnali HEP", "capacity_mw": 1902.0},
    "Humla Karnali": {"name": "Humla Karnali HEP", "capacity_mw": 24.0},
    "Tila / Karnali": {"name": "Tila HEP", "capacity_mw": 1.8},
    "Karnali / Neruwa": {"name": "Neruwa Khola HEP", "capacity_mw": 5.0},
    "Kavre Khola": {"name": "Kavre Khola HEP", "capacity_mw": 9.4},
    "Bheri / Sani Bheri": {"name": "Bheri-Babai Diversion HEP", "capacity_mw": 48.0},
    "Bheri / Lungri": {"name": "Lungri HEP", "capacity_mw": 7.5},
    "Bheri / Bangad": {"name": "Bangad Kupinde HEP", "capacity_mw": 4.2},
    "Karnali / Bheri": {"name": "Karnali–Bheri Hydro", "capacity_mw": 10.0},
    # Sudurpashchim (7)
    "Seti / Budhiganga": {"name": "Budhiganga HEP", "capacity_mw": 10.0},
    "Seti": {"name": "Seti River HEP", "capacity_mw": 2.0},
    "Budhiganga / Khadganga": {"name": "Khadganga HEP", "capacity_mw": 14.9},
    "Seti Nadi": {"name": "Seti Gandaki HEP", "capacity_mw": 3.6},
    "Karnali / Mahakali": {"name": "Karnali–Mahakali Hydro", "capacity_mw": 10.0},
    "Mahakali / Sarda": {"name": "Sarada HEP", "capacity_mw": 5.0},
    "Dadeldhura Khola": {"name": "Dadeldhura HEP", "capacity_mw": 0.9},
    "Mahakali / Seti": {"name": "Chameliya (partial)", "capacity_mw": 10.0},
    "Mahakali / Chameliya": {"name": "Chameliya HEP", "capacity_mw": 30.0},
}


def _seasonal_rainfall(day_of_year: np.ndarray, rain_mult: float) -> np.ndarray:
    """Model Nepal's monsoon-driven rainfall pattern."""
    monsoon_peak = 200  # ~mid-July
    monsoon_width = 50
    monsoon_component = 28 * rain_mult * np.exp(-0.5 * ((day_of_year - monsoon_peak) / monsoon_width) ** 2)

    pre_monsoon_peak = 130
    pre_monsoon = 6 * rain_mult * np.exp(-0.5 * ((day_of_year - pre_monsoon_peak) / 30) ** 2)

    base_rain = 1.0 * rain_mult
    return base_rain + monsoon_component + pre_monsoon


def _seasonal_temperature(day_of_year: np.ndarray, base_temp: float) -> np.ndarray:
    """Sinusoidal annual temperature cycle peaking in June."""
    peak_day = 170  # ~mid-June
    amplitude = 8
    return base_temp + amplitude * np.sin(2 * np.pi * (day_of_year - peak_day + 91) / 365)


def _river_flow(rainfall_mm: np.ndarray, temperature: np.ndarray, elevation: float) -> np.ndarray:
    """
    Simplified river flow model:
    flow = f(cumulative recent rainfall, snowmelt proxy, base flow)
    """
    kernel_size = 7
    kernel = np.exp(-np.arange(kernel_size) / 3)
    kernel /= kernel.sum()
    cumulative_rain = np.convolve(rainfall_mm, kernel, mode="same")

    snowmelt = np.where(temperature > 5, 0.3 * (temperature - 5) * (elevation / 3000), 0)

    base_flow = 15 + elevation * 0.005
    flow = base_flow + 2.5 * cumulative_rain + snowmelt
    return np.clip(flow, 5, 500)


def _hydro_generation(river_flow: np.ndarray, capacity_mw: float) -> np.ndarray:
    """
    Hydropower output as a function of river flow.
    Uses a saturating curve: output plateaus near plant capacity.
    """
    flow_median = np.median(river_flow)
    normalized = river_flow / (flow_median + 1e-6)
    efficiency = 1 - np.exp(-1.2 * normalized)
    generation = capacity_mw * efficiency
    return np.clip(generation, 0, capacity_mw)


def generate_weather_data(
    start_date: str = "2020-01-01",
    end_date: str = "2025-12-31",
    seed: int = 42,
) -> pd.DataFrame:
    """Generate multi-district daily weather data."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start_date, end_date, freq="D")
    day_of_year = dates.dayofyear.values.astype(float)

    rows = []
    for district, props in DISTRICTS.items():
        n = len(dates)
        rain_base = _seasonal_rainfall(day_of_year, props["rain_mult"])
        rainfall = np.maximum(0, rain_base + rng.normal(0, rain_base * 0.4, n))
        dry_mask = rng.random(n) < np.where(rain_base < 3, 0.6, 0.1)
        rainfall[dry_mask] = 0
        rainfall = np.round(rainfall, 1)

        temp_base = _seasonal_temperature(day_of_year, props["base_temp"])
        temperature = np.round(temp_base + rng.normal(0, 2, n), 1)

        humidity = np.clip(55 + 0.8 * rainfall + rng.normal(0, 5, n), 20, 100).round(1)
        wind_speed = np.clip(rng.gamma(2, 2, n) + 1, 0.5, 25).round(1)
        pressure = np.round(1013 - props["elevation"] * 0.12 + rng.normal(0, 3, n), 1)

        df_district = pd.DataFrame({
            "date": dates,
            "district": district,
            "temperature_c": temperature,
            "rainfall_mm": rainfall,
            "humidity_pct": humidity,
            "wind_speed_kmh": wind_speed,
            "pressure_hpa": pressure,
        })
        rows.append(df_district)

    return pd.concat(rows, ignore_index=True)


def generate_river_data(weather_df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """Generate river flow data derived from weather patterns."""
    rng = np.random.default_rng(seed)
    rows = []
    for district, props in DISTRICTS.items():
        subset = weather_df[weather_df["district"] == district].copy()
        flow = _river_flow(
            subset["rainfall_mm"].values,
            subset["temperature_c"].values,
            props["elevation"],
        )
        flow += rng.normal(0, flow * 0.08)
        flow = np.clip(flow, 5, 500).round(2)

        river_name = props["river"]
        plant = HYDRO_PLANTS[river_name]
        generation = _hydro_generation(flow, plant["capacity_mw"])
        generation += rng.normal(0, generation * 0.03)
        generation = np.clip(generation, 0, plant["capacity_mw"]).round(2)

        df_river = pd.DataFrame({
            "date": subset["date"].values,
            "district": district,
            "river": river_name,
            "river_flow_cumecs": flow,
            "hydro_plant": plant["name"],
            "plant_capacity_mw": plant["capacity_mw"],
            "generation_mw": generation,
        })
        rows.append(df_river)

    return pd.concat(rows, ignore_index=True)


def generate_hourly_forecast_data(
    base_date: str = "2026-03-15",
    district: str = "Kaski",
    seed: int = 99,
) -> pd.DataFrame:
    """Generate 72-hour ahead hourly forecast data for dashboard demo."""
    rng = np.random.default_rng(seed)
    hours = pd.date_range(base_date, periods=72, freq="h")
    hour_of_day = hours.hour.values.astype(float)
    day_of_year = hours.dayofyear.values.astype(float)

    props = DISTRICTS[district]
    temp_daily = _seasonal_temperature(day_of_year, props["base_temp"])
    temp_hourly = temp_daily + 4 * np.sin(2 * np.pi * (hour_of_day - 14) / 24)
    temperature = np.round(temp_hourly + rng.normal(0, 1, len(hours)), 1)

    rain_base = _seasonal_rainfall(day_of_year, props["rain_mult"])
    rain_hourly = rain_base / 24
    rainfall = np.maximum(0, rain_hourly + rng.exponential(rain_hourly * 0.5, len(hours)))
    dry_mask = rng.random(len(hours)) < 0.5
    rainfall[dry_mask] = 0
    rainfall = np.round(rainfall, 2)

    humidity = np.clip(55 + 2.5 * rainfall + rng.normal(0, 3, len(hours)), 25, 100).round(1)

    flow = _river_flow(rainfall * 24, temperature, props["elevation"])
    flow += rng.normal(0, flow * 0.05)
    flow = np.clip(flow, 5, 500).round(2)

    plant = HYDRO_PLANTS[props["river"]]
    generation = _hydro_generation(flow, plant["capacity_mw"])
    generation += rng.normal(0, generation * 0.02)
    generation = np.clip(generation, 0, plant["capacity_mw"]).round(2)

    return pd.DataFrame({
        "datetime": hours,
        "temperature_c": temperature,
        "rainfall_mm": rainfall,
        "humidity_pct": humidity,
        "river_flow_cumecs": flow,
        "predicted_generation_mw": generation,
        "plant_capacity_mw": plant["capacity_mw"],
    })


def save_all_data(output_dir: str = "data"):
    """Generate and persist all datasets."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("Generating weather data (2020-2025, 5 districts)...")
    weather = generate_weather_data()
    weather.to_csv(out / "nepal_weather_2020_2025.csv", index=False)
    print(f"  → {len(weather):,} rows saved to nepal_weather_2020_2025.csv")

    print("Generating river flow & hydro generation data...")
    river = generate_river_data(weather)
    river.to_csv(out / "nepal_hydro_generation.csv", index=False)
    print(f"  → {len(river):,} rows saved to nepal_hydro_generation.csv")

    print("Generating 72-hour forecast demo data...")
    forecast = generate_hourly_forecast_data()
    forecast.to_csv(out / "forecast_demo_72h.csv", index=False)
    print(f"  → {len(forecast)} rows saved to forecast_demo_72h.csv")

    print("\nAll data generated successfully!")
    return weather, river, forecast


if __name__ == "__main__":
    save_all_data()
