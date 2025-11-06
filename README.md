# SiteAdaptationForWestJava
Estimate Global Horizontal Irradiance (GHI) using a semi-empirical model, followed by post-processing with regression and deep learning for site adaptation. The smoothing step helps improve the accuracy of semi-empirical models by adjusting them to local atmospheric and geographic conditions.

# Project Overview 🌞

This repository provides a spatial estimation of GHI across tropical regions, with a specific focus on West Java, Indonesia. The workflow implements a semi-empirical model for GHI estimation, enhanced through site adaptation frameworks designed to improve model accuracy. Two site adaptation approaches are employed: a regression-based method and a deep learning approach using LSTM. Both techniques aim to adjust semi-empirical model outputs to better reflect local atmospheric and geographic conditions, resulting in improved predictive performance.

---

## Main Features ✨

- Semi-empirical estimation of Global Horizontal Irradiance (GHI)
- Site adaptation using regression and LSTM-based deep learning models to reduce bias and enhance local accuracy
- Generation of spatial GHI output matrices for the entire West Java region

---

## Outputs 🗺️

The model produces spatially distributed GHI matrices in CSV format, representing GHI estimation across West Java. These files can be visualized directly in GIS software (e.g., ArcGIS Pro or QGIS) using the Table to Point function to explore the spatial distribution of predicted GHI.

---

## How to Run ⚙️

1. Install the necessary Python libraries
   - joblib
   - tensorflow
   - numpy
   - pandas
   - netCDF4
   - scipy
3. Place all required input datasets (e.g., NC files, elevation data).
4. Adjust file paths in the source scripts as needed — particularly for NC files, elevation data, and deep learning models.  
5. Run the main script to generate spatial GHI estimations.

---

## Notes 📝

- This framework is specifically developed for West Java (UTC+7) using region-specific historical data.  
- The site adaptation parameters are not directly applicable to other regions, but the overall framework and methodology can be adapted for similar studies elsewhere.

---

## Authors 👩‍💻

- Afina Aristiani Zahra  
- Bintang Lamra Soetoopo
- Pranda Mulya Putra Garniwa
- Rifdah Octavi Azzahra  

**UNIVERSITAS INDONESIA**  



---
Developed as part of the early-phase implementation of the RE-NUSANTARA 1.0 platform, focusing on spatially adaptive machine learning for near real-time solar radiation prediction in tropical regions.
