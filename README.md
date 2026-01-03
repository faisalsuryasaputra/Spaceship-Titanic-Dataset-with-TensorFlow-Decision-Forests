# 🚀 Spaceship Titanic: TensorFlow Decision Forests

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.11%2B-orange)
![TF-DF](https://img.shields.io/badge/TF--DF-Random%20Forest-green)
![Status](https://img.shields.io/badge/Status-Completed-success)

> **Sebuah solusi baseline yang kuat menggunakan TensorFlow Decision Forests (TF-DF) untuk kompetisi Kaggle Spaceship Titanic.**

## 📌 Gambaran Umum (Overview)

Proyek ini bertujuan untuk memprediksi apakah seorang penumpang di Spaceship Titanic ikut terangkut ke dimensi lain atau tidak. Kami menggunakan **TensorFlow Decision Forests (TF-DF)**, sebuah library yang memungkinkan pelatihan model berbasis *tree* (seperti Random Forest) dengan API Keras yang mudah digunakan.

Pendekatan ini sangat efektif untuk data tabular karena membutuhkan pra-pemrosesan (preprocessing) yang minimal dibandingkan dengan jaringan saraf tiruan (neural networks) klasik.

## 📊 Dataset

Dataset berasal dari kompetisi Kaggle: [Spaceship Titanic](https://www.kaggle.com/c/spaceship-titanic).
* **Total Data Latih:** 8,693 entri dengan 14 fitur.
* **Target:** `Transported` (Boolean) - Apakah penumpang selamat/terangkut.
* **Fitur Utama:** `HomePlanet`, `CryoSleep`, `Cabin`, `Destination`, `Age`, `VIP`, dan pengeluaran fasilitas mewah (`RoomService`, `FoodCourt`, dll).



## 🛠️ Metodologi & Alur Kerja

Kode dalam repositori ini mencakup langkah-langkah *end-to-end* berikut:

### 1. Eksplorasi Data (EDA)
Kami melakukan analisis distribusi data numerik dan kategorikal untuk memahami karakteristik penumpang.


### 2. Pra-pemrosesan Data (Preprocessing)
TF-DF menangani banyak tipe data secara *native*, namun beberapa penyesuaian tetap dilakukan:
* **Handling Missing Values:** Mengisi nilai null pada kolom numerik dan boolean dengan `0`.
* **Boolean Conversion:** TF-DF saat ini belum mendukung tipe data boolean secara langsung, sehingga kolom seperti `Transported`, `VIP`, dan `CryoSleep` dikonversi menjadi `integer` (0 atau 1).
* **Dropping Columns:** Menghapus kolom `PassengerId` dan `Name` yang tidak relevan untuk pelatihan.

### 3. Feature Engineering
Fitur `Cabin` yang berisi format string `Deck/Num/Side` dipecah menjadi 3 fitur baru yang lebih informatif:
* `Deck`
* `Cabin_num`
* `Side`

### 4. Pemodelan (Modeling)
Kami menggunakan algoritma **Random Forest** standar dari TF-DF.
* **Model:** `tfdf.keras.RandomForestModel()`
* **Split Data:** 80% Training, 20% Validation.
* **Format Data:** Konversi dari Pandas DataFrame ke `tf.data.Dataset` untuk kinerja optimal.



## 📈 Hasil Evaluasi & Performa

Model dievaluasi menggunakan metrik akurasi pada data validasi dan *Out-of-Bag* (OOB) data.

* **Waktu Pelatihan:** ~54 detik.
* **OOB Accuracy:** ~79.73%
* **Validation Accuracy:** **80.25%**

Grafik di bawah ini menunjukkan peningkatan akurasi OOB seiring bertambahnya jumlah pohon (trees) dalam model:


### Feature Importance (Kepentingan Fitur)
Berdasarkan metrik `NUM_AS_ROOT` (seberapa sering fitur menjadi akar pohon), fitur yang paling berpengaruh adalah:
1.  **CryoSleep** (Sangat dominan)
2.  **RoomService**
3.  **Spa**
4.  **VRDeck**

## 💻 Cara Menjalankan

1.  **Instalasi Dependencies:**
    ```bash
    pip install tensorflow tensorflow_decision_forests pandas numpy seaborn matplotlib
    ```

2.  **Jalankan Notebook:**
    Buka file notebook (misalnya `spaceship_titanic_tfdf.ipynb`) di Jupyter Notebook, Google Colab, atau Kaggle Kernel.

3.  **Output:**
    Script akan menghasilkan file `submission.csv` yang siap diunggah ke Kaggle.

## 🤝 Kontribusi
Repositori ini dirancang sebagai *baseline* pembelajaran. Silakan lakukan *fork* dan eksperimen dengan:
* Menggunakan `GradientBoostedTreesModel` alih-alih Random Forest.
* Melakukan hyperparameter tuning yang lebih mendalam.
* Teknik *imputation* data hilang yang lebih canggih.

---
*Dibuat berdasarkan implementasi TensorFlow Decision Forests v1.2.0*
