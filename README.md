
# 🦷 Dental-Diagnosis-Ops: AI Tabanlı Radyografi Analiz Sistemi
> **YOLOv8**, **Gelişmiş Görüntü İşleme (CLAHE)** ve **Streamlit** kullanılarak geliştirilmiş, panoramik diş röntgenlerinde çürük, gömülü diş ve lezyon tespiti yapan uçtan uca MLOps projesi.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![YOLOv8](https://img.shields.io/badge/Model-YOLOv8_Large-green)
![Streamlit](https://img.shields.io/badge/App-Streamlit-red)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## 🎯 Proje Hakkında
Bu proje, diş hekimlerinin teşhis sürecine hız kazandırmak ve gözden kaçabilecek başlangıç seviyesindeki patolojileri tespit etmek amacıyla geliştirilmiştir. Sistem, panoramik röntgen görüntülerini (OPG) analiz eder ve aşağıdaki durumları tespit eder:

* **Gömülü Diş (Impacted)**
* **Çürük (Caries)**
* **Derin Çürük (Deep Caries)**
* **Periapical Lezyon (Lesion)**

---

## 🛠️ Teknik Mimari ve Mühendislik Kararları

Bu projede rastgele araçlar değil, probleme özel optimize edilmiş mühendislik çözümleri seçilmiştir:

### 1. Model Seçimi: Neden YOLOv8?
Diş röntgenlerinde hem "Teşhis" (Bu nedir?) hem de "Konumlandırma" (Hangi dişte?) gerektiği için Sınıflandırma (ResNet vb.) yerine **Nesne Tespiti (Object Detection)** mimarisi gereklidir.
* **Hız/Performans Dengesi:** YOLOv8, tek aşamalı (one-stage) bir dedektör olduğu için Faster R-CNN gibi modellere göre çok daha hızlıdır ve gerçek zamanlı kullanıma uygundur.
* **Global Bağlam:** YOLO resmin bütününe baktığı için, dişin konumunu çevresindeki kemik yapısıyla ilişkilendirerek daha doğru karar verir.

### 2. Görüntü İşleme: CLAHE Tekniği
Röntgen görüntülerindeki en büyük zorluk, düşük kontrast ve homojen olmayan aydınlatmadır. Çürükler genellikle diş minesiyle benzer gri tonlarında olduğu için modelin ayırt etmesi zordur.

Bu sorunu çözmek için **CLAHE (Contrast Limited Adaptive Histogram Equalization)** tekniği entegre edilmiştir:
* **Nasıl Çalışır?** Standart histogram eşitlemenin aksine, CLAHE görüntüyü küçük bölgelere (tiles) ayırır ve her bölgenin kontrastını yerel olarak artırır. Gürültüyü (noise) engellemek için kontrast artışını sınırlar (Clip Limit).
* **Sonuç:** Diş köklerindeki lezyonlar ve mine üzerindeki küçük çürükler, "parlatılarak" model için görünür hale getirilmiştir. Bu işlem, Recall (Yakalanma) oranını %30'dan %50+ seviyesine çıkarmıştır.

### 3. API ve Arayüz: Neden Streamlit?
Modelin son kullanıcıya (doktorlara) sunulması aşamasında Flask veya Django yerine **Streamlit** tercih edilmiştir.
* **Hızlı Prototipleme:** Karmaşık Frontend (HTML/CSS/JS) süreçleriyle vakit kaybetmek yerine, doğrudan Python kodu ile interaktif bir web arayüzü oluşturulmasını sağlar.
* **Veri Odaklı:** Streamlit, veri bilimi projeleri için optimize edilmiştir. Görüntü işleme sonuçlarını, güven skorlarını ve rapor tablolarını göstermek için yerleşik ve hızlı bileşenler sunar.

### 4. Veri Seti Stratejisi
DENTEX veri setinin hiyerarşik yapısı analiz edilmiş ve sadece **`quadrant-enumeration-disease`** alt kümesi kullanılmıştır. Diğer klasörler hastalık etiketi içermediği için elenmiştir.

---

## 📊 Model Performansı
Model, **Google Colab (Tesla T4 GPU)** üzerinde **1280px** yüksek çözünürlükle eğitilmiştir.

| Sınıf | mAP50 (Başarı) | Yorum |
| :--- | :---: | :--- |
| **Gömülü Diş** | **%95.3** | Mükemmel tespit başarısı. |
| **Derin Çürük** | **%66.4** | Büyük deformasyonlar net tespit ediliyor. |
| **Çürük** | **%51.5** | Küçük ve zorlu vakalarda (CLAHE sayesinde) yüksek başarı. |
| **Lezyon** | **%51.9** | En zor sınıf olan kök ucu lezyonlarında istikrarlı tespit. |
| **GENEL** | **%66.3** | Ortalama başarı (Baseline modele göre +%14 artış). |

---

## 🚀 Kurulum ve Çalıştırma

Projeyi yerel ortamınızda çalıştırmak için adımları izleyin:

```bash
# 1. Repoyu klonlayın
git clone [https://github.com/KULLANICI_ADINIZ/Dental-Diagnosis-Ops.git](https://github.com/KULLANICI_ADINIZ/Dental-Diagnosis-Ops.git)
cd Dental-Diagnosis-Ops

# 2. Sanal ortamı kurun ve aktif edin
python -m venv .venv
source .venv/bin/activate  # Windows için: .venv\Scripts\activate

# 3. Gerekli kütüphaneleri yükleyin
pip install -r requirements.txt

# 4. Uygulamayı başlatın
streamlit run src/app/app.py
````

-----

## 📂 Proje Yapısı

```
Dental-Diagnosis-Ops/
├── data/               # Ham ve işlenmiş veriler (Gitignored - 11GB)
├── models/             # Eğitilmiş modeller (.pt dosyaları)
├── notebooks/          # EDA, Preprocessing ve Colab çalışmaları
├── reports/            # Tahmin görselleri ve karşılaştırma raporları
├── src/
│   ├── app/            # Streamlit web uygulaması (app.py)
│   ├── data/           # Veri işleme scriptleri (make_dataset.py)
│   └── models/         # Eğitim ve tahmin scriptleri
└── requirements.txt    # Proje bağımlılıkları
```

-----

## 📚 Kaynakça ve Lisans

Bu projede kullanılan **DENTEX** veri seti, İbrahim Ethem Hamamcı ve ekibi tarafından hazırlanmıştır. Veri seti **CC-BY-NC-SA 4.0** lisansı altındadır.

**Referans Makaleler:**

1.  *Hamamci, I. E., et al. "DENTEX: An Abnormal Tooth Detection with Dental Enumeration and Diagnosis Benchmark for Panoramic X-rays." arXiv preprint arXiv:2305.19112 (2023).*
2.  *Hamamci, I. E., et al. "Diffusion-based hierarchical multi-label object detection to analyze panoramic dental x-rays." MICCAI (2023).*
