import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont 
import io

# --- AYARLAR ---
MODEL_PATH = "models/ultimate_colab.pt" # Eğittiğin en iyi modelin yolu
IMG_SIZE = 1280                         # Eğitim boyutu

# Sınıf İsimleri ve Renkler (Görselleştirme için)
CLASS_NAMES = {0: 'Impacted (Gömülü)', 1: 'Caries (Çürük)', 2: 'Lesion (Lezyon)', 3: 'Deep Caries (Derin Çürük)'}
COLORS = {
    0: (0, 120, 255),   # Mavi (Impacted)
    1: (255, 200, 0),   # Sarı (Caries) - Dikkat çekici
    2: (255, 0, 255),   # Mor (Lesion)
    3: (255, 50, 50)    # Kırmızı (Deep Caries) - Acil
}

# --- YARDIMCI FONKSİYONLAR ---

@st.cache_resource
def load_model():
    """Modeli hafızaya yükler ve cache'ler."""
    try:
        model = YOLO(MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"Model yüklenemedi! Yol: {MODEL_PATH}. Hata: {e}")
        return None

def apply_clahe(image_np):
    """Eğitimdeki ön işlemenin (CLAHE) aynısını uygular."""
    # Görüntü renkli ise griye çevir
    if len(image_np.shape) == 3:
        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    else:
        gray = image_np
    
    # CLAHE uygula (Kontrast artırma)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    
    # Tekrar 3 kanala çevir (YOLO ve Ekran için)
    return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)

def draw_text_with_pil(img_np, text, pos, color):
    # Numpy resmini PIL resmine çevir
    img_pil = Image.fromarray(img_np)
    draw = ImageDraw.Draw(img_pil)
    
    # Mac için Arial fontunu yükle (Türkçe destekler)
    try:
        font = ImageFont.truetype("/Library/Fonts/Arial.ttf", 22)
    except:
        font = ImageFont.load_default() # Bulamazsa varsayılan
    
    # Arka plan kutusu çiz (Yazı okunsun diye)
    bbox = draw.textbbox(pos, text, font=font)
    draw.rectangle([bbox[0]-5, bbox[1]-5, bbox[2]+5, bbox[3]+5], fill=color)
    
    # Yazıyı yaz (Siyah veya Beyaz)
    text_color = (0, 0, 0) if sum(color) > 400 else (255, 255, 255)
    draw.text(pos, text, font=font, fill=text_color)
    
    # Tekrar Numpy dizisine çevirip geri ver
    return np.array(img_pil)

def draw_predictions(image, results):
    """Tahmin kutularını resim üzerine çizer."""
    plot_img = image.copy()
    counts = {name: 0 for name in CLASS_NAMES.values()}
    
    for result in results:
        boxes = result.boxes
        for box in boxes:
            # Koordinatlar
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            
            class_name = CLASS_NAMES.get(cls, "Unknown")
            counts[class_name] += 1
            
            color = COLORS.get(cls, (0, 255, 0))
            label = f"{class_name} {conf:.2f}"
            
            # Kutu Çiz
            cv2.rectangle(plot_img, (x1, y1), (x2, y2), color, 3)
            
            plot_img = draw_text_with_pil(plot_img, label, (x1, y1 - 30), color)
            
    return plot_img, counts

# --- ANA UYGULAMA ---

st.set_page_config(page_title="Dental Diagnosis AI", layout="wide")

# Başlık
st.title("🦷 Dental Diagnosis Ops: AI Asistanı")
st.markdown("Panoramik diş röntgenlerinde **Çürük, Gömülü Diş ve Lezyon** tespiti.")

# Yan Menü (Sidebar)
st.sidebar.header("⚙️ Ayarlar")
conf_threshold = st.sidebar.slider("Güven Eşiği (Confidence)", 0.0, 1.0, 0.15, 0.05)
st.sidebar.info("Düşük eşik daha fazla tespit (ve yanlış alarm) demektir. Yüksek eşik sadece kesin olanları gösterir.")

# Model Yükleme
model = load_model()

# Dosya Yükleme Alanı
uploaded_file = st.file_uploader("Bir Röntgen Görüntüsü Yükleyin (JPG/PNG)", type=["jpg", "png", "jpeg"])

if uploaded_file is not None and model is not None:
    # Resmi Oku
    image = Image.open(uploaded_file)
    image_np = np.array(image)
    
    # İki Sütunlu Düzen
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🖼️ Orijinal Görüntü")
        st.image(image, use_container_width=True)
        
    # İşlem Butonu (Otomatik de yapılabilir ama buton daha kontrollü)
    if st.sidebar.button("Analiz Et") or True: # 'or True' dosyayı yükleyince otomatik çalıştırır
        with st.spinner('Yapay Zeka görüntüyü inceliyor...'):
            
            # 1. Ön İşleme (CLAHE)
            processed_img = apply_clahe(image_np)
            
            # 2. Tahmin (TTA Açık!)
            results = model.predict(processed_img, imgsz=IMG_SIZE, conf=conf_threshold, augment=True, verbose=False)
            
            # 3. Çizim
            result_img, detection_counts = draw_predictions(processed_img, results)
            
        with col2:
            st.subheader("🎯 AI Tespiti")
            st.image(result_img, use_container_width=True)
            
        # Rapor Kısmı
        st.divider()
        st.subheader("📋 Teşhis Raporu")
        
        # Metrikler yan yana
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Çürük (Caries)", detection_counts['Caries (Çürük)'], delta_color="inverse")
        m2.metric("Derin Çürük", detection_counts['Deep Caries (Derin Çürük)'], delta_color="inverse")
        m3.metric("Lezyon", detection_counts['Lesion (Lezyon)'], delta_color="inverse")
        m4.metric("Gömülü Diş", detection_counts['Impacted (Gömülü)'])
        
        if sum(detection_counts.values()) == 0:
            st.success("✅ Herhangi bir sorun tespit edilmedi.")
        else:
            st.warning(f"⚠️ Toplam {sum(detection_counts.values())} adet bulgu işaretlendi.")

else:
    st.info("Lütfen analiz etmek için bir röntgen görüntüsü yükleyin.")