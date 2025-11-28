import json
import shutil
from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm  # İlerleme çubuğu için

# --- AYARLAR ---
# Proje kök dizinini bul (src klasöründen bir üst dizine çık)
BASE_DIR = Path(__file__).resolve().parent.parent

# Ham Veri Yolları (Senin LS çıktılarına göre)
RAW_DATA_DIR = BASE_DIR / "data/raw/DENTEX"

# Train Yolları
TRAIN_INPUT_DIR = RAW_DATA_DIR / "train/training_data/quadrant-enumeration-disease"
TRAIN_JSON = TRAIN_INPUT_DIR / "train_quadrant_enumeration_disease.json"
TRAIN_IMAGES = TRAIN_INPUT_DIR / "xrays"

# Val Yolları (Dikkat: Val klasöründe tire yerine alt çizgi kullanılmıştı)
VAL_INPUT_DIR = RAW_DATA_DIR / "val/validation_data/quadrant_enumeration_disease"
# Val JSON dosyası ana dizindeydi, onu buraya referans veriyoruz
VAL_JSON = RAW_DATA_DIR / "validation_triple.json" 
VAL_IMAGES = VAL_INPUT_DIR / "xrays"

# Hedef Klasör
PROCESSED_DIR = BASE_DIR / "data/processed"
IMG_SIZE = 640  # YOLO standart boyutu

# Sınıf Haritası (Validation JSON'dan öğrendiğimiz hastalık ID'leri)
# YOLO sınıf ID'leri 0'dan başlamalıdır.
CLASS_MAPPING = {
    0: 0, # Impacted
    1: 1, # Caries
    2: 2, # Periapical Lesion
    3: 3  # Deep Caries
}

def setup_directories():
    """YOLO klasör yapısını oluşturur."""
    for split in ['train', 'val']:
        (PROCESSED_DIR / split / 'images').mkdir(parents=True, exist_ok=True)
        (PROCESSED_DIR / split / 'labels').mkdir(parents=True, exist_ok=True)
    print(f"✅ Klasör yapısı hazır: {PROCESSED_DIR}")

def convert_bbox_to_yolo(bbox, img_width, img_height):
    """COCO bbox (x_min, y_min, w, h) -> YOLO bbox (x_center, y_center, w, h) normalize."""
    x_min, y_min, w, h = bbox
    
    x_center = (x_min + w / 2) / img_width
    y_center = (y_min + h / 2) / img_height
    width = w / img_width
    height = h / img_height
    
    return x_center, y_center, width, height

def resize_image_letterbox(image, target_size):
    """Resmi bozmadan (aspect ratio koruyarak) target_size'a sığdırır ve padding ekler."""
    h, w = image.shape[:2]
    scale = min(target_size / w, target_size / h)
    nw, nh = int(w * scale), int(h * scale)
    
    image_resized = cv2.resize(image, (nw, nh))
    
    # Padding (Dolgu) oluştur
    image_padded = np.full((target_size, target_size, 3), 128, dtype=np.uint8) # Gri dolgu
    
    # Resmi merkeze yerleştir
    dx = (target_size - nw) // 2
    dy = (target_size - nh) // 2
    image_padded[dy:dy+nh, dx:dx+nw] = image_resized
    
    return image_padded, scale, dx, dy

def process_dataset(json_path, image_dir, split_name):
    """Verilen veri setini işler, dönüştürür ve kaydeder."""
    print(f"\n🚀 {split_name.upper()} veri seti işleniyor...")
    
    # JSON Yükle
    with open(json_path, 'r') as f:
        data = json.load(f)
        
    # Resim ID'lerine göre dosya isimlerini eşleştir
    img_dict = {img['id']: img for img in data['images']}
    
    # Anotasyonları resim ID'sine göre grupla
    ann_dict = {}
    for ann in data['annotations']:
        img_id = ann['image_id']
        if img_id not in ann_dict:
            ann_dict[img_id] = []
        ann_dict[img_id].append(ann)
        
    # İşlem döngüsü
    for img_id, img_info in tqdm(img_dict.items()):
        file_name = img_info['file_name']
        src_path = image_dir / file_name
        
        # Resim dosyasını kontrol et
        if not src_path.exists():
            # Bazen dosya isimleri json ile diskte uyuşmayabilir, basit bir kontrol
            continue
            
        # 1. RESMİ OKU VE BOYUTLANDIR
        img = cv2.imread(str(src_path))
        if img is None:
            continue
            
        img_h, img_w = img.shape[:2]
        processed_img, scale, dx, dy = resize_image_letterbox(img, IMG_SIZE)
        
        # 2. ANOTASYONLARI İŞLE
        yolo_labels = []
        if img_id in ann_dict:
            for ann in ann_dict[img_id]:
                # Sadece Hastalık (Diagnosis) kategorisini al (category_id_3)
                cat_id = ann.get('category_id_3')
                
                # Eğer hastalık ID'si bizim haritamızda varsa işle
                if cat_id in CLASS_MAPPING:
                    class_id = CLASS_MAPPING[cat_id]
                    bbox = ann['bbox'] # x, y, w, h
                    
                    # Orijinal bbox'ı resize işlemine göre güncelle
                    # (Resmi küçülttük ve padding ekledik, kutular da kaymalı)
                    x = bbox[0] * scale + dx
                    y = bbox[1] * scale + dy
                    w = bbox[2] * scale
                    h = bbox[3] * scale
                    
                    # YOLO formatına çevir (Normalize et)
                    xc, yc, nw, nh = convert_bbox_to_yolo((x, y, w, h), IMG_SIZE, IMG_SIZE)
                    
                    yolo_labels.append(f"{class_id} {xc:.6f} {yc:.6f} {nw:.6f} {nh:.6f}")
        
        # 3. KAYDET
        # Resmi kaydet
        save_img_path = PROCESSED_DIR / split_name / 'images' / file_name
        cv2.imwrite(str(save_img_path), processed_img)
        
        # Etiketi kaydet (Eğer etiket varsa)
        if yolo_labels:
            label_name = Path(file_name).stem + ".txt"
            save_label_path = PROCESSED_DIR / split_name / 'labels' / label_name
            with open(save_label_path, 'w') as f:
                f.write("\n".join(yolo_labels))

if __name__ == "__main__":
    setup_directories()
    
    # Önce Validation setini işle (Daha küçük, test için iyi)
    if VAL_JSON.exists() and VAL_IMAGES.exists():
        process_dataset(VAL_JSON, VAL_IMAGES, 'val')
    else:
        print("❌ Val dosyaları bulunamadı, atlanıyor.")

    # Sonra Train setini işle
    if TRAIN_JSON.exists() and TRAIN_IMAGES.exists():
        process_dataset(TRAIN_JSON, TRAIN_IMAGES, 'train')
    else:
        print(f"❌ Train dosyaları bulunamadı: {TRAIN_JSON}")