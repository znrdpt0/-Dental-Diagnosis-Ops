from ultralytics import YOLO
import torch
import os
from pathlib import Path

# --- AYARLAR ---
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_YAML = BASE_DIR / "data.yaml"
PROJECT_DIR = BASE_DIR / "models"  
NAME = "v1_yolov8s_dental"

def main():
    # 1. Cihaz Kontrolü (Mac MPS Desteği)
    if torch.backends.mps.is_available():
        device = 'mps'
        print(f"🚀 Apple Silicon GPU (MPS) algılandı.")
    else:
        device = 'cpu'
        print("⚠️ GPU bulunamadı, işlem CPU üzerinden ilerleyiyor.")

    # 2. Modeli Başlat (Transfer Learning)
    model = YOLO('yolov8s.pt') 

    print("🧠 Model eğitimi başlıyor...")

    # 3. Eğitimi Başlat
    results = model.train(
        data=str(DATA_YAML),
        project=str(PROJECT_DIR),
        name=NAME,
        
        # --- Donanım Ayarları ---
        device=device,
        epochs=50,          
        imgsz=640,          
        batch=8,            
        workers=4,         
        
        # --- AUGMENTATION (Veri Çoğaltma) ---
        mosaic=1.0,         
        mixup=0.1,          
        degrees=10.0,       
        fliplr=0.5,        
        scale=0.5,          
        
        # --- İleri Seviye ---
        patience=10,        # Early Stopping
        save=True,          
        exist_ok=True,      
        verbose=True
    )

    print(f"✅ Eğitim tamamlandı! Sonuçlar: {PROJECT_DIR}/{NAME}")

if __name__ == '__main__':
    main()