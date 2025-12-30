import os
import random
import glob
from ultralytics import YOLO



def run_inference(image_path, model_path='runs/train/ship_obb_model4/weights/best.pt'):

    print(f"\n--- İşleniyor: {os.path.basename(image_path)} ---")


    if not os.path.exists(model_path):
        print(f"UYARI: Model bulunamadı: {model_path}.")
        print("Eğitilmemiş varsayılan modelle (yolov8n-obb.pt) deneme yapılıyor...")
        try:
            model = YOLO('yolov8n-obb.pt')
        except Exception as e:
            print(f"Kritik Hata: Varsayılan model de yüklenemedi. İnternet bağlantısını kontrol edin. Hata: {e}")
            return
    else:
        model = YOLO(model_path)


    results = model.predict(source=image_path, imgsz=1024, conf=0.25, save=True, project='runs/obb', name='inference_results', exist_ok=True)

    for result in results:
        print(f"Sonuç görseli kaydedildi: {result.save_dir}")

        # OBB kutularını al (poligon formatında)
        if result.obb:
            for i, box in enumerate(result.obb):
                conf_score = box.conf.item()
                cls_id = int(box.cls.item())
                cls_name = result.names[cls_id]
                print(f"  - Algılama {i+1}: Sınıf='{cls_name}', Güven={conf_score:.2f}")
        else:
            print("  - Bu resimde hiç gemi (OBB nesnesi) algılanamadı.")

if __name__ == "__main__":

    validation_images_dir = r"yolov8_obb/ship_dataset_yolo/images/val/"


    image_extensions = ['*.jpg', '*.jpeg', '*.png']
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(validation_images_dir, ext)))

    if not image_files:
        print(f"HATA: Belirtilen klasörde ({validation_images_dir}) hiç resim bulunamadı!")
    else:
        total_images = len(image_files)
        print(f"Klasörde toplam {total_images} resim bulundu.")

        num_to_select = min(5, total_images)
        selected_images = random.sample(image_files, num_to_select)

        print(f"Rastgele {num_to_select} resim seçildi. İşlem başlıyor...\n")

        for img_path in selected_images:
            run_inference(img_path)

        print("\n--- Tüm işlemler tamamlandı. ---")
        print("Sonuçları 'runs/obb/inference_results' klasöründe bulabilirsiniz.")