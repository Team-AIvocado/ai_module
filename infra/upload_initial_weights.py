import boto3
import os

BUCKET_NAME = "caloreat-model-registry-20251213194629951300000002"
BASE_DIR = "../inference_module/weights"

# Mapping: Local Path (relative to BASE_DIR) -> S3 Key
MODELS = {
    "detector/kfood_integrated_detector/weights/best.pt": "detector/local-dev/best.pt",
    "classifier_effnet_finetuned/best_effnet_b0.pt": "efficientnet-b0-cls/local-dev/best_effnet_b0.pt",
    "classifier_yolo_finetuned/weights/best.pt": "yolo-cls/local-dev/best.pt"
}

def upload_models():
    s3 = boto3.client('s3')
    print(f"--- Uploading Initial Weights to {BUCKET_NAME} ---")
    
    for local_rel, s3_key in MODELS.items():
        local_path = os.path.join(BASE_DIR, local_rel)
        if not os.path.exists(local_path):
            print(f"❌ File not found: {local_path}")
            continue
            
        print(f"Uploading {local_rel} -> s3://{BUCKET_NAME}/{s3_key}")
        try:
            s3.upload_file(local_path, BUCKET_NAME, s3_key)
            print("✅ Uploaded")
        except Exception as e:
            print(f"❌ Failed: {e}")

if __name__ == "__main__":
    upload_models()
