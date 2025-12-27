import boto3
import sys
import time

def verify_trigger():
    # 1. Bucket Name Input
    if len(sys.argv) < 2:
        print("Usage: python verify_mlops.py <BUCKET_NAME>")
        sys.exit(1)
        
    bucket_name = sys.argv[1]
    
    # 2. Create Dummy CSV
    csv_content = "image_path,label\n/dummy/img1.jpg,Kimchi\n/dummy/img2.jpg,Bulgogi"
    file_key = f"datasets/raw/test_trigger_{int(time.time())}.csv"
    
    # 3. Upload to S3
    print(f"--- MLOps Trigger Verification ---")
    print(f"Target Bucket: {bucket_name}")
    print(f"Uploading dummy CSV to: s3://{bucket_name}/{file_key} ...")
    
    try:
        s3 = boto3.client('s3')
        s3.put_object(
            Bucket=bucket_name,
            Key=file_key,
            Body=csv_content.encode('utf-8')
        )
        print("✅ Upload Success!")
        print("-" * 30)
        print("Next Steps:")
        print("1. Go to AWS Console -> CloudWatch -> Log groups")
        print("2. Find '/ecs/caloreat-training'")
        print("3. Check if a new log stream started.")
        print("-" * 30)
    except Exception as e:
        print(f"❌ Upload Failed: {e}")

if __name__ == "__main__":
    verify_trigger()
