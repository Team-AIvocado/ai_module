import json
import boto3
import os
import time

sagemaker = boto3.client("sagemaker")


def lambda_handler(event, context):
    print("Received event:", json.dumps(event))

    # EventBridge S3 event structure can vary depending on configuration
    # Assuming "detail" -> "bucket" -> "name" and "detail" -> "object" -> "key"
    try:
        detail = event.get("detail", {})
        bucket_name = detail.get("bucket", {}).get("name")
        object_key = detail.get("object", {}).get("key")

        if not bucket_name or not object_key:
            print("Error: Could not extract bucket or key from event")
            return {"statusCode": 400, "body": "Invalid event structure"}

        s3_uri = f"s3://{bucket_name}/{object_key}"
        print(f"Triggering training for dataset: {s3_uri}")

        # Configuration from Environment Variables
        role_arn = os.environ.get("SAGEMAKER_ROLE_ARN")
        image_uri = os.environ.get("TRAINING_IMAGE_URI")
        instance_type = os.environ.get("INSTANCE_TYPE", "ml.g4dn.xlarge")
        job_name_prefix = "caloreat-training"

        timestamp = time.strftime("%Y-%m-%d-%H-%M-%S", time.gmtime())
        job_name = f"{job_name_prefix}-{timestamp}"

        # Create Training Job
        response = sagemaker.create_training_job(
            TrainingJobName=job_name,
            AlgorithmSpecification={
                "TrainingImage": image_uri,
                "TrainingInputMode": "File",
                "EnableSageMakerMetricsTimeSeries": True,  # Metrics tracking
            },
            RoleArn=role_arn,
            InputDataConfig=[
                {
                    "ChannelName": "train",
                    "DataSource": {
                        "S3DataSource": {
                            "S3DataType": "S3Prefix",
                            "S3Uri": s3_uri,
                            "S3DataDistributionType": "FullyReplicated",
                        }
                    },
                    "ContentType": "text/csv",
                    "InputMode": "File",
                }
            ],
            OutputDataConfig={"S3OutputPath": f"s3://{bucket_name}/output/models/"},
            ResourceConfig={
                "InstanceType": instance_type,
                "InstanceCount": 1,
                "VolumeSizeInGB": 50,
            },
            StoppingCondition={
                "MaxRuntimeInSeconds": 3600,
                "MaxWaitTimeInSeconds": 3600 # Required for Spot Training (>= MaxRuntime)
            },
            EnableManagedSpotTraining=True,  # Cost optimization!
            CheckpointConfig={  # Required for Spot Training
                "S3Uri": f"s3://{bucket_name}/checkpoints/{job_name}",
                "LocalPath": "/opt/ml/checkpoints",
            },
            # Pass Hyperparameters if needed
            HyperParameters={"epochs": "5", "batch-size": "32"},
        )

        print(f"Started Training Job: {job_name}")
        return {"statusCode": 200, "body": json.dumps(f"Started job {job_name}")}

    except Exception as e:
        print(f"Error: {e}")
        raise e
