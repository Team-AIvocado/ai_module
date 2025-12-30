
# ----------------------------------------------------------------------------------------------
# 1. EventBridge Rule (S3 Upload Trigger) - 기존 유지
# ----------------------------------------------------------------------------------------------
resource "aws_cloudwatch_event_rule" "s3_upload" {
  name        = "caloreat-training-trigger"
  description = "Trigger training when new dataset is uploaded to S3"

  event_pattern = jsonencode({
    source      = ["aws.s3"]
    detail-type = ["Object Created"]
    detail = {
      bucket = {
        name = [module.iam.bucket_name]
      }
      object = {
        key = [{
          prefix = "datasets/raw/"
        }]
      }
    }
  })
}

# ----------------------------------------------------------------------------------------------
# 2. Lambda Function (Trigger for SageMaker)
# ----------------------------------------------------------------------------------------------
data "archive_file" "lambda_zip" {
  type        = "zip"
  source_file = "${path.module}/lambdas/trigger_training.py"
  output_path = "${path.module}/lambdas/trigger_training.zip"
}

resource "aws_lambda_function" "trigger_training" {
  filename         = data.archive_file.lambda_zip.output_path
  function_name    = "caloreat-ai-trigger-training"
  role             = aws_iam_role.lambda_exec.arn
  handler          = "trigger_training.lambda_handler"
  source_code_hash = data.archive_file.lambda_zip.output_base64sha256
  runtime          = "python3.11"
  timeout          = 60

  environment {
    variables = {
      SAGEMAKER_ROLE_ARN = aws_iam_role.sagemaker_exec.arn
      TRAINING_IMAGE_URI = "${module.ecr.repository_url}:latest"
      INSTANCE_TYPE      = "ml.g4dn.xlarge"
    }
  }
}

resource "aws_lambda_permission" "allow_eventbridge" {
  statement_id  = "AllowExecutionFromEventBridge"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.trigger_training.function_name
  principal     = "events.amazonaws.com"
  source_arn    = aws_cloudwatch_event_rule.s3_upload.arn
}

# ----------------------------------------------------------------------------------------------
# 3. EventBridge Target -> Lambda
# ----------------------------------------------------------------------------------------------
resource "aws_cloudwatch_event_target" "lambda_training" {
  rule      = aws_cloudwatch_event_rule.s3_upload.name
  target_id = "TriggerTrainingLambda"
  arn       = aws_lambda_function.trigger_training.arn
}

# ----------------------------------------------------------------------------------------------
# 4. IAM Roles & Policies
# ----------------------------------------------------------------------------------------------

# --- A. Lambda Execution Role ---
resource "aws_iam_role" "lambda_exec" {
  name = "caloreat-lambda-training-trigger-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Action = "sts:AssumeRole"
      Effect = "Allow"
      Principal = { Service = "lambda.amazonaws.com" }
    }]
  })
}

resource "aws_iam_role_policy_attachment" "lambda_basic" {
  role       = aws_iam_role.lambda_exec.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole"
}

resource "aws_iam_role_policy" "lambda_sagemaker_policy" {
  name = "caloreat-lambda-sagemaker-policy"
  role = aws_iam_role.lambda_exec.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "sagemaker:CreateTrainingJob",
          "iam:PassRole" 
        ]
        Resource = "*" # Restrict resource in production
      }
    ]
  })
}

# --- B. SageMaker Execution Role (Used by the Training Job) ---
resource "aws_iam_role" "sagemaker_exec" {
  name = "caloreat-sagemaker-execution-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Action = "sts:AssumeRole"
      Effect = "Allow"
      Principal = { Service = "sagemaker.amazonaws.com" }
    }]
  })
}

# Standard SageMaker Policy (CloudWatch Logs, Metrics, etc.)
resource "aws_iam_role_policy_attachment" "sagemaker_full" {
  role       = aws_iam_role.sagemaker_exec.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonSageMakerFullAccess"
}

# S3 Access Policy for SageMaker (Read Datasets, Write Models)
resource "aws_iam_role_policy" "sagemaker_s3_policy" {
  name = "caloreat-sagemaker-s3-policy"
  role = aws_iam_role.sagemaker_exec.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "s3:GetObject",
          "s3:PutObject",
          "s3:ListBucket"
        ]
        Resource = [
          "arn:aws:s3:::${module.iam.bucket_name}",
          "arn:aws:s3:::${module.iam.bucket_name}/*"
        ]
      }
    ]
  })
}
