resource "aws_s3_bucket" "model_registry" {
  bucket_prefix = var.bucket_prefix
  force_destroy = true
}

# 모델 레지스트리 접근 권한을 위한 IAM Policy
resource "aws_iam_policy" "model_registry_policy" {
  name        = "caloreat-ai-model-registry-policy"
  description = "Access to Model Registry S3"

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
          aws_s3_bucket.model_registry.arn,
          "${aws_s3_bucket.model_registry.arn}/*"
        ]
      }
    ]
  })
}

# AI Module 전용 ECS Task Role
resource "aws_iam_role" "ai_task_role" {
  name = "caloreat-ai-task-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "ecs-tasks.amazonaws.com"
        }
      }
    ]
  })
}

resource "aws_iam_role_policy_attachment" "attach_model_registry" {
  role       = aws_iam_role.ai_task_role.name
  policy_arn = aws_iam_policy.model_registry_policy.arn
}
