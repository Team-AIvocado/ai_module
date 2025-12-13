output "bucket_name" {
  value = aws_s3_bucket.model_registry.bucket
}

output "bucket_arn" {
  value = aws_s3_bucket.model_registry.arn
}

output "task_role_arn" {
  value = aws_iam_role.ai_task_role.arn
}
