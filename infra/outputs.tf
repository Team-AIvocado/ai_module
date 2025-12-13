output "ecr_repository_url" {
  value = module.ecr.repository_url
}

output "model_registry_bucket" {
  value = module.iam.bucket_name
}

output "ai_service_discovery_endpoint" {
  value = module.service.service_discovery_endpoint
}
