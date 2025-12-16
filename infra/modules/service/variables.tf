variable "watson_url" {
  description = "The original Watson URL (HTTP)"
  type        = string
}

variable "vpc_id" { type = string }
variable "aws_region" { type = string }
variable "cluster_id" { type = string }
variable "subnet_ids" { type = list(string) }
variable "security_group_ids" { type = list(string) }
variable "execution_role_arn" { type = string }
variable "task_role_arn" { type = string }
variable "image_url" { type = string }
variable "model_registry_bucket" { type = string }
variable "service_discovery_namespace" {
  type    = string
  default = "caloreat.local"
}
