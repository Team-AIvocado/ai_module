variable "watson_url" {
  description = "URL for WatsonX AI (HTTP)"
  type        = string
}

variable "watson_api_key" {
  description = "API Key for WatsonX"
  type        = string
  sensitive   = true
}

variable "watson_project_id" {
  description = "Project ID for WatsonX"
  type        = string
}
