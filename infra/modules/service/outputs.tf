output "service_name" {
  value = aws_ecs_service.ai.name
}

output "service_discovery_endpoint" {
  value = "${aws_service_discovery_service.ai.name}.${aws_service_discovery_private_dns_namespace.internal.name}"
}
