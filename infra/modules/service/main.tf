# -------------------------------------------------------------------------
# Service Discovery (Internal DNS)
# -------------------------------------------------------------------------

resource "aws_service_discovery_private_dns_namespace" "internal" {
  name        = var.service_discovery_namespace # "caloreat.local"
  description = "Internal service discovery namespace"
  vpc         = var.vpc_id
}

resource "aws_service_discovery_service" "ai" {
  name = "ai"

  dns_config {
    namespace_id = aws_service_discovery_private_dns_namespace.internal.id

    dns_records {
      ttl  = 10
      type = "A"
    }

    routing_policy = "MULTIVALUE"
  }

  health_check_custom_config {
    failure_threshold = 1
  }
}

# -------------------------------------------------------------------------
# ECS Task & Service
# -------------------------------------------------------------------------

resource "aws_ecs_task_definition" "ai" {
  family                   = "caloreat-ai-task"
  network_mode             = "awsvpc"
  requires_compatibilities = ["FARGATE"]
  cpu                      = 1024 # 1 vCPU
  memory                   = 4096 # 4GB

  execution_role_arn = var.execution_role_arn
  task_role_arn      = var.task_role_arn

  container_definitions = jsonencode([
    {
      name      = "ai-module"
      image     = "${var.image_url}:latest"
      essential = true
      portMappings = [
        {
          containerPort = 8001
          hostPort      = 8001
          protocol      = "tcp"
        }
      ]
      logConfiguration = {
        logDriver = "awslogs"
        options = {
          "awslogs-group"         = "/ecs/caloreat-ai"
          "awslogs-region"        = var.aws_region
          "awslogs-stream-prefix" = "ecs"
          "awslogs-create-group"  = "true"
        }
      }
      environment = [
        { name = "MODEL_REGISTRY_BUCKET", value = var.model_registry_bucket }
      ]
    }
  ])
}

resource "aws_ecs_service" "ai" {
  name            = "caloreat-ai-service"
  cluster         = var.cluster_id
  task_definition = aws_ecs_task_definition.ai.arn
  desired_count   = 1
  launch_type     = "FARGATE"

  network_configuration {
    subnets          = var.subnet_ids
    security_groups  = var.security_group_ids
    assign_public_ip = true
  }

  service_registries {
    registry_arn = aws_service_discovery_service.ai.arn
  }
}
