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

resource "aws_cloudwatch_event_target" "ecs_training" {
  rule      = aws_cloudwatch_event_rule.s3_upload.name
  target_id = "TriggerTrainingTask"
  arn       = data.aws_ecs_cluster.main.arn
  role_arn  = aws_iam_role.eventbridge_role.arn

  ecs_target {
    task_count          = 1
    task_definition_arn = aws_ecs_task_definition.training.arn
    launch_type         = "FARGATE"
    network_configuration {
      subnets          = data.aws_subnets.private.ids
      security_groups  = [data.aws_security_group.ecs_sg.id]
      assign_public_ip = true
    }
  }

  input_transformer {
    input_paths = {
      s3_bucket = "$.detail.bucket.name"
      s3_key    = "$.detail.object.key"
    }
    input_template = <<EOF
{
  "containerOverrides": [
    {
      "name": "training-container",
      "command": ["python", "-m", "training.train", "s3://<s3_bucket>/<s3_key>", "--epochs", "5"]
    }
  ]
}
EOF
  }
}

resource "aws_iam_role" "eventbridge_role" {
  name = "caloreat-eventbridge-ecs-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "events.amazonaws.com"
        }
      }
    ]
  })
}

resource "aws_iam_role_policy" "eventbridge_policy" {
  name = "caloreat-eventbridge-ecs-policy"
  role = aws_iam_role.eventbridge_role.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "ecs:RunTask"
        ]
        Resource = [aws_ecs_task_definition.training.arn]
        Condition = {
          ArnEquals = {
            "ecs:cluster" = data.aws_ecs_cluster.main.arn
          }
        }
      },
      {
        Effect = "Allow"
        Action = "iam:PassRole"
        Resource = [
          data.aws_iam_role.execution_role.arn,
          module.iam.task_role_arn
        ]
      }
    ]
  })
}

resource "aws_ecs_task_definition" "training" {
  family                   = "caloreat-training-task"
  network_mode             = "awsvpc"
  requires_compatibilities = ["FARGATE"]
  cpu                      = 2048 # 2 vCPU
  memory                   = 8192 # 8 GB 

  execution_role_arn = data.aws_iam_role.execution_role.arn
  task_role_arn      = module.iam.task_role_arn

  container_definitions = jsonencode([
    {
      name      = "training-container"
      image     = "${module.ecr.repository_url}:latest"
      essential = true
      command   = ["python", "-m", "training.train", "dataset_placeholder.csv"] # Overridden by EventBridge
      
      logConfiguration = {
        logDriver = "awslogs"
        options = {
          "awslogs-group"         = "/ecs/caloreat-training"
          "awslogs-region"        = "ap-northeast-2"
          "awslogs-stream-prefix" = "training"
          "awslogs-create-group"  = "true"
        }
      }
      environment = [
        { name = "MODEL_REGISTRY_BUCKET", value = module.iam.bucket_name }
      ]
    }
  ])
}

# S3 EventBridge Notification Enable (Optional but recommended)
resource "aws_s3_bucket_notification" "bucket_notification" {
  bucket      = module.iam.bucket_name
  eventbridge = true
}
