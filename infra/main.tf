terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region = "ap-northeast-2"
}

# -------------------------------------------------------------------------
# 1. Data Sources (기존 인프라 참조)
# -------------------------------------------------------------------------

# VPC: Default VPC 사용 (Backend와 동일)
data "aws_vpc" "main" {
  default = true
}

# Subnets: Default VPC 내의 모든 서브넷 (Public)
data "aws_subnets" "private" {
  filter {
    name   = "vpc-id"
    values = [data.aws_vpc.main.id]
  }
}

# Existing ECS Cluster
data "aws_ecs_cluster" "main" {
  cluster_name = "caloreat-cluster"
}

# ECS Task Execution Role
data "aws_iam_role" "execution_role" {
  name = "caloreat-ecs-task-execution-role"
}

# ECS Security Group (Backend에서 생성한 SG)
data "aws_security_group" "ecs_sg" {
  filter {
    name   = "group-name"
    values = ["caloreat-ecs-sg"]
  }
  vpc_id = data.aws_vpc.main.id
}

# -------------------------------------------------------------------------
# 2. Modules
# -------------------------------------------------------------------------

module "ecr" {
  source = "./modules/ecr"
}

module "iam" {
  source = "./modules/iam"
}

module "service" {
  source = "./modules/service"

  vpc_id                = data.aws_vpc.main.id
  aws_region            = "ap-northeast-2"
  cluster_id            = data.aws_ecs_cluster.main.id
  subnet_ids            = data.aws_subnets.private.ids
  security_group_ids    = [data.aws_security_group.ecs_sg.id]
  execution_role_arn    = data.aws_iam_role.execution_role.arn
  task_role_arn         = module.iam.task_role_arn
  image_url             = module.ecr.repository_url
  model_registry_bucket = module.iam.bucket_name
  watson_url            = var.watson_url
}
