#!/bin/bash

#-------------------------------------------
# SET_VARIABLES
#-------------------------------------------
source ./scripts/project_config # PROJECT_NAME WORKDIR BUCKET GIT_USER GIT_CLONE GCP_LOGIN_NAME
TPU_NAME=$1
VM_NAME=$1
TRAIN_SCRIPT=scripts/train.py

gcloud compute ssh $VM_NAME -- pkill -f $TRAIN_SCRIPT