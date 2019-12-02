#! /bin/bash

#-------------------------------------------
# SET_VARIABLES
#-------------------------------------------
source ./scripts/project_config # PROJECT_NAME WORKDIR BUCKET GIT_USER GIT_CLONE GCP_LOGIN_NAME
TPU_NAME=$1
VM_NAME=$1
TRAIN_SCRIPT=scripts/train.py

gcloud compute scp --recurse ./scripts $VM_NAME:~/$PROJECT_NAME/$WORKDIR
gcloud compute ssh $VM_NAME --command "export TPU_NAME=$TPU_NAME; cd ~/$PROJECT_NAME/$WORKDIR && python3 $TRAIN_SCRIPT"