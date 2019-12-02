#!/bin/bash
TPU_NAME=$1
VM_NAME=$1

# Stop TPU
echo Stopping TPU $TPU_NAME &
yes Y | gcloud -q compute tpus stop $TPU_NAME &

# # Stop VM
echo Stopping instance $VM_NAME &
yes Y | gcloud -q compute instances stop $VM_NAME &

wait
echo Done