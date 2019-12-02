#!/bin/bash
TPU_NAME=$1
VM_NAME=$1

# Start TPU
echo Starting TPU $TPU_NAME &
gcloud compute tpus start $TPU_NAME &

# # Create VM
echo Starting instance $VM_NAME &
gcloud compute instances start $VM_NAME &

wait
echo Done