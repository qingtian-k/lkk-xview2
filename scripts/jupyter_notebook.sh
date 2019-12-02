#!/bin/bash
VM_NAME=$1

# Get IP Address
IP_ADDRESS=$(gcloud compute instances list | grep $VM_NAME | awk '{print $5}')

# Assuming VM is running,  ssh to it, start jupyter notebook and return link
gcloud compute ssh $VM_NAME --command "jupyter-notebook --no-browser --port=5000" &
echo Connecting at "http://${IP_ADDRESS}:5000" &
sensible-browser $IP_ADDRESS:5000 &