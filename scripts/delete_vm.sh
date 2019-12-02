TPU_NAME=$1
VM_NAME=$1

# Delete TPU
echo Deleting TPU $TPU_NAME &
yes Y | gcloud -q compute tpus delete $TPU_NAME &

# # Delete VM
echo Deleting instance $VM_NAME &
yes Y | gcloud -q compute instances delete $VM_NAME &

wait
echo Done