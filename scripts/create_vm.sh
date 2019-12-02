#! /bin/bash

#-------------------------------------------
# SET_VARIABLES
#-------------------------------------------
source ./scripts/project_config # PROJECT_NAME WORKDIR BUCKET GIT_USER GIT_CLONE GCP_LOGIN_NAME
TPU_NAME=$1
VM_NAME=$1
STARTUP_SCRIPT="echo export TPU_NAME=$TPU_NAME > /etc/profile.d/tpu-env.sh"
SSH_SCRIPT="ssh-keyscan github.com >> ~/.ssh/known_hosts
git clone $GIT_CLONE
mkdir -p ~/.jupyter"

#-------------------------------------------
# CREATE TPU
#-------------------------------------------
echo --------------------------------------- &
echo Creating TPU $TPU_NAME &
echo --------------------------------------- &
yes Y | ctpu up --name=$TPU_NAME \
    --tf-version='1.15' \
    --tpu-size=v2-8 \
    --tpu-only &

#-------------------------------------------
# CREATE VM
#-------------------------------------------
echo --------------------------------------- &
echo Creating instance $VM_NAME &
echo --------------------------------------- &
gcloud compute instances create $VM_NAME \
    --image template-image \
    --machine-type n1-standard-4 \
    --scopes cloud-platform,userinfo-email \
    --boot-disk-size=300 \
    --metadata proxy-user-mail="$GCP_LOGIN_NAME",install-nvidia-driver=True,startup-script="$STARTUP_SCRIPT" \
    --tags http-server,https-server &

# -------------------------------------------
# CLONE REPOSITORY AND SET JUPYTER CONFIGS
# -------------------------------------------
wait
echo -----------------------------------------------
echo Clonning repository and setting jupyter configs to VM $VM_NAME
echo -----------------------------------------------
for i in 1 2 3 4 5; do
    gcloud compute ssh $VM_NAME \
        --ssh-key-file=~/.ssh/id_rsa --ssh-flag="-A" \
        --command "$SSH_SCRIPT" &&
        gcloud compute scp --recurse ./jupyter_notebook_config* $VM_NAME:~/.jupyter/ &&
        break
    echo Attempt $i failed: Retrying...
    sleep 5
done
