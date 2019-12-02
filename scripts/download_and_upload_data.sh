#! /bin/bash

#-------------------------------------------
# SET_VARIABLES
#-------------------------------------------
source ./scripts/project_config # PROJECT_NAME WORKDIR BUCKET GIT_USER GIT_CLONE GCP_LOGIN_NAME

python3 scripts/download_and_upload_data.py \
    --dest_dir $BUCKET