# Automatically show list with make command
.PHONY: list
list:
	@$(MAKE) -pRrq -f $(lastword $(MAKEFILE_LIST)) : 2>/dev/null | \
	awk -v RS= -F: '/^# File/,/^# Finished Make data base/ {if ($$1 \
	!~ "^[#.]") {print $$1}}' | sort | egrep -v -e '^[^[:alnum:]]' -e \
	'^$@$$'

######################################################################
# SET VARIABLES
######################################################################
CUR_DIR=$(shell basename $(CURDIR))
PROJECT_NAME=$(CUR_DIR)
ZONE=us-central1-f

GITHUB_REPOSITORY=$(PROJECT_NAME)
GITHUB_USER=lucaskawazoi

GCP_LOGIN_NAME=lucas.key.kawazoi@gmail.com

######################################################################
# VIRTUAL MACHINES
######################################################################
LOCALHOST_CHANNEL=8881
INSTANCE_TYPE=n1-highmem-8 # budget: "n1-highmem-4"
IMAGE_FAMILY=tf-1-13-cpu
IMAGE_PROJECT=deeplearning-platform-release 
TPU_NAME=$(PROJECT_NAME)
STARTUP_SCRIPT="echo export TPU_NAME=$(TPU_NAME) > /etc/profile.d/tpu-env.sh; \
				git clone git@github.com:$(GITHUB_USER)/$(GITHUB_REPOSITORY).git"
# FAMILY AND PROJECTS...
# - pytorch gpu: 		deeplearning-platform-release	pytorch-latest-gpu
# - pytorch non-gpu: 	deeplearning-platform-release 	pytorch-latest-cpu
# - pytorch tpu: 		ml-images 						torch-xla
# - tensorflow-gpu: 	deeplearning-platform-release	tf-latest-gpu
# - tensorflow-cpu: 	deeplearning-platform-release	tf-latest-cpu

gcc-create:
	#-----------------------------------------------------------------
	# gcc-create: Creating a new instance named $(PROJECT_NAME)
	#-----------------------------------------------------------------
	gcloud compute instances create $(PROJECT_NAME) \
	        --zone=$(ZONE) \
	        --image-family=$(IMAGE_FAMILY) \
	        --image-project=$(IMAGE_PROJECT) \
	        --maintenance-policy=TERMINATE \
	        --machine-type=$(INSTANCE_TYPE) \
	        --boot-disk-size=200GB \
	        --no-boot-disk-auto-delete \
	        --scopes cloud-platform,userinfo-email \
	        --metadata proxy-user-mail=${GCP_LOGIN_NAME},install-nvidia-driver=True,startup-script=${STARTUP_SCRIPT}
# 	        --preemptible
# 	        --accelerator="type=nvidia-tesla-p100,count=1" \ 
# 							budget: 'type=nvidia-tesla-k80,count=1'
# 	        --disk name=$(PROJECT_NAME),boot=yes,auto-delete=no \

gcc-start:
	#-----------------------------------------------------------------
	# gcc-start
	#-----------------------------------------------------------------
	gcloud compute instances start $(PROJECT_NAME)

gcc-ssh:
	#-----------------------------------------------------------------
	# gcc-ssh
	#-----------------------------------------------------------------
	gcloud compute ssh --zone=$(ZONE) jupyter@$(PROJECT_NAME) \
	--ssh-key-file=~/.ssh/id_rsa --ssh-flag="-A" \
	-- -L $(LOCALHOST_CHANNEL):localhost:8080

gcc-stop:
	#-----------------------------------------------------------------
	# gcc-stop
	#-----------------------------------------------------------------
	gcloud compute instances stop $(PROJECT_NAME)

gcc-delete-instance:
	#-----------------------------------------------------------------
	# gcc-delete-instance
	#-----------------------------------------------------------------
	gcloud compute instances delete $(PROJECT_NAME)

gcc-delete-disk:
	#-----------------------------------------------------------------
	# gcc-delete-disk
	#-----------------------------------------------------------------
	gcloud compute disks delete $(PROJECT_NAME)

gcc-list: 
	#-----------------------------------------------------------------
	# gcc-list
	#-----------------------------------------------------------------
	gcloud compute instances list
	#-----------------------------------------------------------------
	gcloud compute disks list
	#-----------------------------------------------------------------
	gcloud compute snapshots list
	#-----------------------------------------------------------------
	gcloud compute tpus list

gcc-tpu-create:
	#-----------------------------------------------------------------
	# gcc-tpu-create
	#-----------------------------------------------------------------
	gcloud compute tpus create $(PROJECT_NAME) \
	--zone=$(ZONE) \
	--network=default \
	--range=10.2.3.0 \
	--version='1.13' \
	--accelerator-type=v2-8

gcc-tpu-config-env-variables:
	#-----------------------------------------------------------------
	# gcc-tpu-config-env-variables  - TODO
	# (https://cloud.google.com/tpu/docs/tutorials/transformer-pytorch)
	#-----------------------------------------------------------------
	export TPU_IP_ADDRESS=$(shell gcloud compute tpus list | grep $(PROJECT_NAME) | awk '{printf $$4}')
	export XRT_TPU_CONFIG="tpu_worker;0;$(shell gcloud compute tpus list | grep $(PROJECT_NAME) | awk '{printf $$4}'):8470"
	export TPU_NAME=$(PROJECT_NAME)

gcc-tpu-delete:
	#-----------------------------------------------------------------
	# gcc-tpu-delete
	#-----------------------------------------------------------------
	gcloud compute tpus delete $(PROJECT_NAME)

gcc-tpu-start:
	#-----------------------------------------------------------------
	# gcc-tpu-start
	#-----------------------------------------------------------------
	gcloud compute tpus start $(PROJECT_NAME)

gcc-tpu-stop:
	#-----------------------------------------------------------------
	# gcc-tpu-stop
	#-----------------------------------------------------------------
	gcloud compute tpus stop $(PROJECT_NAME)

	

######################################################################
# STORAGE
######################################################################
STORAGE_CLASS=standard
REGION=us-central1
BUCKET=$(PROJECT_NAME)

gcs-create:
	#-----------------------------------------------------------------
	# gcs-create: Create bucket named $(BUCKET)
	#-----------------------------------------------------------------
	gsutil mb -c $(STORAGE_CLASS) -l $(REGION) gs://$(BUCKET)/

gcs-upload-all:
	#-----------------------------------------------------------------
	# gcs-upload: Upload ./data/ to gs://$(BUCKET)/data
	#-----------------------------------------------------------------
	mkdir -p data
	gsutil -m rsync -r data gs://$(BUCKET)/data

gcs-upload-minimum:
	#-----------------------------------------------------------------
	# gcs-upload-minimum: Upload data from data/raw/compressed 
	# data/output  data/models to gs://$(BUCKET)
	#-----------------------------------------------------------------
	mkdir -p data/raw/compressed data/output data/models
	gsutil -m rsync -r data/raw/compressed gs://$(BUCKET)/data/raw/compressed
	gsutil -m rsync -r data/output gs://$(BUCKET)/data/output
	gsutil -m rsync -r data/models gs://$(BUCKET)/data/models

gcs-download: aux-gcs-make-data-folder-in-bucket
	#-----------------------------------------------------------------
	# gcs-download: Download data from gs://$(BUCKET)/data to /data
	#-----------------------------------------------------------------
	mkdir -p data
	gsutil -m rsync -r gs://$(BUCKET)/data data

gcs-delete:
	#-----------------------------------------------------------------
	# gcs-delete: Delete local /data and bucket gs://$(BUCKET)/data
	#-----------------------------------------------------------------
	gsutil -m rm -r gs://$(BUCKET)

gcs-make-public:
	#-----------------------------------------------------------------
	# gcs-make-public: Make bucket gs://$(BUCKET)/data public
	#-----------------------------------------------------------------
	gsutil iam ch allUsers:objectViewer gs://$(BUCKET)

gcs-list:
	#-----------------------------------------------------------------
	# gcs-list: List files in bucket gs://$(BUCKET)
	#-----------------------------------------------------------------
	gsutil ls -L -b gs://$(BUCKET)
	#-----------------------------------------------------------------
	gsutil ls -r -lh  gs://$(BUCKET)


gcs-size:
	#-----------------------------------------------------------------
	# gcs-size: Get size of bucket gs://$(BUCKET)/data
	#-----------------------------------------------------------------
	gsutil du -s -h gs://$(BUCKET)


######################################################################
# ENVIRONMENT
######################################################################
conda-create-empty:
	#-----------------------------------------------------------------
	# conda-create-empty
	#-----------------------------------------------------------------
	conda create --name $(PROJECT_NAME)

conda-create-clone-base:
	#-----------------------------------------------------------------
	# conda-create-clone-base
	#-----------------------------------------------------------------
	conda create --name $(PROJECT_NAME) --clone base

conda-create-from-file:
	#-----------------------------------------------------------------
	# conda-create-from-file
	#-----------------------------------------------------------------
	conda env create --name $(PROJECT_NAME) --file environment.yml

conda-add-kernel-to-jupyter:
	#-----------------------------------------------------------------
	# conda-add-kernel-to-jupyter
	#-----------------------------------------------------------------
	conda install -y -c anaconda ipykernel
	python -m ipykernel install --user --name $(PROJECT_NAME)

conda-activate:
	#-----------------------------------------------------------------
	# conda-activate
	# >>> Not working, please use 'conda activate $(PROJECT_NAME)'
	#-----------------------------------------------------------------

conda-deactivate:
	#-----------------------------------------------------------------
	# conda-deactivate
	# >>> Not working, please use 'conda deactivate'
	#-----------------------------------------------------------------

conda-export:
	#-----------------------------------------------------------------
	# conda-export
	# >>> Export active conda environment to environment.yml
	#-----------------------------------------------------------------
	conda env export > environment.yml


conda-update:
	#-----------------------------------------------------------------
	# conda-update
	#-----------------------------------------------------------------
	conda env update -n $(PROJECT_NAME) -f environment.yml

conda-delete:
	#-----------------------------------------------------------------
	# conda-delete
	#-----------------------------------------------------------------
	conda remove --name $(PROJECT_NAME) --all

conda-list:
	#-----------------------------------------------------------------
	# conda-list
	#-----------------------------------------------------------------
	conda env list

conda-install-basics:
	#-----------------------------------------------------------------
	# conda-install-basics
	#-----------------------------------------------------------------
	conda install -y -c anaconda pip
	conda install -y -c anaconda pandas
	conda install -y -c anaconda numpy
	conda install -y -c anaconda jupyter
	conda install -y -c anaconda ipykernel
	conda install -y -c conda-forge matplotlib
	conda install -y -c anaconda seaborn


######################################################################
# FOLDER STRUCTURE
######################################################################
makefile-update:
	#-----------------------------------------------------------------
	# update-makefile
	#-----------------------------------------------------------------
	git add Makefile
	git commit -m "Makefile updated"
	git push
	cp Makefile data/aux
	gsutil cp Makefile gs://lkk-kaw/data/aux

makefile-download:
	gsutil cp gs://lkk-kaw/data/aux/Makefile .

readme-update:
	#-----------------------------------------------------------------
	# readme-update
	#-----------------------------------------------------------------
	git add README.md
	git commit -m "README updated"
	git push

folder-structure:
	mkdir -p data data/raw/compressed data/raw/uncompressed data/processed data/output data/models nbs src
	gsutil cp gs://lkk-kaw/data/aux/.gitignore .
	gsutil cp gs://lkk-kaw/data/templates/pynb/empty.ipynb nbs
	gsutil cp gs://lkk-kaw/data/templates/py/utils.py src

######################################################################
# KAGGLE
######################################################################
kaggle-json:
	mkdir -p $(HOME)/.kaggle
	touch $(HOME)/.kaggle/kaggle.json
	vim $(HOME)/.kaggle/kaggle.json

kaggle-download-dataset:
	python3 src/kaggle_download.py 


######################################################################
# AUX FUNCTIONS
######################################################################
aux-gcs-make-data-folder-in-bucket:
	mkdir -p dummy
	touch dummy/.dummy
	gsutil -m rsync -r dummy gs://$(BUCKET)/data
	rm -r dummy

aux-standardize-encoding-data-raw-compressed:
	convmv --notest -r -f ISO-8859-1 -t UTF-8 data/raw/compressed

aux-move-project-to-hd1:
	mv ../$(CUR_DIR) /hd1/$(CUR_DIR)

aux-printvars:
	@$(foreach V,$(sort $(.VARIABLES)), $(if $(filter-out environment% default automatic, $(origin $V)),$(warning $V=$($V) ($(value $V)))))

aux-check-memory:
	free -m -h

aux-check-disk:
	df -h
	sudo du -sh ~/*

######################################################################
# THE END
######################################################################


######################################################################
# PYTHON - 
######################################################################
# py_download_data:
# 	#-----------------------------------------------------------------
# 	# download_data
# 	#-----------------------------------------------------------------
# 	echo "TODO: create download_data.py to first time download"

# py_process_data:
# 	#-----------------------------------------------------------------
# 	# process_data
# 	#-----------------------------------------------------------------
# 	echo "TODO: create process_data.py"


######################################################################
# PROJECT
######################################################################
# project-create:  		\
# 	gcc-create 			\
# 	gcs-create 			\
# 	conda-create-clone-base

# project-save:			\
# 	conda-export		\
# 	git-push			\

# TODO-project-backup:	\ 
# 	conda-export		\
# 	gcs-sync-project 	\
# 	gcs-backup-cold-storage

# TODO-project-archive:

# project-delete:  		\
# 	project-backup		\
# 	gcc-delete 			\
# 	gcs-delete 			\
# 	conda-export 		\
# 	conda-delete
# 	echo "Remember to archive Github repository and local directory"


######################################################################
# COMBOS - DAILY SHORTCUTS
######################################################################
# combo-start-day-local:	\
# 	git-pull 			\
# 	gcc-start 			\
# 	gcs-download 		\
# 	conda-update 		\
# 	conda-activate

# combo-end-day-local: 	\
# 	gcc-stop 			\
# 	gcs-upload	 		\
# 	conda-export 		\
# 	conda-deactivate
# 	echo "Remember to push changes to github"

# TODO-combo-start-day-vm:

# TODO-combo-end-day-vm:

# combo-list: 			\
# 	gcc-list 			\
# 	gcs-list 			\
# 	conda-list