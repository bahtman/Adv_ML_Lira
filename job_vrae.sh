#!/bin/sh
### General options
### - specify queue --
#BSUB -q gpuv100
### -- set the job Name --
#BSUB -J cnn_segmentation
### -- ask for number of cores (default: 1) --
#BSUB -n 1
### -- Select the resources: 1 gpu in exclusive process mode --
#BSUB -gpu "num=1:mode=exclusive_process"
### -- set walltime limit: hh:mm --  maximum 24 hours for GPU-queues right now
#BSUB -W 24:00
# request 5GB of system-memory
#BSUB -R "rusage[mem=5GB]"
### -- set the email address --
# please uncomment the following line and put in your e-mail address,
# if you want to receive e-mail notifications on a non-default address
##BSUB -u your_email_address
### -- send notification at start --
#BSUB -B
### -- send notification at completion--
#BSUB -N
### -- Specify the output and error file. %J is the job-id --
### -- -o and -e mean append, -oo and -eo mean overwrite --
#BSUB -oo job_out/gpu-%J.out
#BSUB -eo job_out/gpu_%J.err
# -- end of LSF options --

 bash python3 -m wandb agent bahtman/timeseries-clustering-vae/1b0vsvr8