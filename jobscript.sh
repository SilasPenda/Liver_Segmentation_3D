#!/bin/ksh
#$ -q gpu
#$ -m abe
#$ -M sp165339@etu.u-bourgogne.fr
#$ -N kewl_script_executor
#$ -o worklog.log
 
## job start time
printf "\n Job started at : $(date)\n-----\n\n"
 
## what do we need? modules, envs etc.
module purge ## clear out the current module list
# module load python/3.10
export PYTHONUSERBASE=/work/c-2iia/sp165339/LS3D_venv
module load pytorch/2.0.0/gpu
module list

printf "\n --- begin of python execution ---\n"

## set a work dir OR variables with path to feed to python (ARGPARSER, click, etc)
cd /work/c-2iia/sp165339/Liver_Segmentation_3D

## run script
python train.py


printf "\n --- end of python execution ---\n\n"

## job end time
printf "----\njob ended at $(date)\n\n"
