#!/bin/bash

####
#a) Define slurm job parameters
####

#SBATCH --job-name=DNNChallenge

#resources:

#SBATCH --cpus-per-task=4
# the job can use and see 4 CPUs (from max 24).

#SBATCH --partition=test
# the slurm partition the job is queued to.

#SBATCH --mem-per-cpu=3G
# the job will need 12GB of memory equally distributed on 4 cpus.  (251GB are available in total on one node)

#SBATCH --gres=gpu:1
#the job can use and see 1 GPUs (4 GPUs are available in total on one node)

#SBATCH --time=5
# the maximum time the scripts needs to run

#SBATCH --error=job.%J.err
# write the error output to job.*jobID*.err

#SBATCH --output=job.%J.out
# write the standard output to job.*jobID*.out

#SBATCH --mail-type=ALL
#write a mail if a job begins, ends, fails, gets requeued or stages out

#SBATCH --mail-user=***@uni-tuebingen.de
# your mail address

####
#b) copy all needed data to the jobs scratch folder
####

cp -R ~/challenge/WS1920_challenge_data_set.npy /scratch/$SLURM_JOB_ID/

####
#c) Execute your tensorflow code in a specific singularity container
#d) Write your checkpoints to your home directory, so that you still have them if your job fails
####

singularity exec --nv /common/singularityImages/TCML-Cuda10_0Tensorflow2_0_0.simg python ~/DNN_Challenge/challenge.py --data_dir="/scratch/$SLURM_JOB_ID/" --epochs=5 --batch_size=100

echo DONE!

