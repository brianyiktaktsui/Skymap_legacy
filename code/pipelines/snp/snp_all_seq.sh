#! /bin/bash
# Which shell to use
#$ -S /bin/bash
# Transfer all variables to job script (e.g. PATH, LD_LIBRARY_PATH, etc.)
#$ -V
# Queue to schedule jobs to
#$ -l h_rt=8:00:00
## -l u14
#$ -l h_vmem=2g
#$ -l long
#$ -l tmpfree=2T
#$ -l h=!(nrnb-5-0|nrnb-5-1|nrnb-5-2|nrnb-5-3|nrnb-5-4|nrnb-5-5|nrnb-5-6)
# Directory to send stdout and stderr
#$ -o /dev/null
#$ -e /dev/null
## -o /cellar/users/btsui/Data/sgeOut/
## -e /cellar/users/btsui/Data/sgeOut/
# Run in current working directory
#$ -cwd
# Array job 14175
#$ -t 1-6
## -p -0
#$ -r y
#$ -tc 20
#$ -pe smp 3
#SRR_FTP="/cellar/users/btsui/Data/SRA/META/processing.srr"
hostname
/cellar/users/btsui/anaconda2/bin/python -u /cellar/users/btsui/Project/METAMAP/notebook/RapMapTest/Pipelines/snp/run_one.py $SGE_TASK_ID
