#!/bin/tcsh
#BSUB -W 1440
#BSUB -n 1
#BSUB -q gpu
#BSUB -R "select[p100]"
#BSUB -gpu "num=1:mode=shared:mps=no"
#BSUB -o tmp/out.%J
#BSUB -e tmp/err.%J
setenv JULIA_DEPOT_PATH /share/tmschaef/jkott/julia
module load julia
module load cuda/10.2
