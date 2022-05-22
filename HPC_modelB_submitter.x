#!/bin/bash
path=`pwd`

# setup for creating initial conditions
for i in {1..16} ; do

 L=8
 id=$RANDOM # not important -- just to provide a random seed for the start

 # create a tmp file for submission 
 TMPFILE=`mktemp tmp.XXXXXXXXXXXX`
 
 # populate the file with needed script 
 cp run_short.sh $TMPFILE
 echo "julia -t 16 modelB_thermalizer.jl  $id $L $RANDOM  >  /home/jkott/tmp_modelB_${L}_${id}.dat"  >> $TMPFILE
 echo "rm $path/$TMPFILE "  >> $TMPFILE
 
 # submit
 chmod u+x $TMPFILE
 bsub < $TMPFILE

done
