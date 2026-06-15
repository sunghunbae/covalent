#!/usr/bin/bash

for xyz in $( ls ../si/*.xyz )
do
    outfile=$( basename $xyz  )
    echo $xyz $outfile

    if [[ "$xyz" == *"carbanion"* ]]; then
    	xtb $xyz --chrg -1 --alpb water --opt tight
    else
    	xtb $xyz --chrg  0 --alpb water --opt tight
    fi

    mv xtbopt.xyz $outfile

done
