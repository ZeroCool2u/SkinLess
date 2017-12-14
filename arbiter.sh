#!/bin/sh
rm /Shared/bdagroup3/MGoutput.txt
rm /Shared/bdagroup3/FPGoutput.txt
qsub FPG.job
qsub MG.job
clear
qstat -u tlinnemann