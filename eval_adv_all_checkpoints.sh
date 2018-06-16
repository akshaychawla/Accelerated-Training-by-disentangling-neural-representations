echo "Processing Checkpoints in folder: $1"

CHECKPOINTS=$1
CHECKPOINTS="$CHECKPOINTS*"
echo "=======================================" > RESULTS_adv_allchecks.txt
for filename in $CHECKPOINTS;
do
    python fgsm.py $filename 0.05 >> RESULTS_adv_allchecks.txt 
    echo "=======================================" >> RESULTS_adv_allchecks.txt
done 

