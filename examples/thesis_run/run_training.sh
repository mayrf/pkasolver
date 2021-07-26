#! /bin/bash


for NUMBER in None {0..4}; do
for MODEL in RFR PLS; do
for PAIR in pair prot deprot; do
for EDGE in edge; do
echo "MODEL=${MODEL}, PAIR=${PAIR},EDGE=${EDGE}, ${NUMBER}"
qsub test.sh ${MODEL} ${PAIR} ${EDGE} ${NUMBER} 
done
done
done
done


for NUMBER in None {0..4}; do
for MODEL in GCN; do
for PAIR in pair prot deprot; do
for EDGE in edge no-edge; do
echo "MODEL=${MODEL}, PAIR=${PAIR},EDGE=${EDGE}, ${NUMBER}"
qsub test.sh ${MODEL} ${PAIR} ${EDGE} ${NUMBER} 
done
done
done
done
