#!/bin/sh

#conda install pymc3 #version==3.7

for i in {0..3}; do for prior in dir LN; do for dataset in basic_vocabulary_data.csv caused_motion_data.csv reciprocal_data.csv topological_relations_data.csv; do for mode in TRUE FALSE; do
	#run model for one chain
	python3 DPMM_pm.py $dataset $prior $i $mode
	python3 eval_posterior.py $dataset $prior $i $mode
	#screen -dmS $dataset$prior$i$mode
	#screen -S $dataset$prior$i$mode -p 0 -X stuff "python3 DPMM_pm.py $dataset $prior $i $mode\n"
	#run model with held-out data
	python3 DPMM_pm.py $dataset $prior 0 FALSE hold_out $i
	#screen -dmS holdout$dataset$prior$i
	#screen -S holdout$dataset$prior$i -p 0 -X stuff "python3 DPMM_pm.py $dataset $prior 0 FALSE hold_out $i\n"
done
done
done

for prior in dir LN; do for dataset in basic_vocabulary_data.csv caused_motion_data.csv reciprocal_data.csv topological_relations_data.csv; do
	python3 held_out_loglik.py $dataset $prior
done
done


python3 eval_posterior.py
python3 make_v_measure.py
make_graphics.py
python3 predict.py
Rscript unpredictability.R
