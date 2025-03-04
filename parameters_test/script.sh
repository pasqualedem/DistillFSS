python refine.py grid --parameters=parameters_test/ISIC.yaml
python refine.py grid --parameters=parameters_test/KVASIR.yaml
python refine.py grid --parameters=parameters_test/baselines/Pothole.yaml

# running
python refine.py grid --parameters=parameters_test/KVASIR.yaml
python refine.py grid --parameters=parameters_test/LungCancer.yaml
python refine.py grid --parameters=parameters_test/ISIC.yaml
python refine.py grid --parameters=parameters_test/Nucleus.yaml

# to run
python refine.py grid --parameters=parameters_test/distill/WeedMap.yaml
python refine.py grid --parameters=parameters_test/distill/KVASIR.yaml
python refine.py grid --parameters=parameters_test/distill/LungCancer.yaml
python refine.py grid --parameters=parameters_test/distill/ISIC.yaml
python refine.py grid --parameters=parameters_test/distill/Nucleus.yaml 