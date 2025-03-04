python refine.py grid --parameters=parameters_test/ISIC.yaml
python refine.py grid --parameters=parameters_test/KVASIR.yaml
python refine.py grid --parameters=parameters_test/baselines/Pothole.yaml
python refine.py grid --parameters=parameters_test/KVASIR.yaml
python refine.py grid --parameters=parameters_test/LungCancer.yaml
python refine.py grid --parameters=parameters_test/ISIC.yaml

# running
python refine.py grid --parameters=parameters_test/distill/KVASIR.yaml --parallel
python refine.py grid --parameters=parameters_test/distill/Pothole.yaml --parallel
python refine.py grid --parameters=parameters_test/distill/Nucleus.yaml --parallel
python refine.py grid --parameters=parameters_test/distill/LungCancer.yaml --parallel
python refine.py grid --parameters=parameters_test/distill/ISIC.yaml --parallel
python refine.py grid --parameters=parameters_test/Nucleus.yaml --parallel --only_create

# to run