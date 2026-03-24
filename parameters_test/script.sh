# DistillFSS
python refine.py grid --parameters=parameters_test/distill/KVASIR.yaml --parallel
python refine.py grid --parameters=parameters_test/distill/Pothole.yaml --parallel
python refine.py grid --parameters=parameters_test/distill/Nucleus.yaml --parallel
python refine.py grid --parameters=parameters_test/distill/LungCancer.yaml --parallel
python refine.py grid --parameters=parameters_test/distill/ISIC.yaml --parallel
python refine.py grid --parameters=parameters_test/distill/Industrial.yaml --parallel
# TransferFSS
python refine.py grid --parameters=parameters_test/ISIC.yaml --parallel
python refine.py grid --parameters=parameters_test/KVASIR.yaml --parallel
python refine.py grid --parameters=parameters_test/LungCancer.yaml --parallel
python refine.py grid --parameters=parameters_test/Nucleus.yaml --parallel
python refine.py grid --parameters=parameters_test/Pothole.yaml --parallel
python refine.py grid --parameters=parameters_test/WeedMap.yaml --parallel
python refine.py grid --parameters=parameters_test/Industrial.yaml --parallel
# Baselines
python refine.py grid --parameters=parameters_test/baselines/ISIC.yaml --parallel
python refine.py grid --parameters=parameters_test/baselines/KVASIR.yaml --parallel
python refine.py grid --parameters=parameters_test/baselines/LungCancer.yaml --parallel
python refine.py grid --parameters=parameters_test/baselines/Nucleus.yaml --parallel
python refine.py grid --parameters=parameters_test/baselines/Pothole.yaml --parallel
python refine.py grid --parameters=parameters_test/baselines/WeedMap.yaml --parallel
python refine.py grid --parameters=parameters_test/baselines/Industrial.yaml --parallel

# Speed
python speed.py grid --parameters=parameters_test/speed.yaml --parallel