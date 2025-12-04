python refine.py grid --parameters=parameters_test/ISIC.yaml
python refine.py grid --parameters=parameters_test/KVASIR.yaml
python refine.py grid --parameters=parameters_test/baselines/Pothole.yaml
python refine.py grid --parameters=parameters_test/KVASIR.yaml
python refine.py grid --parameters=parameters_test/LungCancer.yaml
python refine.py grid --parameters=parameters_test/ISIC.yaml
python refine.py grid --parameters=parameters_test/distill/KVASIR.yaml --parallel
python refine.py grid --parameters=parameters_test/distill/Pothole.yaml --parallel
python refine.py grid --parameters=parameters_test/distill/Nucleus.yaml --parallel
python refine.py grid --parameters=parameters_test/distill/LungCancer.yaml --parallel
python refine.py grid --parameters=parameters_test/distill/ISIC.yaml --parallel
python refine.py grid --parameters=parameters_test/Nucleus.yaml --parallel

# running
python refine.py grid --parameters=parameters_test/distill/WeedMap.yaml --parallel
###
python refine.py grid --parameters=parameters_test/ISIC.yaml --parallel
python refine.py grid --parameters=parameters_test/KVASIR.yaml --parallel
python refine.py grid --parameters=parameters_test/LungCancer.yaml --parallel
python refine.py grid --parameters=parameters_test/Nucleus.yaml --parallel
python refine.py grid --parameters=parameters_test/Pothole.yaml --parallel
python refine.py grid --parameters=parameters_test/WeedMap.yaml --parallel
##
python refine.py grid --parameters=parameters_test/baselines/ISIC.yaml --parallel
python refine.py grid --parameters=parameters_test/baselines/KVASIR.yaml --parallel
python refine.py grid --parameters=parameters_test/baselines/LungCancer.yaml --parallel
python refine.py grid --parameters=parameters_test/baselines/Nucleus.yaml --parallel
python refine.py grid --parameters=parameters_test/baselines/Pothole.yaml --parallel
python refine.py grid --parameters=parameters_test/baselines/WeedMap.yaml --parallel

# to run
python refine.py grid --parameters=parameters_test/distill/KVASIR.yaml --parallel

----
python refine.py grid --parameters=parameters_test/distill/ISIC.yaml --parallel
python refine.py grid --parameters=parameters_test/ISIC.yaml --parallel
python refine.py grid --parameters=parameters_test/distill/KVASIR.yaml --parallel

python refine.py grid --parameters=parameters_test/distill/LungCancer.yaml --parallel
python refine.py grid --parameters=parameters_test/distill/Nucleus.yaml --parallel
python refine.py grid --parameters=parameters_test/distill/Pothole.yaml --parallel
python refine.py grid --parameters=parameters_test/distill/WeedMap.yaml --parallel

---

python speed.py grid --parameters=parameters_test/speed.yaml --parallel