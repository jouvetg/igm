# This permits to perform a minimaum aamount of testing

cd real_glacier 
igm_run
cd ..

cd synthetic 
igm_run
cd ..

cd instructed_oggm 
python run_instructed_oggm.py
cd ..


