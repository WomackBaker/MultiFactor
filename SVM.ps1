$gen_users = 100
$gen_rows = 1000
$data_samples = 10000
$attackers = 2000

cd .\GenerateData
python.exe .\generate.py $gen_users $gen_rows
cd ..\GAN
python.exe .\gan.py $data_samples
cd ..\SVM
python.exe .\split.py $attackers
python.exe .\svm.py
python.exe .\fpr_fnr.py
cd ..\