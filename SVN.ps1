cd .\GenerateData
python.exe .\generate.py
cd ..\GAN
python.exe .\gan.py
cd ..\SVN
python.exe .\split.py
python.exe .\svm.py
cd ..\