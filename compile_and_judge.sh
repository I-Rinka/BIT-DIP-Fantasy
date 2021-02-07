mkdir build
cd build; cmake .. 
cmake --build . --parallel $(nproc)
cp ./DipFantasy ..
cd ..
python3 ./judge.py

python3 ./judge/lane.py ./judge/predict.json ./judge/groundtruth.json