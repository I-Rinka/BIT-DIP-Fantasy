mkdir build
cd build; cmake .. 
cmake --build . --parallel $(nproc)
cp ./DipFantasy ..
cd ..
python3 ./get_predict_json.py

python3 ./judge/lane.py ./judge/predict.json ./judge/groundtruth.json