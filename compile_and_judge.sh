mkdir build
cd build; cmake .. 
cmake --build . --parallel $(nproc)
cp ./DipFantasy ..
cd ..
python3 ./judge.py