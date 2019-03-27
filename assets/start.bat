call C:\ProgramData\Anaconda3\Scripts\activate.bat tensorflow
cd src
python gesture_control.py ../assets/hands_128_7c.hdf5 7 --run_training --image_size 128

call "C:\Program Files\mRayClient\bin\mRayClient.exe" --hotkey
CMD