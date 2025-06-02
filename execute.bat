mkdir make
cd make
@REM cmake .. -G "MinGW Makefiles"
@REM make
cmake ..
cmake --build .
Debug\cpu_matrix_multiplication.exe
Debug\gpu_matrix_multiplication.exe
pause