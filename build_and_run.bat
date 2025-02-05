cd build
cmake ../src
cd ..
cmake --build build --config Release
move .\build\bin\Release\ray_tracer.exe .
ray_tracer.exe