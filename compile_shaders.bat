echo "compiling glsl shaders..."
python ./scripts/compile_shaders.py ./data/shaders

echo "generating cpp header / source file including all shaders..."
./cmake-build-debug/scripts/Debug/scripts.exe "C:\Users\joebh\CLionProjects\vglib.github.io\data" "C:\Users\joebh\CLionProjects\vglib.github.io\VulkanBase\include" "C:\Users\joebh\CLionProjects\vglib.github.io\VulkanBase\source" glsl_shaders "C:\Users\joebh\CLionProjects\vglib.github.io"
