# Build and Run 
## Build

```powershell
# Ran from repo root
cmake -S nn -B nn/build --toolchain C:/vcpkg/scripts/buildsystems/vcpkg.cmake -DVCPKG_TARGET_TRIPLET=x64-windows -DCMAKE_PREFIX_PATH=C:/vcpkg/installed/x64-windows
cmake --build nn/build --config Debug
```
