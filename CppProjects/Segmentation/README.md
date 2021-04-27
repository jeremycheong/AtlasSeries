### 概括
程序入口在`example.cpp`中，根据实际情况修改。       

```sh
mkdir build && cd build 
cmake ..
make -j

cd ../out
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/out/folder/path
./main
```