# waifu2x-metal

## How to Compile and Run

```
git clone https://github.com/safx/waifu2x-metal.git
cd waifu2x-metal
xcodebuild -project waifu2x-metal.xcodeproj
cd build/Release
cp ../../model/scale2.0x_model.json .
./waifu2x-metal Sample.jpg
```
