
# tflite-react-native

A React Native library for accessing TensorFlow Lite API. Supports Classification, Object Detection, Deeplab and PoseNet on both iOS and Android.

### Table of Contents

- [Installation](#Installation)
- [Usage](#Usage)
    - [Image Classification](#Image-Classification)
    - [Object Detection](#Object-Detection)
      - [SSD MobileNet](#SSD-MobileNet)
      - [YOLO](#Tiny-YOLOv2)
    - [Deeplab](#Deeplab)
    - [PoseNet](#PoseNet)
- [Example](#Example)

## Installation

`$ npm install tflite-react-native --save`

### iOS (only)

TensorFlow Lite is installed using CocoaPods:

1. Initialize Pod:
	```
	cd ios
	pod init
	```

2. Open Podfile and add:
	```
	target '[your project's name]' do
		pod 'TensorFlowLite', '1.12.0'
	end
	```

3. Install:
	```
	pod install
	```

### Automatic link

`$ react-native link tflite-react-native`

### Manual link

#### iOS

1. In XCode, in the project navigator, right click `Libraries` ➜ `Add Files to [your project's name]`
2. Go to `node_modules` ➜ `tflite-react-native` and add `TfliteReactNative.xcodeproj`
3. In XCode, in the project navigator, select your project. Add `libTfliteReactNative.a` to your project's `Build Phases` ➜ `Link Binary With Libraries`
4. Run your project (`Cmd+R`)<

#### Android

1. Open up `android/app/src/main/java/[...]/MainApplication.java`
  - Add `import com.reactlibrary.TfliteReactNativePackage;` to the imports at the top of the file
  - Add `new TfliteReactNativePackage()` to the list returned by the `getPackages()` method
2. Append the following lines to `android/settings.gradle`:
    ```
    include ':tflite-react-native'
    project(':tflite-react-native').projectDir = new File(rootProject.projectDir,   '../node_modules/tflite-react-native/android')
    ```
3. Insert the following lines inside the dependencies block in `android/app/build.gradle`:
    ```
      compile project(':tflite-react-native')
    ```

## Add models to the project

### iOS

In XCode, right click on the project folder, click **Add Files to "xxx"...**, select the model and label files.

### Android

1. In Android Studio (1.0 & above), right-click on the `app` folder and go to **New > Folder > Assets Folder**. Click **Finish** to create the assets folder.

2. Place the model and label files at `app/src/main/assets`.

2. In `android/app/build.gradle`, add the following setting in `android` block.

```
    aaptOptions {
        noCompress 'tflite'
    }
```

## Usage

```javascript
import Tflite from 'tflite-react-native';

let tflite = new Tflite();
```

### Load model:

```javascript
tflite.loadModel({
  model: 'models/mobilenet_v1_1.0_224.tflite',// required
  labels: 'models/mobilenet_v1_1.0_224.txt',  // required
  numThreads: 1,                              // defaults to 1  
},
(err, res) => {
  if(err)
    console.log(err);
  else
    console.log(res);
});
```

### Image classification:

```javascript
tflite.runModelOnImage({
  path: imagePath,  // required
  imageMean: 128.0, // defaults to 127.5
  imageStd: 128.0,  // defaults to 127.5
  numResults: 3,    // defaults to 5
  threshold: 0.05   // defaults to 0.1
},
(err, res) => {
  if(err)
    console.log(err);
  else
    console.log(res);
});
```

- Output fomart:
```
{
  index: 0,
  label: "person",
  confidence: 0.629
}
```

### Object detection:

#### SSD MobileNet

```javascript
tflite.detectObjectOnImage({
  path: imagePath,
  model: 'SSDMobileNet',
  imageMean: 127.5,
  imageStd: 127.5,
  threshold: 0.3,       // defaults to 0.1
  numResultsPerClass: 2,// defaults to 5
},
(err, res) => {
  if(err)
    console.log(err);
  else
    console.log(res);
});
```

#### Tiny YOLOv2

```javascript
tflite.detectObjectOnImage({
  path: imagePath,
  model: 'YOLO',
  imageMean: 0.0,
  imageStd: 255.0,
  threshold: 0.3,        // defaults to 0.1
  numResultsPerClass: 2, // defaults to 5
  anchors: [...],        // defaults to [0.57273,0.677385,1.87446,2.06253,3.33843,5.47434,7.88282,3.52778,9.77052,9.16828]
  blockSize: 32,         // defaults to 32 
},
(err, res) => {
  if(err)
    console.log(err);
  else
    console.log(res);
});
```

- Output fomart:

`x, y, w, h` are between [0, 1]. You can scale `x, w` by the width and `y, h` by the height of the image.

```
{
  detectedClass: "hot dog",
  confidenceInClass: 0.123,
  rect: {
    x: 0.15,
    y: 0.33,
    w: 0.80,
    h: 0.27
  }
}
```

### Deeplab

```javascript
tflite.runSegmentationOnImage({
  path: imagePath,
  imageMean: 127.5,      // defaults to 127.5
  imageStd: 127.5,       // defaults to 127.5
  labelColors: [...],    // defaults to https://github.com/shaqian/tflite-react-native/blob/master/index.js#L59
  outputType: "png",     // defaults to "png"
},
(err, res) => {
  if(err)
    console.log(err);
  else
    console.log(res);
});
```

- Output format:
  
  The output of Deeplab inference is Uint8List type. Depending on the `outputType` used, the output is:

  - (if outputType is png) byte array of a png image 

  - (otherwise) byte array of r, g, b, a values of the pixels 


### PoseNet

> Model is from [StackOverflow thread](https://stackoverflow.com/a/55288616).

```javascript
tflite.runPoseNetOnImage({
  path: imagePath,
  imageMean: 127.5,      // defaults to 127.5
  imageStd: 127.5,       // defaults to 127.5
  numResults: 3,         // defaults to 5
  threshold: 0.8,        // defaults to 0.5
  nmsRadius: 20,         // defaults to 20 
},
(err, res) => {
  if(err)
    console.log(err);
  else
    console.log(res);
});
```

- Output format:

`x, y` are between [0, 1]. You can scale `x` by the width and `y` by the height of the image.

```
[ // array of poses/persons
  { // pose #1
    score: 0.6324902,
    keypoints: {
      0: {
        x: 0.250,
        y: 0.125,
        part: nose,
        score: 0.9971070
      },
      1: {
        x: 0.230,
        y: 0.105,
        part: leftEye,
        score: 0.9978438
      }
      ......
    }
  },
  { // pose #2
    score: 0.32534285,
    keypoints: {
      0: {
        x: 0.402,
        y: 0.538,
        part: nose,
        score: 0.8798978
      },
      1: {
        x: 0.380,
        y: 0.513,
        part: leftEye,
        score: 0.7090239
      }
      ......
    }
  },
  ......
]
```

### Release resources:

```
tflite.close();
```

## Example

Refer to the [example](https://github.com/shaqian/tflite-react-native/tree/master/example).
