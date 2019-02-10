
# tflite-react-native

A React Native library for accessing TensorFlow Lite API. Supports Classification and Object Detection on both iOS and Android.

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

### Object detection:

- SSD MobileNet
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

- Tiny YOLOv2
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

### Release resources:

```
tflite.close();
```

# Demo

Refer to the [example](https://github.com/shaqian/tflite-react-native/tree/master/example).
