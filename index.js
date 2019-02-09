import { NativeModules, Image } from 'react-native';

const { TfliteReactNative } = NativeModules;

class Tflite {
  loadModel(args, callback) {
    TfliteReactNative.loadModel(
      args['model'], 
      args['labels'], 
      args['numThreads'] || 1,
      (error, response) => {
        callback && callback(error, response);
      });
  }

  runModelOnImage(args, callback) {
    TfliteReactNative.runModelOnImage(
      args['path'], 
      args['imageMean'] != null ? args['imageMean'] : 127.5, 
      args['imageStd']  != null ? args['imageStd'] : 127.5,
      args['numResults'] || 5,
      args['threshold'] != null ? args['threshold'] : 0.1,
      (error, response) => {
        callback && callback(error, response);
      });
  }

  detectObjectOnImage(args, callback) {
    TfliteReactNative.detectObjectOnImage(
      args['path'], 
      args['model'] || "SSDMobileNet", 
      args['imageMean'] != null ? args['imageMean'] : 127.5, 
      args['imageStd'] != null ? args['imageStd'] : 127.5,
      args['threshold'] != null ? args['threshold'] : 0.1,
      args['numResultsPerClass'] || 5,
      args['anchors'] || [
        0.57273,
        0.677385,
        1.87446,
        2.06253,
        3.33843,
        5.47434,
        7.88282,
        3.52778,
        9.77052,
        9.16828
      ],
      args['blockSize'] || 32,
      (error, response) => {
        callback && callback(error, response);
      });
  }

  close() {
    TfliteReactNative.close();
  }
}

export default Tflite;
