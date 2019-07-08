import React, { Component } from 'react';
import { Platform, StyleSheet, Image, Text, View, TouchableOpacity } from 'react-native';
import Tflite from 'tflite-react-native';
import ImagePicker from 'react-native-image-picker';

let tflite = new Tflite();

const height = 350;
const width = 350;
const blue = "#25d5fd";
const mobile = "MobileNet";
const ssd = "SSD MobileNet";
const yolo = "Tiny YOLOv2";
const deeplab = "Deeplab";
const posenet = "PoseNet";

type Props = {};
export default class App extends Component<Props> {
  constructor(props) {
    super(props);
    this.state = {
      model: null,
      source: null,
      imageHeight: height,
      imageWidth: width,
      recognitions: []
    };
  }

  onSelectModel(model) {
    this.setState({ model });
    switch (model) {
      case ssd:
        var modelFile = 'models/ssd_mobilenet.tflite';
        var labelsFile = 'models/ssd_mobilenet.txt';
        break;
      case yolo:
        var modelFile = 'models/yolov2_tiny.tflite';
        var labelsFile = 'models/yolov2_tiny.txt';
        break;
      case deeplab:
        var modelFile = 'models/deeplabv3_257_mv_gpu.tflite';
        var labelsFile = 'models/deeplabv3_257_mv_gpu.txt';
        break;
      case posenet:
        var modelFile = 'models/posenet_mv1_075_float_from_checkpoints.tflite';
        var labelsFile = '';
        break;
      default:
        var modelFile = 'models/mobilenet_v1_1.0_224.tflite';
        var labelsFile = 'models/mobilenet_v1_1.0_224.txt';
    }
    tflite.loadModel({
      model: modelFile,
      labels: labelsFile,
    },
      (err, res) => {
        if (err)
          console.log(err);
        else
          console.log(res);
      });
  }

  onSelectImage() {
    const options = {
      title: 'Select Avatar',
      customButtons: [{ name: 'fb', title: 'Choose Photo from Facebook' }],
      storageOptions: {
        skipBackup: true,
        path: 'images',
      },
    };
    ImagePicker.launchImageLibrary(options, (response) => {
      if (response.didCancel) {
        console.log('User cancelled image picker');
      } else if (response.error) {
        console.log('ImagePicker Error: ', response.error);
      } else if (response.customButton) {
        console.log('User tapped custom button: ', response.customButton);
      } else {
        var path = Platform.OS === 'ios' ? response.uri : 'file://' + response.path;
        var w = response.width;
        var h = response.height;
        this.setState({
          source: { uri: path },
          imageHeight: h * width / w,
          imageWidth: width
        });

        switch (this.state.model) {
          case ssd:
            tflite.detectObjectOnImage({
              path,
              threshold: 0.2,
              numResultsPerClass: 1,
            },
              (err, res) => {
                if (err)
                  console.log(err);
                else
                  this.setState({ recognitions: res });
              });
            break;

          case yolo:
            tflite.detectObjectOnImage({
              path,
              model: 'YOLO',
              imageMean: 0.0,
              imageStd: 255.0,
              threshold: 0.4,
              numResultsPerClass: 1,
            },
              (err, res) => {
                if (err)
                  console.log(err);
                else
                  this.setState({ recognitions: res });
              });
            break;

          case deeplab:
            tflite.runSegmentationOnImage({
              path
            },
              (err, res) => {
                if (err)
                  console.log(err);
                else
                  this.setState({ recognitions: res });
              });
            break;

          case posenet:
            tflite.runPoseNetOnImage({
              path,
              threshold: 0.8
            },
              (err, res) => {
                if (err)
                  console.log(err);
                else
                  this.setState({ recognitions: res });
              });
            break;

          default:
            tflite.runModelOnImage({
              path,
              imageMean: 128.0,
              imageStd: 128.0,
              numResults: 3,
              threshold: 0.05
            },
              (err, res) => {
                if (err)
                  console.log(err);
                else
                  this.setState({ recognitions: res });
              });
        }
      }
    });
  }

  renderResults() {
    const { model, recognitions, imageHeight, imageWidth } = this.state;
    switch (model) {
      case ssd:
      case yolo:
        return recognitions.map((res, id) => {
          var left = res["rect"]["x"] * imageWidth;
          var top = res["rect"]["y"] * imageHeight;
          var width = res["rect"]["w"] * imageWidth;
          var height = res["rect"]["h"] * imageHeight;
          return (
            <View key={id} style={[styles.box, { top, left, width, height }]}>
              <Text style={{ color: 'white', backgroundColor: blue }}>
                {res["detectedClass"] + " " + (res["confidenceInClass"] * 100).toFixed(0) + "%"}
              </Text>
            </View>
          )
        });
        break;

      case deeplab:
        return (
          recognitions.length > 0 ?
            <Image
              style={{ flex: 1, width: imageWidth, height: imageHeight }}
              source={{ uri: 'data:image/png;base64,' + recognitions }}
              opacity={0.6}
            /> : undefined
        );
        break;

      case posenet:
        return recognitions.map((res, id) => {
          return Object.values(res["keypoints"]).map((k, id) => {
            var left = k["x"] * imageWidth - 6;
            var top = k["y"] * imageHeight - 6;
            var width = imageWidth;
            var height = imageHeight;
            return (
              <View key={id} style={{position: 'absolute', top, left, width, height }}>
                <Text style={{ color: blue, fontSize: 12 }}>
                  {"● " + k["part"]}
                </Text>
              </View>
            )
          });
        });
        break;

      default:
        return recognitions.map((res, id) => {
          return (
            <Text key={id} style={{ color: 'black' }}>
              {res["label"] + "-" + (res["confidence"] * 100).toFixed(0) + "%"}
            </Text>
          )
        });
    }
  }

  render() {
    const { model, source, imageHeight, imageWidth } = this.state;
    var renderButton = (m) => {
      return (
        <TouchableOpacity style={styles.button} onPress={this.onSelectModel.bind(this, m)}>
          <Text style={styles.buttonText}>{m}</Text>
        </TouchableOpacity>
      );
    }
    return (
      <View style={styles.container}>
        {model ?
          <TouchableOpacity style={
            [styles.imageContainer, {
              height: imageHeight,
              width: imageWidth,
              borderWidth: source ? 0 : 2
            }]} onPress={this.onSelectImage.bind(this)}>
            {
              source ?
                <Image source={source} style={{
                  height: imageHeight, width: imageWidth
                }} resizeMode="contain" /> :
                <Text style={styles.text}>Select Picture</Text>
            }
            <View style={styles.boxes}>
              {this.renderResults()}
            </View>
          </TouchableOpacity>
          :
          <View>
            {renderButton(mobile)}
            {renderButton(ssd)}
            {renderButton(yolo)}
            {renderButton(deeplab)}
            {renderButton(posenet)}
          </View>
        }
      </View>
    );
  }
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: 'white'
  },
  imageContainer: {
    borderColor: blue,
    borderRadius: 5,
    alignItems: "center"
  },
  text: {
    color: blue
  },
  button: {
    width: 200,
    backgroundColor: blue,
    borderRadius: 10,
    height: 40,
    alignItems: 'center',
    justifyContent: 'center',
    marginBottom: 10
  },
  buttonText: {
    color: 'white',
    fontSize: 15
  },
  box: {
    position: 'absolute',
    borderColor: blue,
    borderWidth: 2,
  },
  boxes: {
    position: 'absolute',
    left: 0,
    right: 0,
    bottom: 0,
    top: 0,
  }
});
