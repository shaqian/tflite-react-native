
package com.reactlibrary;

import android.content.res.AssetFileDescriptor;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Color;
import android.graphics.Matrix;
import android.graphics.Canvas;
import android.util.Base64;

import com.facebook.react.bridge.Arguments;
import com.facebook.react.bridge.ReactApplicationContext;
import com.facebook.react.bridge.ReactContextBaseJavaModule;
import com.facebook.react.bridge.ReactMethod;
import com.facebook.react.bridge.Callback;
import com.facebook.react.bridge.ReadableArray;
import com.facebook.react.bridge.WritableArray;
import com.facebook.react.bridge.WritableMap;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.Tensor;

import java.io.BufferedReader;
import java.io.ByteArrayOutputStream;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.PriorityQueue;
import java.util.Vector;

public class TfliteReactNativeModule extends ReactContextBaseJavaModule {

  private final ReactApplicationContext reactContext;
  private Interpreter tfLite;
  private int inputSize = 0;
  private Vector<String> labels;
  float[][] labelProb;
  private static final int BYTES_PER_CHANNEL = 4;


  public TfliteReactNativeModule(ReactApplicationContext reactContext) {
    super(reactContext);
    this.reactContext = reactContext;
  }

  @Override
  public String getName() {
    return "TfliteReactNative";
  }

  @ReactMethod
  private void loadModel(final String modelPath, final String labelsPath, final int numThreads, final Callback callback)
      throws IOException {
    AssetManager assetManager = reactContext.getAssets();
    AssetFileDescriptor fileDescriptor = assetManager.openFd(modelPath);
    FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
    FileChannel fileChannel = inputStream.getChannel();
    long startOffset = fileDescriptor.getStartOffset();
    long declaredLength = fileDescriptor.getDeclaredLength();
    MappedByteBuffer buffer = fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);

    final Interpreter.Options tfliteOptions = new Interpreter.Options();
    tfliteOptions.setNumThreads(numThreads);
    tfLite = new Interpreter(buffer, tfliteOptions);

    if (labelsPath.length() > 0) {
      loadLabels(assetManager, labelsPath);
    }

    callback.invoke(null, "success");
  }

  private void loadLabels(AssetManager assetManager, String path) {
    BufferedReader br;
    try {
      br = new BufferedReader(new InputStreamReader(assetManager.open(path)));
      String line;
      labels = new Vector<>();
      while ((line = br.readLine()) != null) {
        labels.add(line);
      }
      labelProb = new float[1][labels.size()];
      br.close();
    } catch (IOException e) {
      throw new RuntimeException("Failed to read label file", e);
    }
  }


  private WritableArray GetTopN(int numResults, float threshold) {
    PriorityQueue<WritableMap> pq =
        new PriorityQueue<>(
            1,
            new Comparator<WritableMap>() {
              @Override
              public int compare(WritableMap lhs, WritableMap rhs) {
                return Double.compare(rhs.getDouble("confidence"), lhs.getDouble("confidence"));
              }
            });

    for (int i = 0; i < labels.size(); ++i) {
      float confidence = labelProb[0][i];
      if (confidence > threshold) {
        WritableMap res = Arguments.createMap();
        res.putInt("index", i);
        res.putString("label", labels.size() > i ? labels.get(i) : "unknown");
        res.putDouble("confidence", confidence);
        pq.add(res);
      }
    }

    WritableArray results = Arguments.createArray();
    int recognitionsSize = Math.min(pq.size(), numResults);
    for (int i = 0; i < recognitionsSize; ++i) {
      results.pushMap(pq.poll());
    }
    return results;
  }

  ByteBuffer feedInputTensorImage(String path, float mean, float std) throws IOException {
    Tensor tensor = tfLite.getInputTensor(0);
    inputSize = tensor.shape()[1];
    int inputChannels = tensor.shape()[3];

    InputStream inputStream = new FileInputStream(path.replace("file://", ""));
    Bitmap bitmapRaw = BitmapFactory.decodeStream(inputStream);

    Matrix matrix = getTransformationMatrix(bitmapRaw.getWidth(), bitmapRaw.getHeight(),
        inputSize, inputSize, false);

    int[] intValues = new int[inputSize * inputSize];
    int bytePerChannel = tensor.dataType() == DataType.UINT8 ? 1 : BYTES_PER_CHANNEL;
    ByteBuffer imgData = ByteBuffer.allocateDirect(1 * inputSize * inputSize * inputChannels * bytePerChannel);
    imgData.order(ByteOrder.nativeOrder());

    Bitmap bitmap = Bitmap.createBitmap(inputSize, inputSize, Bitmap.Config.ARGB_8888);
    final Canvas canvas = new Canvas(bitmap);
    canvas.drawBitmap(bitmapRaw, matrix, null);
    bitmap.getPixels(intValues, 0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight());

    int pixel = 0;
    for (int i = 0; i < inputSize; ++i) {
      for (int j = 0; j < inputSize; ++j) {
        int pixelValue = intValues[pixel++];
        if (tensor.dataType() == DataType.FLOAT32) {
          imgData.putFloat((((pixelValue >> 16) & 0xFF) - mean) / std);
          imgData.putFloat((((pixelValue >> 8) & 0xFF) - mean) / std);
          imgData.putFloat(((pixelValue & 0xFF) - mean) / std);
        } else {
          imgData.put((byte) ((pixelValue >> 16) & 0xFF));
          imgData.put((byte) ((pixelValue >> 8) & 0xFF));
          imgData.put((byte) (pixelValue & 0xFF));
        }
      }
    }

    return imgData;
  }

  @ReactMethod
  private void runModelOnImage(final String path, final float mean, final float std, final int numResults,
                               final float threshold, final Callback callback) throws IOException {

    tfLite.run(feedInputTensorImage(path, mean, std), labelProb);

    callback.invoke(null, GetTopN(numResults, threshold));
  }

  @ReactMethod
  private void detectObjectOnImage(final String path, final String model, final float mean, final float std,
                                   final float threshold, final int numResultsPerClass, final ReadableArray ANCHORS,
                                   final int blockSize, final Callback callback) throws IOException {

    ByteBuffer imgData = feedInputTensorImage(path, mean, std);

    if (model.equals("SSDMobileNet")) {
      int NUM_DETECTIONS = 10;
      float[][][] outputLocations = new float[1][NUM_DETECTIONS][4];
      float[][] outputClasses = new float[1][NUM_DETECTIONS];
      float[][] outputScores = new float[1][NUM_DETECTIONS];
      float[] numDetections = new float[1];

      Object[] inputArray = {imgData};
      Map<Integer, Object> outputMap = new HashMap<>();
      outputMap.put(0, outputLocations);
      outputMap.put(1, outputClasses);
      outputMap.put(2, outputScores);
      outputMap.put(3, numDetections);

      tfLite.runForMultipleInputsOutputs(inputArray, outputMap);

      callback.invoke(null,
          parseSSDMobileNet(NUM_DETECTIONS, numResultsPerClass, outputLocations, outputClasses, outputScores));
    } else {
      int gridSize = inputSize / blockSize;
      int numClasses = labels.size();
      final float[][][][] output = new float[1][gridSize][gridSize][(numClasses + 5) * 5];
      tfLite.run(imgData, output);

      callback.invoke(null,
          parseYOLO(output, inputSize, blockSize, 5, numClasses, ANCHORS, threshold, numResultsPerClass));
    }
  }

  private WritableArray parseSSDMobileNet(int numDetections, int numResultsPerClass, float[][][] outputLocations,
                                          float[][] outputClasses, float[][] outputScores) {
    Map<String, Integer> counters = new HashMap<>();
    WritableArray results = Arguments.createArray();

    for (int i = 0; i < numDetections; ++i) {
      String detectedClass = labels.get((int) outputClasses[0][i] + 1);

      if (counters.get(detectedClass) == null) {
        counters.put(detectedClass, 1);
      } else {
        int count = counters.get(detectedClass);
        if (count >= numResultsPerClass) {
          continue;
        } else {
          counters.put(detectedClass, count + 1);
        }
      }

      WritableMap rect = Arguments.createMap();
      float ymin = Math.max(0, outputLocations[0][i][0]);
      float xmin = Math.max(0, outputLocations[0][i][1]);
      float ymax = outputLocations[0][i][2];
      float xmax = outputLocations[0][i][3];
      rect.putDouble("x", xmin);
      rect.putDouble("y", ymin);
      rect.putDouble("w", Math.min(1 - xmin, xmax - xmin));
      rect.putDouble("h", Math.min(1 - ymin, ymax - ymin));

      WritableMap result = Arguments.createMap();
      result.putMap("rect", rect);
      result.putDouble("confidenceInClass", outputScores[0][i]);
      result.putString("detectedClass", detectedClass);

      results.pushMap(result);
    }

    return results;
  }

  private WritableArray parseYOLO(float[][][][] output, int inputSize, int blockSize, int numBoxesPerBlock, int numClasses,
                                  ReadableArray anchors, float threshold, int numResultsPerClass) {
    PriorityQueue<WritableMap> pq =
        new PriorityQueue<>(
            1,
            new Comparator<WritableMap>() {
              @Override
              public int compare(WritableMap lhs, WritableMap rhs) {
                return Double.compare(rhs.getDouble("confidenceInClass"), lhs.getDouble("confidenceInClass"));
              }
            });

    int gridSize = inputSize / blockSize;

    for (int y = 0; y < gridSize; ++y) {
      for (int x = 0; x < gridSize; ++x) {
        for (int b = 0; b < numBoxesPerBlock; ++b) {
          final int offset = (numClasses + 5) * b;

          final float confidence = sigmoid(output[0][y][x][offset + 4]);

          final float[] classes = new float[numClasses];
          for (int c = 0; c < numClasses; ++c) {
            classes[c] = output[0][y][x][offset + 5 + c];
          }
          softmax(classes);

          int detectedClass = -1;
          float maxClass = 0;
          for (int c = 0; c < numClasses; ++c) {
            if (classes[c] > maxClass) {
              detectedClass = c;
              maxClass = classes[c];
            }
          }

          final float confidenceInClass = maxClass * confidence;
          if (confidenceInClass > threshold) {
            final float xPos = (x + sigmoid(output[0][y][x][offset + 0])) * blockSize;
            final float yPos = (y + sigmoid(output[0][y][x][offset + 1])) * blockSize;

            final float w = (float) (Math.exp(output[0][y][x][offset + 2]) * anchors.getDouble(2 * b + 0)) * blockSize;
            final float h = (float) (Math.exp(output[0][y][x][offset + 3]) * anchors.getDouble(2 * b + 1)) * blockSize;

            final float xmin = Math.max(0, (xPos - w / 2) / inputSize);
            final float ymin = Math.max(0, (yPos - h / 2) / inputSize);

            WritableMap rect = Arguments.createMap();
            rect.putDouble("x", xmin);
            rect.putDouble("y", ymin);
            rect.putDouble("w", Math.min(1 - xmin, w / inputSize));
            rect.putDouble("h", Math.min(1 - ymin, h / inputSize));

            WritableMap result = Arguments.createMap();
            result.putMap("rect", rect);
            result.putDouble("confidenceInClass", confidenceInClass);
            result.putString("detectedClass", labels.get(detectedClass));

            pq.add(result);
          }
        }
      }
    }

    Map<String, Integer> counters = new HashMap<>();
    WritableArray results = Arguments.createArray();

    for (int i = 0; i < pq.size(); ++i) {
      WritableMap result = pq.poll();
      String detectedClass = result.getString("detectedClass").toString();

      if (counters.get(detectedClass) == null) {
        counters.put(detectedClass, 1);
      } else {
        int count = counters.get(detectedClass);
        if (count >= numResultsPerClass) {
          continue;
        } else {
          counters.put(detectedClass, count + 1);
        }
      }
      results.pushMap(result);
    }

    return results;
  }

  byte[] fetchArgmax(ByteBuffer output, ReadableArray labelColors, String outputType) {
    Tensor outputTensor = tfLite.getOutputTensor(0);
    int outputBatchSize = outputTensor.shape()[0];
    assert outputBatchSize == 1;
    int outputHeight = outputTensor.shape()[1];
    int outputWidth = outputTensor.shape()[2];
    int outputChannels = outputTensor.shape()[3];

    Bitmap outputArgmax = null;
    byte[] outputBytes = new byte[outputWidth * outputHeight * 4];
    if (outputType.equals("png")) {
      outputArgmax = Bitmap.createBitmap(outputWidth, outputHeight, Bitmap.Config.ARGB_8888);
    }

    if (outputTensor.dataType() == DataType.FLOAT32) {
      for (int i = 0; i < outputHeight; ++i) {
        for (int j = 0; j < outputWidth; ++j) {
          int maxIndex = 0;
          float maxValue = 0.0f;
          for (int c = 0; c < outputChannels; ++c) {
            float outputValue = output.getFloat();
            if (outputValue > maxValue) {
              maxIndex = c;
              maxValue = outputValue;
            }
          }
          int labelColor = labelColors.getInt(maxIndex);
          if (outputType.equals("png")) {
            outputArgmax.setPixel(j, i, Color.rgb((labelColor >> 16) & 0xFF, (labelColor >> 8) & 0xFF, labelColor & 0xFF));
          } else {
            setPixel(outputBytes, i * outputWidth + j, labelColor);
          }
        }
      }
    } else {
      for (int i = 0; i < outputHeight; ++i) {
        for (int j = 0; j < outputWidth; ++j) {
          int maxIndex = 0;
          int maxValue = 0;
          for (int c = 0; c < outputChannels; ++c) {
            int outputValue = output.get();
            if (outputValue > maxValue) {
              maxIndex = c;
              maxValue = outputValue;
            }
          }
          int labelColor = labelColors.getInt(maxIndex);
          if (outputType.equals("png")) {
            outputArgmax.setPixel(j, i, Color.rgb((labelColor >> 16) & 0xFF, (labelColor >> 8) & 0xFF, labelColor & 0xFF));
          } else {
            setPixel(outputBytes, i * outputWidth + j, labelColor);
          }
        }
      }
    }
    if (outputType.equals("png")) {
      return compressPNG(outputArgmax);
    } else {
      return outputBytes;
    }
  }

  void setPixel(byte[] rgba, int index, long color) {
    rgba[index * 4] = (byte) ((color >> 16) & 0xFF);
    rgba[index * 4 + 1] = (byte) ((color >> 8) & 0xFF);
    rgba[index * 4 + 2] = (byte) (color & 0xFF);
    rgba[index * 4 + 3] = (byte) ((color >> 24) & 0xFF);
  }

  byte[] compressPNG(Bitmap bitmap) {
    // https://stackoverflow.com/questions/4989182/converting-java-bitmap-to-byte-array#4989543
    ByteArrayOutputStream stream = new ByteArrayOutputStream();
    bitmap.compress(Bitmap.CompressFormat.PNG, 100, stream);
    byte[] byteArray = stream.toByteArray();
    // bitmap.recycle();
    return byteArray;
  }

  @ReactMethod
  private void runSegmentationOnImage(final String path, final float mean, final float std, final ReadableArray labelColors,
                                      final String outputType, final Callback callback) throws IOException {
    int i = tfLite.getOutputTensor(0).numBytes();
    ByteBuffer output = ByteBuffer.allocateDirect(tfLite.getOutputTensor(0).numBytes());
    output.order(ByteOrder.nativeOrder());
    tfLite.run(feedInputTensorImage(path, mean, std), output);

    if (output.position() != output.limit()) {
      callback.invoke("Unexpected output position", null);
      return;
    }
    output.flip();

    byte[] res = fetchArgmax(output, labelColors, outputType);
    String base64String = Base64.encodeToString(res, Base64.NO_WRAP);

    callback.invoke(null, base64String);
  }

  String[] partNames = {
      "nose", "leftEye", "rightEye", "leftEar", "rightEar", "leftShoulder",
      "rightShoulder", "leftElbow", "rightElbow", "leftWrist", "rightWrist",
      "leftHip", "rightHip", "leftKnee", "rightKnee", "leftAnkle", "rightAnkle"
  };

  String[][] poseChain = {
      {"nose", "leftEye"}, {"leftEye", "leftEar"}, {"nose", "rightEye"},
      {"rightEye", "rightEar"}, {"nose", "leftShoulder"},
      {"leftShoulder", "leftElbow"}, {"leftElbow", "leftWrist"},
      {"leftShoulder", "leftHip"}, {"leftHip", "leftKnee"},
      {"leftKnee", "leftAnkle"}, {"nose", "rightShoulder"},
      {"rightShoulder", "rightElbow"}, {"rightElbow", "rightWrist"},
      {"rightShoulder", "rightHip"}, {"rightHip", "rightKnee"},
      {"rightKnee", "rightAnkle"}
  };

  Map<String, Integer> partsIds = new HashMap<>();
  List<Integer> parentToChildEdges = new ArrayList<>();
  List<Integer> childToParentEdges = new ArrayList<>();


  void initPoseNet(Map<Integer, Object> outputMap) {
    if (partsIds.size() == 0) {
      for (int i = 0; i < partNames.length; ++i)
        partsIds.put(partNames[i], i);

      for (int i = 0; i < poseChain.length; ++i) {
        parentToChildEdges.add(partsIds.get(poseChain[i][1]));
        childToParentEdges.add(partsIds.get(poseChain[i][0]));
      }
    }

    for (int i = 0; i < tfLite.getOutputTensorCount(); i++) {
      int[] shape = tfLite.getOutputTensor(i).shape();
      float[][][][] output = new float[shape[0]][shape[1]][shape[2]][shape[3]];
      outputMap.put(i, output);
    }
  }

  @ReactMethod
  private void runPoseNetOnImage(final String path, final float mean, final float std, final int numResults,
                                 final float threshold, final int nmsRadius, final Callback callback) throws IOException {
    int localMaximumRadius = 1;
    int outputStride = 16;

    ByteBuffer imgData = feedInputTensorImage(path, mean, std);
    Object[] input = new Object[]{imgData};

    Map<Integer, Object> outputMap = new HashMap<>();
    initPoseNet(outputMap);

    tfLite.runForMultipleInputsOutputs(input, outputMap);

    float[][][] scores = ((float[][][][]) outputMap.get(0))[0];
    float[][][] offsets = ((float[][][][]) outputMap.get(1))[0];
    float[][][] displacementsFwd = ((float[][][][]) outputMap.get(2))[0];
    float[][][] displacementsBwd = ((float[][][][]) outputMap.get(3))[0];

    PriorityQueue<Map<String, Object>> pq = buildPartWithScoreQueue(scores, threshold, localMaximumRadius);

    int numParts = scores[0][0].length;
    int numEdges = parentToChildEdges.size();
    int sqaredNmsRadius = nmsRadius * nmsRadius;

    List<Map<String, Object>> results = new ArrayList<>();

    while (results.size() < numResults && pq.size() > 0) {
      Map<String, Object> root = pq.poll();
      float[] rootPoint = getImageCoords(root, outputStride, numParts, offsets);

      if (withinNmsRadiusOfCorrespondingPoint(
          results, sqaredNmsRadius, rootPoint[0], rootPoint[1], (int) root.get("partId")))
        continue;

      Map<String, Object> keypoint = new HashMap<>();
      keypoint.put("score", root.get("score"));
      keypoint.put("part", partNames[(int) root.get("partId")]);
      keypoint.put("y", rootPoint[0] / inputSize);
      keypoint.put("x", rootPoint[1] / inputSize);

      Map<Integer, Map<String, Object>> keypoints = new HashMap<>();
      keypoints.put((int) root.get("partId"), keypoint);

      for (int edge = numEdges - 1; edge >= 0; --edge) {
        int sourceKeypointId = parentToChildEdges.get(edge);
        int targetKeypointId = childToParentEdges.get(edge);
        if (keypoints.containsKey(sourceKeypointId) && !keypoints.containsKey(targetKeypointId)) {
          keypoint = traverseToTargetKeypoint(edge, keypoints.get(sourceKeypointId),
              targetKeypointId, scores, offsets, outputStride, displacementsBwd);
          keypoints.put(targetKeypointId, keypoint);
        }
      }

      for (int edge = 0; edge < numEdges; ++edge) {
        int sourceKeypointId = childToParentEdges.get(edge);
        int targetKeypointId = parentToChildEdges.get(edge);
        if (keypoints.containsKey(sourceKeypointId) && !keypoints.containsKey(targetKeypointId)) {
          keypoint = traverseToTargetKeypoint(edge, keypoints.get(sourceKeypointId),
              targetKeypointId, scores, offsets, outputStride, displacementsFwd);
          keypoints.put(targetKeypointId, keypoint);
        }
      }

      Map<String, Object> result = new HashMap<>();
      result.put("keypoints", keypoints);
      result.put("score", getInstanceScore(keypoints, numParts));
      results.add(result);
    }

    WritableArray outputs = Arguments.createArray();
    for (Map<String, Object> result : results) {
      Map<Integer, Map<String, Object>> keypoints = (Map<Integer, Map<String, Object>>) result.get("keypoints");

      WritableMap _keypoints = Arguments.createMap();
      for (Map.Entry<Integer, Map<String, Object>> keypoint : keypoints.entrySet()) {
        Map<String, Object> keypoint_ = keypoint.getValue();
        WritableMap _keypoint = Arguments.createMap();
        _keypoint.putDouble("score", Double.valueOf(keypoint_.get("score").toString()));
        _keypoint.putString("part", keypoint_.get("part").toString());
        _keypoint.putDouble("y", Double.valueOf(keypoint_.get("y").toString()));
        _keypoint.putDouble("x", Double.valueOf(keypoint_.get("x").toString()));
        _keypoints.putMap(keypoint.getKey().toString(), _keypoint);
      }

      WritableMap output = Arguments.createMap();
      output.putMap("keypoints", _keypoints);
      output.putDouble("score", Double.valueOf(result.get("score").toString()));

      outputs.pushMap(output);
    }

    callback.invoke(null, outputs);
  }


  PriorityQueue<Map<String, Object>> buildPartWithScoreQueue(float[][][] scores,
                                                             double threshold,
                                                             int localMaximumRadius) {
    PriorityQueue<Map<String, Object>> pq =
        new PriorityQueue<>(
            1,
            new Comparator<Map<String, Object>>() {
              @Override
              public int compare(Map<String, Object> lhs, Map<String, Object> rhs) {
                return Float.compare((float) rhs.get("score"), (float) lhs.get("score"));
              }
            });

    for (int heatmapY = 0; heatmapY < scores.length; ++heatmapY) {
      for (int heatmapX = 0; heatmapX < scores[0].length; ++heatmapX) {
        for (int keypointId = 0; keypointId < scores[0][0].length; ++keypointId) {
          float score = sigmoid(scores[heatmapY][heatmapX][keypointId]);
          if (score < threshold) continue;

          if (scoreIsMaximumInLocalWindow(
              keypointId, score, heatmapY, heatmapX, localMaximumRadius, scores)) {
            Map<String, Object> res = new HashMap<>();
            res.put("score", score);
            res.put("y", heatmapY);
            res.put("x", heatmapX);
            res.put("partId", keypointId);
            pq.add(res);
          }
        }
      }
    }

    return pq;
  }

  boolean scoreIsMaximumInLocalWindow(int keypointId,
                                      float score,
                                      int heatmapY,
                                      int heatmapX,
                                      int localMaximumRadius,
                                      float[][][] scores) {
    boolean localMaximum = true;
    int height = scores.length;
    int width = scores[0].length;

    int yStart = Math.max(heatmapY - localMaximumRadius, 0);
    int yEnd = Math.min(heatmapY + localMaximumRadius + 1, height);
    for (int yCurrent = yStart; yCurrent < yEnd; ++yCurrent) {
      int xStart = Math.max(heatmapX - localMaximumRadius, 0);
      int xEnd = Math.min(heatmapX + localMaximumRadius + 1, width);
      for (int xCurrent = xStart; xCurrent < xEnd; ++xCurrent) {
        if (sigmoid(scores[yCurrent][xCurrent][keypointId]) > score) {
          localMaximum = false;
          break;
        }
      }
      if (!localMaximum) {
        break;
      }
    }

    return localMaximum;
  }

  float[] getImageCoords(Map<String, Object> keypoint,
                         int outputStride,
                         int numParts,
                         float[][][] offsets) {
    int heatmapY = (int) keypoint.get("y");
    int heatmapX = (int) keypoint.get("x");
    int keypointId = (int) keypoint.get("partId");
    float offsetY = offsets[heatmapY][heatmapX][keypointId];
    float offsetX = offsets[heatmapY][heatmapX][keypointId + numParts];

    float y = heatmapY * outputStride + offsetY;
    float x = heatmapX * outputStride + offsetX;

    return new float[]{y, x};
  }

  boolean withinNmsRadiusOfCorrespondingPoint(List<Map<String, Object>> poses,
                                              float squaredNmsRadius,
                                              float y,
                                              float x,
                                              int keypointId) {
    for (Map<String, Object> pose : poses) {
      Map<Integer, Object> keypoints = (Map<Integer, Object>) pose.get("keypoints");
      Map<String, Object> correspondingKeypoint = (Map<String, Object>) keypoints.get(keypointId);
      float _x = (float) correspondingKeypoint.get("x") * inputSize - x;
      float _y = (float) correspondingKeypoint.get("y") * inputSize - y;
      float squaredDistance = _x * _x + _y * _y;
      if (squaredDistance <= squaredNmsRadius)
        return true;
    }

    return false;
  }

  Map<String, Object> traverseToTargetKeypoint(int edgeId,
                                               Map<String, Object> sourceKeypoint,
                                               int targetKeypointId,
                                               float[][][] scores,
                                               float[][][] offsets,
                                               int outputStride,
                                               float[][][] displacements) {
    int height = scores.length;
    int width = scores[0].length;
    int numKeypoints = scores[0][0].length;
    float sourceKeypointY = (float) sourceKeypoint.get("y") * inputSize;
    float sourceKeypointX = (float) sourceKeypoint.get("x") * inputSize;

    int[] sourceKeypointIndices = getStridedIndexNearPoint(sourceKeypointY, sourceKeypointX,
        outputStride, height, width);

    float[] displacement = getDisplacement(edgeId, sourceKeypointIndices, displacements);

    float[] displacedPoint = new float[]{
        sourceKeypointY + displacement[0],
        sourceKeypointX + displacement[1]
    };

    float[] targetKeypoint = displacedPoint;

    final int offsetRefineStep = 2;
    for (int i = 0; i < offsetRefineStep; i++) {
      int[] targetKeypointIndices = getStridedIndexNearPoint(targetKeypoint[0], targetKeypoint[1],
          outputStride, height, width);

      int targetKeypointY = targetKeypointIndices[0];
      int targetKeypointX = targetKeypointIndices[1];

      float offsetY = offsets[targetKeypointY][targetKeypointX][targetKeypointId];
      float offsetX = offsets[targetKeypointY][targetKeypointX][targetKeypointId + numKeypoints];

      targetKeypoint = new float[]{
          targetKeypointY * outputStride + offsetY,
          targetKeypointX * outputStride + offsetX
      };
    }

    int[] targetKeypointIndices = getStridedIndexNearPoint(targetKeypoint[0], targetKeypoint[1],
        outputStride, height, width);

    float score = sigmoid(scores[targetKeypointIndices[0]][targetKeypointIndices[1]][targetKeypointId]);

    Map<String, Object> keypoint = new HashMap<>();
    keypoint.put("score", score);
    keypoint.put("part", partNames[targetKeypointId]);
    keypoint.put("y", targetKeypoint[0] / inputSize);
    keypoint.put("x", targetKeypoint[1] / inputSize);

    return keypoint;
  }

  int[] getStridedIndexNearPoint(float _y, float _x, int outputStride, int height, int width) {
    int y_ = Math.round(_y / outputStride);
    int x_ = Math.round(_x / outputStride);
    int y = y_ < 0 ? 0 : y_ > height - 1 ? height - 1 : y_;
    int x = x_ < 0 ? 0 : x_ > width - 1 ? width - 1 : x_;
    return new int[]{y, x};
  }

  float[] getDisplacement(int edgeId, int[] keypoint, float[][][] displacements) {
    int numEdges = displacements[0][0].length / 2;
    int y = keypoint[0];
    int x = keypoint[1];
    return new float[]{displacements[y][x][edgeId], displacements[y][x][edgeId + numEdges]};
  }

  float getInstanceScore(Map<Integer, Map<String, Object>> keypoints, int numKeypoints) {
    float scores = 0;
    for (Map.Entry<Integer, Map<String, Object>> keypoint : keypoints.entrySet())
      scores += (float) keypoint.getValue().get("score");
    return scores / numKeypoints;
  }

  @ReactMethod
  private void close() {
    tfLite.close();
    labels = null;
    labelProb = null;
  }


  private float sigmoid(final float x) {
    return (float) (1. / (1. + Math.exp(-x)));
  }

  private void softmax(final float[] vals) {
    float max = Float.NEGATIVE_INFINITY;
    for (final float val : vals) {
      max = Math.max(max, val);
    }
    float sum = 0.0f;
    for (int i = 0; i < vals.length; ++i) {
      vals[i] = (float) Math.exp(vals[i] - max);
      sum += vals[i];
    }
    for (int i = 0; i < vals.length; ++i) {
      vals[i] = vals[i] / sum;
    }
  }

  private static Matrix getTransformationMatrix(final int srcWidth,
                                                final int srcHeight,
                                                final int dstWidth,
                                                final int dstHeight,
                                                final boolean maintainAspectRatio) {
    final Matrix matrix = new Matrix();

    if (srcWidth != dstWidth || srcHeight != dstHeight) {
      final float scaleFactorX = dstWidth / (float) srcWidth;
      final float scaleFactorY = dstHeight / (float) srcHeight;

      if (maintainAspectRatio) {
        final float scaleFactor = Math.max(scaleFactorX, scaleFactorY);
        matrix.postScale(scaleFactor, scaleFactor);
      } else {
        matrix.postScale(scaleFactorX, scaleFactorY);
      }
    }

    matrix.invert(new Matrix());
    return matrix;
  }

}