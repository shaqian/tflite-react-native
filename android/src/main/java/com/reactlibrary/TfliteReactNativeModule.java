
package com.reactlibrary;

import android.content.res.AssetFileDescriptor;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Matrix;
import android.graphics.Canvas;

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

    loadLabels(assetManager, labelsPath);

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
      throw new RuntimeException("Failed to read label file" , e);
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

    InputStream inputStream = new FileInputStream(path.replace("file://",""));
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
          imgData.put((byte)((pixelValue >> 16) & 0xFF));
          imgData.put((byte)((pixelValue >> 8) & 0xFF));
          imgData.put((byte)(pixelValue & 0xFF));
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

          final float confidence = expit(output[0][y][x][offset + 4]);

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
            final float xPos = (x + expit(output[0][y][x][offset + 0])) * blockSize;
            final float yPos = (y + expit(output[0][y][x][offset + 1])) * blockSize;

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

  private float expit(final float x) {
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
                                                final boolean maintainAspectRatio)
  {
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

  @ReactMethod
  private void close() {
    tfLite.close();
    labels = null;
    labelProb = null;
  }
}