
Pod::Spec.new do |s|
  s.name         = "TfliteReactNative"
  s.version      = "0.0.4"
  s.summary      = "TfliteReactNative"
  s.description  = <<-DESC
                  A React Native library for accessing TensorFlow Lite API. Supports Classification and Object Detection on both iOS and Android.
                   DESC
  s.homepage     = "https://github.com/shaqian/tflite-react-native"
  s.license      = "MIT"
  # s.license     = { :type => "MIT", :file => "../LICENSE" }
  s.author       = { 'Qian Sha' => 'https://github.com/shaqian' }
  s.platform     = :ios, "7.0"
  s.source       = { :git => "https://github.com/shaqian/tflite-react-native.git", :tag => "master" }
  s.source_files  = "TfliteReactNative/**/*.{h,m}"
  s.requires_arc = true


  s.dependency "React"
  s.dependency 'TensorFlowLite'

end

  