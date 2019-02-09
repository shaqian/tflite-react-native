
Pod::Spec.new do |s|
  s.name         = "TfliteReactNative"
  s.version      = "0.0.3"
  s.summary      = "TfliteReactNative"
  s.description  = <<-DESC
                  TfliteReactNative
                   DESC
  s.homepage     = ""
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

  