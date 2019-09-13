require 'json'

package = JSON.parse(File.read(File.join(__dir__, 'package.json')))

Pod::Spec.new do |s|
  s.name         = package['name']
  s.version      = package['version']
  s.summary      = package['description']
  s.license      = package['license']
  s.authors      = package['author']
  s.homepage     = package['homepage']
  s.platform     = :ios, "9.0"

  # s.source        = { :git => "https://github.com/shaqian/tflite-react-native.git", :tag => "master" }
  s.source        = { :git => '' }
  s.source_files  = "ios/**/*.{h,m,mm}"  
  s.dependency 'React'
  s.dependency 'TensorFlowLite', '~> 1.13.1'

  s.requires_arc = true
  s.pod_target_xcconfig = {
    'HEADER_SEARCH_PATHS' => "'${SRCROOT}/TensorFlowLite/Frameworks/tensorflow_lite.framework/Headers'",
    'OTHER_LDFLAGS'       => "-force_load '${SRCROOT}/TensorFlowLite/Frameworks/tensorflow_lite.framework/tensorflow_lite' '-L ${SRCROOT}/TensorFlowLite/Frameworks/tensorflow_lite.framework'"
  }
end