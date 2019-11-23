#include "imageclassifier.h"

using tensorflow::Flag;
using tensorflow::Tensor;
using tensorflow::Status;
using tensorflow::string;
using tensorflow::int32;
using namespace std;

//调用此静态函数的时候不需要初始化对象，只需要在声明的时候初始化就好
Status ImageClassifier::readEntireFile(tensorflow::Env* env, const string& filename,
                             Tensor* output) {
  tensorflow::uint64 file_size = 0;
  TF_RETURN_IF_ERROR(env->GetFileSize(filename, &file_size));

  string contents;
  contents.resize(file_size);

  std::unique_ptr<tensorflow::RandomAccessFile> file;
  TF_RETURN_IF_ERROR(env->NewRandomAccessFile(filename, &file));

  tensorflow::StringPiece data;
  TF_RETURN_IF_ERROR(file->Read(0, file_size, &data, &(contents)[0]));
  if (data.size() != file_size) {
    return tensorflow::errors::DataLoss("Truncated read of '", filename,
                                        "' expected ", file_size, " got ",
                                        data.size());
  }
  output->scalar<string>()() = string(data);
  return Status::OK();
}


ImageClassifier::ImageClassifier()
{

}

ImageClassifier::~ImageClassifier()
{

}
ImageClassifier::ImageClassifier(const int& imageWidth, const int& imageHeight,
                                 const std::string& in_layer, const std::string& out_layer,
                                 const std::string& label_path,
                                 const float& mean_val, const float& std_val )
{
    m_inputWidth = imageWidth;
    m_inputHeight = imageHeight;
    m_inputLayer = in_layer;
    m_outputLayer = out_layer;
    m_inputMean = mean_val;
    m_inputStd = std_val;
    readLabelsFile(label_path);
}

Status ImageClassifier:: loadGraph(const std::string& graph_file_path)
{
     Status load_graph_status =
           ReadBinaryProto(tensorflow::Env::Default(), graph_file_path, &m_graphDef);
     if (!load_graph_status.ok()) {
         return tensorflow::errors::NotFound("Failed to load compute graph at '",
                                             graph_file_path, "'");
       }

     this->m_session.reset(tensorflow::NewSession(tensorflow::SessionOptions()));
     //m_session 是一个uique_ptr,.reset(q),->m_session指向q，reset()，释放指针

     Status session_create_status = m_session->Create(m_graphDef);
     if (!session_create_status.ok()) {
        return session_create_status;
      }
     return Status::OK();
}
Status ImageClassifier::readLabelsFile(const string& file_name) {
  std::ifstream file(file_name);
  if (!file) {
    return tensorflow::errors::NotFound("Labels file ", file_name,
                                        " not found.");
  }
  m_labelList.clear();
  string line;
  while (std::getline(file, line)) {
    m_labelList.push_back(line);
  }
  m_labelCount = m_labelList.size();
  const int padding = 16;
  while (m_labelList.size() % padding) {
    m_labelList.emplace_back();
  }
  return Status::OK();
}

Status ImageClassifier::readTensorFromImageFile(const string& file_name,
                               std::vector<Tensor>* out_tensors) {
  auto root = tensorflow::Scope::NewRootScope();
  using namespace ::tensorflow::ops;  // NOLINT(build/namespaces)
  string input_name = "file_reader";
  string output_name = "scaled";
  // 将文件读入tensor
  Tensor input(tensorflow::DT_STRING, tensorflow::TensorShape());
  TF_RETURN_IF_ERROR(
      readEntireFile(tensorflow::Env::Default(), file_name, &input));
  auto file_reader =
      Placeholder(root.WithOpName("input"), tensorflow::DataType::DT_STRING);
  std::vector<std::pair<string, tensorflow::Tensor>> inputs = {
      {"input", input},
  };

 //根据不同的图像类型选择解码器
  tensorflow::Output image_reader;
  if (tensorflow::str_util::EndsWith(file_name, ".png")) {
    image_reader = DecodePng(root.WithOpName("png_reader"), file_reader,
                             DecodePng::Channels(this->m_imageChannel));
  } else if (tensorflow::str_util::EndsWith(file_name, ".gif")) {
    // gif decoder returns 4-D tensor, remove the first dim
    image_reader =
        Squeeze(root.WithOpName("squeeze_first_dim"),
                DecodeGif(root.WithOpName("gif_reader"), file_reader));
  } else if (tensorflow::str_util::EndsWith(file_name, ".bmp")) {
    image_reader = DecodeBmp(root.WithOpName("bmp_reader"), file_reader);
  } else {
    // Assume if it's neither a PNG nor a GIF then it must be a JPEG.
    image_reader = DecodeJpeg(root.WithOpName("jpeg_reader"), file_reader,
                              DecodeJpeg::Channels(this->m_imageChannel));
  }

  // 图像预处理，将图像转为浮点型，/255归一化
  // Now cast the image data to float so we can do normal math on it. //
  auto float_caster =
      Cast(root.WithOpName("float_caster"), image_reader, tensorflow::DT_FLOAT);
  auto dims_expander = ExpandDims(root, float_caster, 0);
  // Bilinearly resize the image to fit the required dimensions. µ÷ÕûÍŒÏñÒÔÊÊÓŠ³ßŽç
  auto resized = ResizeBilinear(
      root, dims_expander,
      Const(root.WithOpName("size"), {m_inputHeight, m_inputWidth}));
  // Subtract the mean and divide by the scale 预处理归一化.
  auto sub1 = Sub(root, resized, {m_inputMean});
  auto normalized =Div(root, sub1, {m_inputStd});
  // scale归一化，output_name用于返回tensor
  Mul(root.WithOpName(output_name), Sub(root, normalized, {m_sub}),
      {m_scale});
  // returns the results in the output tensor.  //运行图，输出预处理后的tensortensor
  tensorflow::GraphDef graph;
  TF_RETURN_IF_ERROR(root.ToGraphDef(&graph));
  std::unique_ptr<tensorflow::Session> session(
      tensorflow::NewSession(tensorflow::SessionOptions()));
  TF_RETURN_IF_ERROR(session->Create(graph));
  TF_RETURN_IF_ERROR(session->Run({inputs}, {output_name}, {}, out_tensors));
  return Status::OK();
}

tensorflow::Status ImageClassifier::testSingleImage(const std::string& image_path,
                                                   std::vector<string>& output_class,
                                                   vector<float>& output_scores)
{
    std::vector<Tensor> resized_tensors;
    Status read_tensor_status =
            readTensorFromImageFile(image_path, &resized_tensors);
    if (!read_tensor_status.ok()) {
       LOG(ERROR) << read_tensor_status;
        return tensorflow::errors::NotFound("Image file ", image_path,
                                                   " not found");
     }
    const Tensor& resized_tensor = resized_tensors[0];
    std::vector<Tensor> outputs;
    Status run_status = m_session->Run({{m_inputLayer, resized_tensor}},
                                      {m_outputLayer}, {}, &outputs);
    //Session::Run(const std::vector< std::pair< string, Tensor > > &inputs,
    //const std::vector< string > &output_tensor_names,
    //const std::vector< string > &target_node_names, std::vector< Tensor > *outputs)
    if (!run_status.ok()) {
       LOG(ERROR) << "Running model failed: " << run_status;
     }
    const int how_many_labels = std::min(5, static_cast<int>(m_labelCount));
    Tensor indices;
    Tensor scores;
    //

    TF_RETURN_IF_ERROR(getTopLabels(outputs, how_many_labels, &indices, &scores));
    //
    tensorflow::TTypes<float>::Flat scores_flat = scores.flat<float>();
    tensorflow::TTypes<int32>::Flat indices_flat = indices.flat<int32>();
    output_class.clear();
    output_scores.clear();
    int predict_index = 0;
    float max_score = 0.0;
    for (int pos = 0; pos < how_many_labels; ++pos) {
       const int label_index = indices_flat(pos);
       output_class.push_back(m_labelList[label_index]);
       const float score = scores_flat(pos);
       output_scores.push_back(score);
//       LOG(INFO) << m_labelList[label_index] << " (" << label_index << "): " << score;
       if(score > max_score)
       {
           max_score = score;
           predict_index = label_index;
       }
     }
//     LOG(INFO) << " predict label is " <<  m_labelList[predict_index];


     return Status::OK();

}



Status ImageClassifier::getTopLabels(const std::vector<Tensor>& outputs, int how_many_labels,
                    Tensor* indices, Tensor* scores) {
  auto root = tensorflow::Scope::NewRootScope();
  using namespace ::tensorflow::ops;  // NOLINT(build/namespaces)
  string output_name = "top_k";   //topk的节点名
// TODO
  TopK(root.WithOpName(output_name), outputs[0], how_many_labels);

  tensorflow::GraphDef graph;

  TF_RETURN_IF_ERROR(root.ToGraphDef(&graph));
  std::unique_ptr<tensorflow::Session> session(
      tensorflow::NewSession(tensorflow::SessionOptions()));
  TF_RETURN_IF_ERROR(session->Create(graph));
//  TopK节点返回两个输出，其得分和他的原始的索引
  std::vector<Tensor> out_tensors;
  TF_RETURN_IF_ERROR(session->Run({}, {output_name + ":0", output_name + ":1"},
                                  {}, &out_tensors));
  *scores = out_tensors[0];
  *indices = out_tensors[1];
  return Status::OK();
}

void ImageClassifier::setInputHeight(const int& height)
{
    this->m_inputHeight = height;
}

void ImageClassifier::setInputWidth(const int& width)
{
    this->m_inputWidth = width;
}

void ImageClassifier::setInputDepth(const int& depth)
{
    this->m_imageChannel = depth;
}

void ImageClassifier::setInputLayer(const std::string& layer_name)
{
    this->m_inputLayer = layer_name;
}

void ImageClassifier::setOutputLayer(const std::string& layer_name)
{
    this->m_outputLayer = layer_name;
}
void ImageClassifier::setInputMeanVal(const float& meanVal)
{
    this->m_inputMean = meanVal;
}

void ImageClassifier::setInputStdVal(const float& stdVal)
{
    this->m_inputStd = stdVal;
}

void ImageClassifier::setInputScaleVal(const float& scaleVal)
{
    this->m_scale = scaleVal;
}

void ImageClassifier::setInputSubVal(const float& subVal)
{
    this->m_sub = subVal;
}



