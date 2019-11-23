#ifndef IMAGECLASSIFIER_H
#define IMAGECLASSIFIER_H
#include <fstream>
#include <utility>
#include <vector>

#include "tensorflow/cc/ops/const_op.h"
#include "tensorflow/cc/ops/image_ops.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/graph/default_device.h"
#include "tensorflow/core/graph/graph_def_builder.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/util/command_line_flags.h"


class ImageClassifier
{
public:
    ImageClassifier();
    ImageClassifier(const int& imageWidth, const int& imageHeight,
                    const std::string& in_layer, const std::string& out_layer,
                    const std::string& label_path,
                    const float& mean_val, const float& std_val);
    ~ImageClassifier();
    tensorflow::Status loadGraph(const std::string& graph_file_path);
    tensorflow::Status readLabelsFile(const std::string& file_name);
    static tensorflow::Status readEntireFile(tensorflow::Env* env, const std::string& filename,
                                 tensorflow::Tensor* output);
    void setInputHeight(const int& height);
    void setInputWidth(const int& width);
    void setInputDepth(const int& depth);
    void setInputLayer(const std::string& layer_name);
    void setOutputLayer(const std::string& layer_name);
    void setInputMeanVal(const float& meanVal);
    void setInputStdVal(const float& stdVal);
    void setInputScaleVal(const float& scaleVal);
    void setInputSubVal(const float& subVal);

    tensorflow::Status testSingleImage(const std::string& imagePath,
                                       std::vector<std::string>& output_class,
                                       std::vector<float>& output_scores);
    tensorflow::Status testBatchImages(const int& batch_size, const std::string& imagePath,
                                       std::vector<std::vector<std::string>>& vec_output_classes,
                                       std::vector<std::vector<float>> &vec_output_scores);


private:
    tensorflow::Status getTopLabels(const std::vector<tensorflow::Tensor>& outputs, int how_many_labels,
                        tensorflow::Tensor* indices, tensorflow::Tensor* scores);
    tensorflow::Status readTensorFromImageFile(const std::string& file_name,
                                   std::vector<tensorflow::Tensor>* out_tensors);
    int m_inputHeight = 0;
    int m_inputWidth = 0;
    int m_imageChannel = 3;
    std::string m_inputLayer = ""; //网络的输入节点名//定义在固化pb的网络中了
    std::string m_outputLayer = "";
    std::vector<std::string> m_labelList; //图像的标签列表
    std:: size_t m_labelCount;
    float m_inputMean = 0;
    float m_inputStd = 255;
    float m_scale = 2.0;
    float m_sub = 0.5;
    std::unique_ptr<tensorflow::Session> m_session; //网络的会话
    tensorflow::GraphDef m_graphDef;
};

#endif // IMAGECLASSIFIER_H
