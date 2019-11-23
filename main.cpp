#include <QCoreApplication>
#include "imageclassifier.h"
#include <QTime>
#include <QDebug>
#include <QDir>
#include <algorithm>
#include "tensorflow/core/util/command_line_flags.h"
using tensorflow::Flag;
int main(int argc, char *argv[])
{
    QCoreApplication a(argc, argv);
    //设置解析参数
    std::string image_path = "/home/lilanluo/jetson_workspace/data/iwatch_aotudian/val/test2.png";
    std::string graph_path = "/home/lilanluo/jetson_workspace/data/iwatch_aotudian/model/dense_net.pb";
    std::string label_path = "/home/lilanluo/jetson_workspace/data/iwatch_aotudian/labels.txt";
    std::vector<Flag> flag_list = {				//参数解析，用于控制台程序
          Flag("image", &image_path, "image to be processed"),		//FLag 的参数分别为参数名，参数输入到，参数的注释
          Flag("graph", &graph_path, "graph to be executed"),
          Flag("labels", &label_path, "name of file containing labels")
      };
      std::string usage = tensorflow::Flags::Usage(argv[0], flag_list);
      const bool parse_result = tensorflow::Flags::Parse(&argc, argv, flag_list);
      if (!parse_result) {
        LOG(ERROR) << usage;
        return -1;
      }

    //需要用这句话来全局初始化tensorflow,不加这句话好像也可以执行
      //猜测是为了看解析参数后还有多少参数
      tensorflow::port::InitMain(argv[0], &argc, &argv);
      if (argc > 1) {
        LOG(ERROR) << "Unknown argument " << argv[1] << "\n" << usage;
        return -1;
      }

    ImageClassifier iwatchClassifier;
    std::vector<std::string> outputLabels;
    std::vector<float> outputscores;
    iwatchClassifier.setInputHeight(224);
    iwatchClassifier.setInputWidth(224);
    iwatchClassifier.setInputDepth(3);
    iwatchClassifier.setInputStdVal(-255.0);
    iwatchClassifier.setInputMeanVal(255);
    iwatchClassifier.setInputScaleVal(2.0);
    iwatchClassifier.setInputSubVal(0.5);
    iwatchClassifier.setInputLayer("inputs/inputs");
    iwatchClassifier.setOutputLayer("Output/predict");
    iwatchClassifier.readLabelsFile(label_path);
    iwatchClassifier.loadGraph(graph_path);
    std::string rootDir = "/home/lilanluo/jetson_workspace/data/iwatch_aotudian/val/NG";
    QTime time;
    QDir dir(QString::fromStdString(rootDir));
    QStringList strList = dir.entryList(QDir::Files);
    int total_image = strList.length();
     double total_time = 0.0;
     for(int index=0; index < 101; index ++)
     {
          std::string image_path = tensorflow::io::JoinPath(rootDir, strList[0].toStdString());
          std::cout << image_path << "\n";
          time.start();
          iwatchClassifier.testSingleImage(image_path, outputLabels, outputscores);
          if(index) //初次运行不计算
          {
              total_time += time.elapsed();

          }
          std::vector<float>::iterator biggest = std::max_element(std::begin(outputscores), std::end(outputscores));
          auto pos = std::distance(std::begin(outputscores), biggest);
          LOG(INFO) << "\n predict label is " << outputLabels[pos] << ", score = " << outputscores[pos];
     }
      qDebug()<< " avg time = "<<total_time/ 100 <<"ms" << "\n";
    return a.exec();
}
