#include <string>
#include <vector>

#define INPUT_BLOB_NAME "input"

#include <NvInfer.h>
#include <NvUffParser.h>
#include <cuda_runtime_api.h>

struct NetParams 
{
    std::vector<std::string> inputTensorNames, outputTensorNames;
    std::string uffFileName;
    int maxBatchSize;

};



class LeNet5RT
{
    private:
    nvinfer1::ICudaEngine* engine_;
    NetParams netParams_;
    std::vector<void*> inf_buffers_;

    bool constructNetwork(nvuffparser::IUffParser*& parser, nvinfer1::INetworkDefinition*& network);
    bool readInputFile(const std::string& fileName, uint8_t* buffer, int inH, int inW);

    public:
    LeNet5RT(NetParams netParams);
    // LeNet5RT(const std::string& net_file, const std::string& model_file, const std::vector<std::string>& outputs, const std::vector<std::string>& inputs);
    bool init();
    bool infer(const std::string& file_name);

};