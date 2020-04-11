#include "lenet5_rt/lenet5_rt.h"

#include <iostream>
#include <fstream>

class Lenet5RTLogger : public nvinfer1::ILogger 
{
    void log(Severity severity, const char *msg) override
    {
        if (severity != Severity::kINFO) 
        {
            std::cout << msg << std::endl;;
        }
    }
} lenet5_logger;

LeNet5RT::LeNet5RT(NetParams netParams)
: netParams_(netParams), inf_buffers_(2)
{

}

bool LeNet5RT::constructNetwork(nvuffparser::IUffParser*& parser, nvinfer1::INetworkDefinition*& network)
{
    if (!parser)
    {
        std::cout << "Parser is a null pointer, please first create and pass in the parser object" << std::endl;
        return false;
    }

    if (!network)
    {
        std::cout << "Please instantiate an empty network" << std::endl;
        return false;
    }
    
    parser->registerInput(netParams_.inputTensorNames[0].c_str(), nvinfer1::Dims3(28, 28, 1), nvuffparser::UffInputOrder::kNHWC);
    parser->registerOutput(netParams_.outputTensorNames[0].c_str());

    parser->parse(netParams_.uffFileName.c_str(), *network, nvinfer1::DataType::kFLOAT);
    return true;
    // nvinfer1::ITensor* input_tensor = network_->addInput(INPUT_BLOB_NAME, nvinfer1::DataType::kFLOAT, nvinfer1::Dims3(IMG_HEIGHT, IMG_WIDTH, NUM_CHANNELS));
    
    // nvinfer1::IConvolutionLayer* conv1 = network->addConvolution(*input_tensor, 6, nvinfer1::DimsHW(5, 5), );
    // conv1->setStride(nvinfer1::DimsHW(1, 1));
}

bool LeNet5RT::init()
{
    nvinfer1::IBuilder* builder = nvinfer1::createInferBuilder(lenet5_logger);
    if (!builder)
    {
        std::cout << "Failed to create builder" << std::endl;
        return false;
    }

    nvinfer1::INetworkDefinition* network = builder->createNetwork();
    if (!network)
    {
        std::cout << "Failed to create network" << std::endl;
        return false;
    }
    nvinfer1::IBuilderConfig* config = builder->createBuilderConfig();
    if (!config)
    {
        std::cout << "Failed to create config" << std::endl;
        return false;
    }

    nvuffparser::IUffParser* parser = nvuffparser::createUffParser();

    if (!parser)
    {
        std::cout << "Failed to create parser" << std::endl;
        return false;
    }
    if(!constructNetwork(parser, network))
    {
        std::cout << "Failed to contruct network" << std::endl;
        return false;
    }

    builder->setMaxBatchSize(netParams_.maxBatchSize);
    config->setMaxWorkspaceSize(16777216); // = 16MiB
    config->setFlag(nvinfer1::BuilderFlag::kGPU_FALLBACK);

    // TODO: try enabling DLA

    engine_ = builder->buildEngineWithConfig(*network, *config);
    if (!engine_)
    {
        std::cout << "Failed to create CUDA engine" << std::endl;
        return false;
    }

    parser->destroy();
    network->destroy();
    config->destroy();
    builder->destroy();
    return true;
}

bool LeNet5RT::readInputFile(const std::string& fileName, uint8_t* buffer, int inH, int inW)
{
    std::ifstream infile(fileName, std::ifstream::binary);
    // assert(infile.is_open() && "Attempting to read from a file that is not open.");
    std::string magic, h, w, max;
    infile >> magic >> h >> w >> max;
    infile.seekg(1, infile.cur);
    infile.read(reinterpret_cast<char*>(buffer), inH * inW);
    return true;
}

bool LeNet5RT::infer(const std::string& fileName)
{
    nvinfer1::IExecutionContext* context = engine_->createExecutionContext();
    if (!context)
    {
        return false;
    }

    const int inputH = 28;
    const int inputW = 28;

    std::vector<uint8_t> fileData(inputH * inputW);
    readInputFile(fileName, fileData.data(), inputH, inputW);

    std::vector<float> hostInputBuffer(inputH * inputW);
    std::vector<float> hostOutputBuffer = {0, 0, 0, 0, 0, 0, 0, 0, 0, 1};
    for (int i = 0; i < inputH * inputW; i++)
    {
        hostInputBuffer[i] = 1.0 - float(fileData[i]) / 255.0;
    }

    for (int i = 0; i < inputH * inputW; i++)
    {
        std::cout << (hostInputBuffer[i] > 0 ? 1 : 0) << (((i + 1) % inputW) ? "" : "\n");
    }
    
    //Creating buffers for input and output on GPU
    int inputTensorBindingIndex = engine_->getBindingIndex(netParams_.inputTensorNames.front().c_str());
    int outputTensorBindingIndex = engine_->getBindingIndex(netParams_.outputTensorNames.front().c_str());

    nvinfer1::Dims3 inputTensorDims = static_cast<nvinfer1::Dims3 &&>(engine_->getBindingDimensions(inputTensorBindingIndex));
    nvinfer1::Dims3 outputTensorDims = static_cast<nvinfer1::Dims3 &&>(engine_->getBindingDimensions(outputTensorBindingIndex));

    int input_count = inputTensorDims.d[0] * inputTensorDims.d[1] * inputTensorDims.d[2] * netParams_.maxBatchSize;
    int output_count = outputTensorDims.d[0] * outputTensorDims.d[1] * outputTensorDims.d[2] * netParams_.maxBatchSize;
    
    cudaMalloc(&inf_buffers_[inputTensorBindingIndex], input_count * sizeof(float));
    cudaMalloc(&inf_buffers_[outputTensorBindingIndex], output_count * sizeof(float));

    cudaStream_t stream;
    cudaStreamCreate(&stream);
    cudaMemcpyAsync(inf_buffers_[inputTensorBindingIndex], hostInputBuffer.data(), hostInputBuffer.size() * sizeof(float), cudaMemcpyHostToDevice, stream);
    context->enqueue(netParams_.maxBatchSize, &inf_buffers_[inputTensorBindingIndex], stream, nullptr);
    cudaMemcpyAsync(hostOutputBuffer.data(), inf_buffers_[outputTensorBindingIndex], hostOutputBuffer.size() * sizeof(float), cudaMemcpyDeviceToHost, stream);
    
    cudaStreamSynchronize(stream);
    cudaFree(&inf_buffers_[inputTensorBindingIndex]);
    cudaFree(&inf_buffers_[outputTensorBindingIndex]);
    cudaStreamDestroy(stream);

    // Get index of the highest probability
    float predicted_char_prob = -1;
    int predicted_char_index = -1;
    for (int i = 0; i < 10; i++)
    {
        if (hostOutputBuffer[i] > predicted_char_prob)
        {
            predicted_char_prob = hostOutputBuffer[i];
            predicted_char_index = i;
        }
    }

    std::cout << "LeNet5 predicted the character to be [" << predicted_char_index << "] with confidence = " << predicted_char_prob << std::endl;

    return true;
}
