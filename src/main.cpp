#include "lenet5_rt/lenet5_rt.h"

#include <iostream>
#include <string>

int main(int argc, char** argv)
{
    if (argc != 3)
    {
        std::cout << "./run_lenet5_inf <path_to_uff file> <pgm_file>" << std::endl;
        return -1;
    }

    NetParams netParams;
    netParams.inputTensorNames = {"conv1_input"};
    netParams.outputTensorNames = {"dense3/Softmax"};
    netParams.uffFileName = std::string(argv[1]);
    netParams.maxBatchSize = 1;
    LeNet5RT lenet5_rt(netParams);
    
    if (!lenet5_rt.init())
    {
        std::cout << "Failed to initialize the model" << std::endl;
        return -1;
    }

    std::cout << "========================= LeNet5 Initialized succesfully =========================" << std::endl;

    lenet5_rt.infer(std::string(argv[2]));

    return 0;
}