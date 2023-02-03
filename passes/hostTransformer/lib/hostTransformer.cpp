//
// HOST TRANSFORMER PASS
//

#include "llvm/Pass.h"
#include "llvm/Support/raw_ostream.h"

#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/User.h"
#include "llvm/Transforms/IPO/PassManagerBuilder.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/ScalarEvolution.h"

#include "hostTransformer.h"
#include "hostRuntime.h"
#include "IRUtils.h"
#include "recurrence.h"
#include "kernellaunch.h"

#include <algorithm>
#include <vector>
#include <map>
#include <array>
#include <string>
#include <set>
#include <fstream>
#include <exception>
#include <cstddef>
#include <memory>

using namespace llvm;


int getCUDAVersion(Module &M, std::pair<unsigned, unsigned>& version)
{
    auto cudaVersionMD = M.getModuleFlag("SDK Version");
    if (cudaVersionMD == nullptr)
    {
        errs() << "No SDK version found. Terminate.\n";
        return 1;
    }
    Value* cudaVersion = cast<ValueAsMetadata>(cudaVersionMD)->getValue();
    auto cudaMajor = cast<ConstantDataArray>(cudaVersion)->getElementAsInteger(0u);
    auto cudaMinor = cast<ConstantDataArray>(cudaVersion)->getElementAsInteger(1u);

    version = std::make_pair(cudaMajor, cudaMinor);
    return 0;
}


// unused
std::vector<std::string> getNamesOfGlobals(Module &M)
{
    std::vector<std::string> names;
    if (M.global_empty())
    {
        errs() << "no globals\n";
    }
    else
    {
        for (auto& g : M.globals())
        {
            names.push_back(g.getName());
        }
    }

    return names;
}


[[deprecated]]
bool getApplyTransformFlag()
{
    std::ifstream host_data ("host.out");
    assert(host_data.is_open() && "Cannot open host file!\n");

    std::string line;
    while (std::getline(host_data, line))
    {
        unsigned pos1 = line.find(',');
        unsigned pos2 = line.find(',', pos1+1);
        std::string applyTransformString = line.substr(pos1+1, pos2-pos1-1);
        if (applyTransformString.compare("0") == 0)
        {
            continue;
        }
        else if (applyTransformString.compare("1") == 0)
        {
            return true;
        }
        else
        {
            throw std::invalid_argument ("applyTransform flag is neither '0' nor '1'.\n");
        }
    }

    return false;
}


// Extract the kernel name from the call of cudaLaunchKernel.
[[deprecated("Kernel name is retrieved in class KernelLaunch.")]]
StringRef extractKernelName(CallBase* CI)
{
    StringRef name;
    Value* kernelOperand = CI->getArgOperand(0);
    if (BitCastOperator* bc = dyn_cast<BitCastOperator>(kernelOperand))
    {
        Value* kernel = bc->getOperand(0);
        if (Function* kernelF = dyn_cast<Function>(kernel))
        {
            name = kernelF->getName();
        }
    }
    return name;
}


// Get a pair of the kernel name and its non-pointer arguments for each
// kernel found with the host analyzer pass.
//
// Idea for future work:
//  customize notation, i.e. include a header row in the output of hostAnalyzer
//  where the argIdx bracket is specified (in this case '[]')
// unused
std::map<std::string, std::vector<unsigned>> getKernelInfo()
{
    std::ifstream host_data ("host.out");
    assert(host_data.is_open() && "Cannot open host file!\n");

    std::map<std::string, std::vector<unsigned>> kernelInfo;
    std::string line;
    while (std::getline(host_data, line))
    {
        std::string::size_type pos1 = line.find(',');
        std::string kernelName = line.substr(0, pos1);
        std::vector<unsigned> argIdxList;
        pos1 = line.find('[');
        while (pos1 != std::string::npos)
        {
            std::string::size_type pos2 = line.find(']', pos1);
            argIdxList.push_back(std::stoul(line.substr(pos1+1, pos2 - (pos1+1))));
            pos1 = line.find('[', pos1+1);
        }

        kernelInfo[kernelName] = argIdxList;
    }

    return kernelInfo;
}


std::vector<Function*> findLaunchWrappers(Function* cudaLaunchKernel)
{
    std::vector<Function*> wrappers;
    for (auto user : cudaLaunchKernel->users())
    {
        CallBase* CI = dyn_cast_or_null<CallBase>(user);
        if (CI == nullptr)
        {
            continue;
        }

        Function* kernel = nullptr;
        extractKernel(CI, kernel);
        if (kernel == nullptr)
        {
            continue;
        }

        StringRef name = kernel->getName();
        BasicBlock* launchBlock = CI->getParent();
        Function* kernelCaller = launchBlock->getParent();
        if (kernelCaller->getName() == name)  // is the kernel wrapper
        {
            wrappers.push_back(kernelCaller);
        }
    }

    return wrappers;
}


void inlineLaunchWrapper(Function* cudaLaunchKernel)
{
    std::vector<Function*> wrappers = findLaunchWrappers(cudaLaunchKernel);
    for (auto& wrap : wrappers)
    {
        for (auto user : wrap->users())
        {
            CallBase* CB = dyn_cast_or_null<CallBase>(user);
            if (CB == nullptr)
            {
                continue;
            }
            InlineFunctionInfo ifi;
            InlineFunction(CB, ifi);
        }
    }
}


std::vector<std::string> getExceptionList()
{
    std::ifstream kernelExceptionFile ("kernel_exceptions.out");
    assert (kernelExceptionFile.is_open() && "Cannot open exception file.\n");
    std::string line;
    std::vector<std::string> kernelExceptions;
    while (std::getline(kernelExceptionFile, line))
    {
        kernelExceptions.push_back(line);
    }

    return kernelExceptions;
}


void writeKernelNames(std::vector<std::string> kernelNames)
{
    std::ofstream kernelNamesFile ("host.out", std::ios::out | std::ios::app);
    assert(kernelNamesFile.is_open() && "Unable to open output file.\n");
    for (const auto& name : kernelNames)
    {
        kernelNamesFile << name << '\n';
    }
}


bool HostTransformer::runOnModule(Module &M)
{
    if (M.getTargetTriple().find("x86_64") == std::string::npos)
        return false;
    errs() << "Hello from host code!\n";

    std::pair<unsigned, unsigned> cudaVersion;
    int success = getCUDAVersion(M, cudaVersion);
    if (success != 0)
    {
        errs() << "Cannot determine CUDA SDK Version. Terminate.\n";
        return false;
    }

    unsigned cudaMajor = cudaVersion.first;
    unsigned cudaMinor = cudaVersion.second;

    errs() << "CUDA version: " << cudaMajor << '.' << cudaMinor << '\n';

    // Link Host Runtime //
    // Load Memory Buffer from Headerfile
    // Note: hostRuntime_ll and hostRuntime_ll_len are defined in hostRuntime.h.
    std::string hostRuntimeNull((const char *)hostRuntime_ll, hostRuntime_ll_len);
    // Add nulltermination
    hostRuntimeNull.append("\0");
    StringRef hostRuntime(hostRuntimeNull.c_str(), hostRuntimeNull.size());
    // Link against current module
    mekong::linkIR(hostRuntime, M);

    Function* cudaLaunchKernelFunc = M.getFunction("cudaLaunchKernel");
    if (cudaLaunchKernelFunc == nullptr)
    {
        errs() << "No kernel launch. Exit.\n";
        return false;
    }

    inlineLaunchWrapper(cudaLaunchKernelFunc);

    // arguments of cudaLaunchKernel:
    // [0] i8*  bitcast of kernel wrapper -> get the name
    // [1] i64  gridDim.x and gridDim.y concatenated
    // [2] i32  gridDim.z
    // [3] i64  blockDim.x and blockDim.y concatenated
    // [4] i32  blockDim.z
    // [5] i8** kernel arguments stored as pointer
    // [6] i64  shared memory size
    // [7] stream ID
    Function* kernelLogger = M.getFunction("logKernelLaunchArguments");
    Function* startClockFunc = M.getFunction("startClock");
    Function* logElapsedTimeFunc = M.getFunction("logElapsedTime");
    Function* callConditionAnalyzerFunc = M.getFunction("callConditionAnalyzerScript");
    Function* writeKernelTimeFunc = M.getFunction("kernelElapsedTime");

    LLVMContext &ctx = M.getContext();
    std::vector<std::string> kernelNamesVec;
    std::vector<std::string> kernelExceptions = getExceptionList();
    bool moduleTransformed = false;

    // main loop
    for (auto user : cudaLaunchKernelFunc->users())
    {
        CallBase* CI = dyn_cast_or_null<CallBase>(user);
        if (CI == nullptr)
        {
            continue;
        }
        Function* kernelCaller = CI->getParent()->getParent();
        LoopInfo& loopInfo = getAnalysis<LoopInfoWrapperPass>(*kernelCaller).getLoopInfo();

        std::unique_ptr<KernelLaunch> launch;
        try
        {
            launch = std::make_unique<KernelLaunch>(KernelLaunch (CI, kernelExceptions, loopInfo));
        }
        catch (ConstructorFailureException& ex)
        {
            errs() << ex.what() << '\n';
            continue;
        }

        kernelNamesVec.push_back(launch->getName().str());

        errs() << "Is the kernel indvar dependent? ";
        if (launch->isIndvarDependent)
            errs() << "Yes.\n";
        else
            errs() << "No.\n";

        launch->insertParameterExtraction(ctx, startClockFunc,
                                          kernelLogger,
                                          logElapsedTimeFunc,
                                          callConditionAnalyzerFunc);
        moduleTransformed |= launch->extractionInserted;
    }

    writeKernelNames(kernelNamesVec);

    return moduleTransformed;
}


void HostTransformer::getAnalysisUsage(AnalysisUsage& AU) const {
    AU.addRequired<LoopInfoWrapperPass>();
    AU.setPreservesAll();
}
