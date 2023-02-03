//
// Device Analyzer Pass
//
// TODO: Write description of pass.
// TODO: Check if all those includes are really necessary.
//

#include "llvm/Pass.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/VersionTuple.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/ScalarEvolution.h"

#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/InstVisitor.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/User.h"
#include "llvm/Transforms/IPO/PassManagerBuilder.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Utils.h"

#include <algorithm>
#include <vector>
#include <string>
#include <set>
#include <bitset>
#include <stdexcept>
#include <iterator>
#include <map>
#include <stack>
#include <fstream>

#include "deviceAnalyzer.h"
#include "traceback.h"
#include "DeviceUtils.h"
#include "recurrence.h"
#include "node.h"
#include "kernelinfo.h"

using namespace llvm;


namespace mekong {

bool usesNewKernelLaunch(llvm::Module &m) {
  return m.getSDKVersion() >= VersionTuple(9, 2);
}

} // namespace mekong


//
// ENTRY POINT OF LLVM
//
bool DeviceAnalyzer::runOnModule(Module& M)
{
    if (M.getTargetTriple().find("nvptx64") == std::string::npos)
        return false;

    errs() << "Hello from device code!\n";
    std::vector<Function*> kernels; 
    mekong::getKernels(M, kernels);


    errs() << "num kernels: " << kernels.size() << '\n';

    std::ofstream block_data ("block_id.out", std::ios::app);
    assert(block_data.is_open() && "Cannot open file to store block IDs.\n");

    std::ofstream kernelExceptionFile ("kernel_exceptions.out");
    assert(kernelExceptionFile.is_open() && "Cannot open file to store unanalyzable kernels.\n");

    // Procedure:
    // For each kernel in kernelNames,
    //  - traverse through list of Basic Blocks
    //  - check if block is terminated by conditional branch -> insert in data structure
    //  - check if condition depends on ID -> constrain thread grid
    for (auto& func : kernels)
    {
        if (func == nullptr)
            continue;

        LoopInfo& loopInfo = getAnalysis<LoopInfoWrapperPass>(*func).getLoopInfo();
        ScalarEvolution& SE = getAnalysis<ScalarEvolutionWrapperPass>(*func).getSE();
        KernelInfo kernel (func, loopInfo, SE);

        try
        {
            kernel.traverseCFG();
        }
        catch (std::exception& ex)
        {
            errs() << ex.what() << '\n';
            errs() << "Kernel " << kernel.getName() << " fails to be analyzed.\n";
            kernelExceptionFile << kernel.getName() << '\n';
            continue;
        }

        kernel.getExactAdjacencyMatrix();
        kernel.writeCFGToFile(false);
        kernel.writeCFGToFile(true);

        kernel.writeBlockIDs(block_data);
    }  // end loop over kernels

    block_data.close();

    return false;
}


void DeviceAnalyzer::getAnalysisUsage(AnalysisUsage& AU) const {
    AU.addRequired<LoopInfoWrapperPass>();
    AU.addRequired<ScalarEvolutionWrapperPass>();
    AU.setPreservesAll();
}
