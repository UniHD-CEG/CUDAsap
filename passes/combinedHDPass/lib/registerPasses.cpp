#include "hostTransformer.h"
#include "deviceAnalyzer.h"

#include "llvm/IR/LegacyPassManager.h"
#include "llvm/PassSupport.h"
#include "llvm/Transforms/IPO/PassManagerBuilder.h"

using namespace llvm;

char HostTransformer::ID = 0;
char DeviceAnalyzer::ID = 0;

static RegisterPass<HostTransformer> X("host-transform", "Host Transformer Pass", false, false);

static RegisterStandardPasses RegisterHostPass(
    PassManagerBuilder::EP_OptimizerLast,
    [](const PassManagerBuilder &Builder,
       legacy::PassManagerBase &PM) { PM.add(new HostTransformer()); });

static RegisterPass<DeviceAnalyzer> Y("device-analysis", "Device Analyzer Pass", false, false);

static RegisterStandardPasses RegisterDevicePass(
    PassManagerBuilder::EP_OptimizerLast,
    [](const PassManagerBuilder &Builder,
       legacy::PassManagerBase &PM) { PM.add(new DeviceAnalyzer()); });
