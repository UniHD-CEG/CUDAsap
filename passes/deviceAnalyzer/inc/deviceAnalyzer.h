#ifndef DEVICEANALYZER_H
#define DEVICEANALYZER_H

#include "llvm/Pass.h"


struct DeviceAnalyzer : llvm::ModulePass
{
    static char ID;
    DeviceAnalyzer() : llvm::ModulePass(ID) {}

    virtual ~DeviceAnalyzer() {}

    virtual bool runOnModule(llvm::Module &M) override;

    virtual void getAnalysisUsage(llvm::AnalysisUsage& AU) const override;
};

#endif // DEVICEANALYZER_H
