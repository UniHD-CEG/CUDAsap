#ifndef HOSTTRANSFORMER_H
#define HOSTTRANSFORMER_H

#include "llvm/Pass.h"


struct HostTransformer : llvm::ModulePass {
  static char ID;
  HostTransformer() : llvm::ModulePass(ID) {}

  bool runOnModule(llvm::Module &M) override;

  virtual void getAnalysisUsage(llvm::AnalysisUsage& AU) const override;
};

#endif // HOSTTRANSFORMER_H
