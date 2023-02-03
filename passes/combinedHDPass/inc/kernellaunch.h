#ifndef KERNELLAUNCH_H
#define KERNELLAUNCH_H

#include "llvm/Pass.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Operator.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/Analysis/LoopInfo.h"

#include <exception>
#include <fstream>
#include <string>

#include "recurrence.h"


// Extract the Function* of the kernel.
// Note that it is necessary to pass a pointer reference in order to retrieve
// the corresponding pointer. Otherwise, it would be deleted upon termination.
// TODO: Maybe return llvm::Function* instead.
inline void extractKernel(const llvm::CallBase* CI, llvm::Function* &kernelFunc)
{
    llvm::Value* kernelOperand = CI->getArgOperand(0);
    if (llvm::BitCastOperator* bc = llvm::dyn_cast<llvm::BitCastOperator>(kernelOperand))
    {
        llvm::Value* kernelVal = bc->getOperand(0);
        if (llvm::Function* kernelF = llvm::dyn_cast<llvm::Function>(kernelVal))
        {
            kernelFunc = kernelF;
            llvm::errs() << "kernel name " << kernelFunc->getName() << '\n';
        }
    }
}

class ConstructorFailureException : public std::runtime_error
{
public:
    ConstructorFailureException() : std::runtime_error("Sanity check of constructor fails.") {}
};


struct LoopIndvars
{
    llvm::Loop* loop;
    std::vector<llvm::PHINode*> indvars;

    LoopIndvars() {}
    LoopIndvars(llvm::Loop* l_, std::vector<llvm::PHINode*>& phis_)
        : loop(l_), indvars(phis_) {}
    LoopIndvars(const LoopIndvars& li) : loop(li.loop), indvars(li.indvars) {}
};


struct KernelArguments
{
    std::vector<llvm::Value*> args;
    std::vector<unsigned> indices;

    KernelArguments() {}
    KernelArguments(std::vector<llvm::Value*> args_, std::vector<unsigned> idx_)
        : args(args_), indices(idx_) {}
    std::size_t size() const { return args.size(); }
    unsigned containsAtIndex(const llvm::Value* v) const;
};


class KernelLaunch
{
private:
    llvm::CallBase* launchCall;
    llvm::Function* kernel;
    llvm::BasicBlock* launchBlock;
    llvm::Function* kernelCaller;
    std::vector<unsigned> argIdxList;
public:
    KernelLaunch();
    KernelLaunch(llvm::CallBase*, const std::vector<std::string>&, llvm::LoopInfo &);
    KernelLaunch(const KernelLaunch&);
    KernelLaunch(KernelLaunch&&);

    void sanityChecks(const std::vector<std::string>&);
    llvm::StringRef getName() const;
    void checkIndvarDependence(llvm::LoopInfo &);
    void insertTimeInstrumentation(llvm::LLVMContext& ctx,
                                   llvm::Function *startClockFunc,
                                   llvm::Function *writeKernelTimeFunc,
                                   llvm::Constant *kernelName);
    void insertParameterExtraction(llvm::LLVMContext& ctx,
                                   llvm::Function* startClockFunc,
                                   llvm::Function* kernelLogger,
                                   llvm::Function* logElapsedTimeFunc,
                                   llvm::Function* callConditionAnalyzerFunc);
    void insertAnalyzerCall(const llvm::LoadInst*, llvm::LLVMContext&);

    bool isSane;
    bool isIndvarDependent;
    bool isInstrumented;
    bool extractionInserted;
    bool analyzerCallInserted;
};

#endif // KERNELLAUNCH_H
