#include "kernellaunch.h"

using namespace llvm;


// Maybe add an argument "depth" since the kernel call might be not completed nested.
void getAllLoopIndvars(Loop* loop, std::vector<LoopIndvars>& allLoopIndvars)
{
    std::vector<PHINode*> indvars = getAllAuxiliaryInductionVariables(loop);
    LoopIndvars li (loop, indvars);
    allLoopIndvars.push_back(li);
    std::vector<Loop*> subLoops = loop->getSubLoops();
    if (subLoops.empty())
    {
        return;
    }
    else
    {
        for (auto sl : subLoops)
        {
            getAllLoopIndvars(sl, allLoopIndvars);
        }
    }
}


void findStore(Value* args,
               KernelArguments& argsUnpacked,
               unsigned index,
               const BasicBlock* launchBlock)
{
    if (StoreInst* si = dyn_cast<StoreInst>(args))
    {
        if (si->getParent() == launchBlock)
        {
            Value* storedVal = si->getValueOperand();
            argsUnpacked.args.push_back(storedVal);
            argsUnpacked.indices.push_back(index);
        }
    }
    else
    {
        for (User* u : args->users())
        {
            if (isa<CallBase>(u))
                continue;

            // Get the index of the kernel argument.
            if (GetElementPtrInst* gep = dyn_cast<GetElementPtrInst>(u))
            {
                unsigned n = gep->getNumIndices();
                if(ConstantInt *CI = dyn_cast<ConstantInt>(gep->getOperand(n)))
                {
                    index = CI->getZExtValue();
                }
            }

            findStore(u, argsUnpacked, index, launchBlock);
        }
    }
}


// Strategy:
//  1) get the GEP instruction
//  2) recursively check its users until a store instruction is found for every 1st level user
//  3) trace back operand(0) of each store instruction
KernelArguments unpackKernelArgs(Value* kernelArgsPacked, const BasicBlock* launchBlock)
{
    KernelArguments unpacked;
    if (GetElementPtrInst* gep = dyn_cast<GetElementPtrInst>(kernelArgsPacked))
    {
        auto argsPtr = gep->getPointerOperand();
        findStore(argsPtr, unpacked, 0, launchBlock);
    }

    return unpacked;
}


void getKernelArgsIndicesToInspect(Value* val,
                                   KernelArguments &unpackedArgs,
                                   std::vector<unsigned>& indicesToInspect,
                                   unsigned recursionDepth)
{
    for (User* user : val->users())
    {
        for (Use &use : user->operands())
        {
            Value* v = use.get();
            unsigned idx = unpackedArgs.containsAtIndex(v);
            if (idx != -1)
            {
                indicesToInspect.push_back(idx);
            }
            else
            {
                if (recursionDepth > 2) {return;}
                getKernelArgsIndicesToInspect(user, unpackedArgs, indicesToInspect, recursionDepth+1);
            }
        }
    }
}

std::vector<unsigned> getNonPointerArgumentIndices(const Function* func)
{
    std::vector<unsigned> argIdxList;
    unsigned i=0;
    for (const auto& arg : func->args())
    {
        if (!arg.getType()->isPointerTy())
        {
            argIdxList.push_back(i);
        }
        ++i;
    }
    return argIdxList;
}


std::string getArgNameFromIndex(std::ifstream& cfgFile,
                                const std::vector<unsigned>& indicesToInspect)
{
    std::string cfg_args;
    std::string argName;
    if (std::getline(cfgFile, cfg_args))
    {
        std::string subscriptIdx = "[" + std::to_string(indicesToInspect[0]) + "]";
        std::size_t posStart = cfg_args.find(subscriptIdx) + subscriptIdx.size();
        std::size_t posEnd = cfg_args.find(";", posStart);
        argName = cfg_args.substr(posStart, posEnd-posStart);
    }

    return argName;
}


bool findArgNamePatternInCFG(const std::string argName, std::ifstream& cfgFile)
{
    bool patternFound = false;
    std::string pattern1 = "(" + argName + " ";
    std::string pattern2 = " " + argName + ")";
    std::string pattern3 = "{" + argName + ",";
    std::string pattern4 = "," + argName + "}";
    std::string condition;

    while (std::getline(cfgFile, condition))
    {
        if (condition.find(pattern1) != std::string::npos
                || condition.find(pattern2) != std::string::npos
                || condition.find(pattern3) != std::string::npos
                || condition.find(pattern4) != std::string::npos)
        {
            patternFound = true;
            break;
        }
    }

    return patternFound;
}


unsigned KernelArguments::containsAtIndex(const Value* v) const
{
    unsigned idx = -1;
    for (unsigned i=0; i<args.size(); ++i)
    {
        if (v == args[i])
        {
            idx = indices[i];
            break;
        }
    }
    return idx;
}


KernelLaunch::KernelLaunch()
{
    launchCall = nullptr;
    kernel = nullptr;
    launchBlock = nullptr;
    kernelCaller = nullptr;
    isSane = false;
    isIndvarDependent = false;
    isInstrumented = false;
    extractionInserted = false;
    analyzerCallInserted = false;
}


KernelLaunch::KernelLaunch(CallBase *callInst,
                           const std::vector<std::string>& kernelExceptions,
                           LoopInfo &li)
    : isInstrumented(false), extractionInserted(false), analyzerCallInserted(false)
{
    launchCall = callInst;
    extractKernel(launchCall, kernel);
    launchBlock = launchCall->getParent();
    kernelCaller = launchBlock->getParent();
    this->sanityChecks(kernelExceptions);

    argIdxList = getNonPointerArgumentIndices(kernel);

    // check for looping kernel calls
    this->checkIndvarDependence(li);
}

KernelLaunch::KernelLaunch(const KernelLaunch &other)
{
    launchCall = other.launchCall;
    kernel = other.kernel;
    launchBlock = other.launchBlock;
    kernelCaller = other.kernelCaller;
    argIdxList = other.argIdxList;
    isSane = other.isSane;
    isIndvarDependent = other.isIndvarDependent;
    isInstrumented = other.isInstrumented;
    extractionInserted = other.extractionInserted;
    analyzerCallInserted = other.analyzerCallInserted;
}

KernelLaunch::KernelLaunch(KernelLaunch&& other)
{
    launchCall = other.launchCall;
    kernel = other.kernel;
    launchBlock = other.launchBlock;
    kernelCaller = other.kernelCaller;
    argIdxList = other.argIdxList;
    isSane = other.isSane;
    isIndvarDependent = other.isIndvarDependent;
    isInstrumented = other.isInstrumented;
    extractionInserted = other.extractionInserted;
    analyzerCallInserted = other.analyzerCallInserted;

    other.launchCall = nullptr;
    other.kernel = nullptr;
    other.launchBlock = nullptr;
    other.kernelCaller = nullptr;
    other.argIdxList = std::vector<unsigned>();
    other.isSane = false;
}


void KernelLaunch::sanityChecks(const std::vector<std::string>& kernelExceptions)
{
    isSane = true;
    if (kernel == nullptr)
    {
        errs() << "extraction failed\n";
        isSane = false;
        throw ConstructorFailureException();
    }

    StringRef name = this->getName();
    // At this place, check if the kernel is one of the non-analyzables.
    auto exceptIter = std::find(kernelExceptions.begin(),
                                kernelExceptions.end(),
                                name.str());
    if (exceptIter != kernelExceptions.end())
    {
        errs() << "Kernel cannot be analyzed. Don't insert call of script.\n";
        isSane = false;
        throw ConstructorFailureException();
    }

    if (kernelCaller->getName() == name)  // is the kernel wrapper
    {
        errs() << "is wrapper\n";
        isSane = false;
        throw ConstructorFailureException();
    }
}

StringRef KernelLaunch::getName() const
{
    return kernel->getName();
}

void KernelLaunch::checkIndvarDependence(LoopInfo &li)
{
    LoopInfo loopInfo = std::move(li);
    std::vector<LoopIndvars> allLoopIndvars;
    isIndvarDependent = false;
    if (loopInfo.getLoopDepth(launchBlock) > 0)
    {
        for (auto& loop : loopInfo)
        {
            getAllLoopIndvars(loop, allLoopIndvars);
        }

        KernelArguments unpackedArgs = unpackKernelArgs(launchCall->getArgOperand(5), launchBlock);

        std::vector<unsigned> indicesToInspect;
        for (LoopIndvars& li : allLoopIndvars)
        {
            for (PHINode* phi : li.indvars)
            {
                getKernelArgsIndicesToInspect(phi, unpackedArgs, indicesToInspect, 0);
            }
        }

        if (!indicesToInspect.empty())
        {
            std::ifstream cfg_file (this->getName().str()+"_cfg.out");
            assert(cfg_file.is_open() && "Cannot open cfg file!\n");

            std::string argName = getArgNameFromIndex(cfg_file, indicesToInspect);
            isIndvarDependent = findArgNamePatternInCFG(argName, cfg_file);
        }
    }
}

void KernelLaunch::insertParameterExtraction(LLVMContext &ctx,
                                             Function* startClockFunc,
                                             Function* kernelLogger,
                                             Function* logElapsedTimeFunc,
                                             Function* callConditionAnalyzerFunc)
{
    unsigned numArgs = argIdxList.size();
    IRBuilder<> builder(ctx);

    Instruction* termInst = kernelCaller->getEntryBlock().getTerminator();
    builder.SetInsertPoint(termInst);

    ConstantInt* loopFlag = builder.getInt1(isIndvarDependent);
    ConstantInt* cnt = builder.getInt32(0);
    Value* allocCnt = builder.CreateAlloca(builder.getInt32Ty(), builder.getInt32(1));
    Value* gep = builder.CreateGEP(allocCnt, builder.getInt32(0));
    builder.CreateStore(cnt, gep);

    builder.SetInsertPoint(launchCall);
    CallInst* callClockStart = builder.CreateCall(dyn_cast<Value>(startClockFunc), {});

    Constant* kernelName = builder.CreateGlobalStringPtr(this->getName());

    // store the two indices at which the kernel receives non-pointer args
    Value* numArgsVal = builder.getInt32(numArgs);
    Value* allocArgIdx = builder.CreateAlloca(builder.getInt32Ty(), numArgsVal);
    for (unsigned i=0; i<numArgs; ++i)
    {
        Value* gep = builder.CreateGEP(allocArgIdx, builder.getInt32(i));
        builder.CreateStore(builder.getInt32(argIdxList[i]), gep);
    }

    CallInst* callPrint = builder.CreateCall(dyn_cast<Value>(kernelLogger),
    {kernelName,
     launchCall->getArgOperand(1),
     launchCall->getArgOperand(2),
     launchCall->getArgOperand(3),
     launchCall->getArgOperand(4),
     launchCall->getArgOperand(5),
     allocArgIdx,
     numArgsVal,
     loopFlag,
     allocCnt});

    CallInst* callClockEnd = builder.CreateCall(dyn_cast<Value>(startClockFunc), {});
    CallInst* logTime = builder.CreateCall(dyn_cast<Value>(logElapsedTimeFunc),
    {kernelName, callClockStart, callClockEnd, loopFlag, allocCnt});

    // Make this selectable at compile-time, maybe.
    CallInst* callConditionAnalyzer = builder.CreateCall(dyn_cast<Value>(callConditionAnalyzerFunc),
    {kernelName, loopFlag, allocCnt});
    analyzerCallInserted = true;

    LoadInst* loadCnt = builder.CreateLoad(builder.getInt32Ty(), allocCnt);
    builder.CreateStore(loadCnt, gep);

    extractionInserted = true;
}

// dummy
void KernelLaunch::insertAnalyzerCall(const LoadInst *loadCnt, LLVMContext &ctx)
{

}


void KernelLaunch::insertTimeInstrumentation(LLVMContext &ctx,
                                             Function* startClockFunc,
                                             Function* writeKernelTimeFunc,
                                             Constant* kernelName)
{
    IRBuilder<> builder(ctx);
    Instruction* nextInst = launchCall->getNextNode();
    if (nextInst)
    {
        CallInst* callClockStartKernel = builder.CreateCall(dyn_cast<Value>(startClockFunc), {});
        builder.SetInsertPoint(nextInst);
        CallInst* callWriteKernelTime = builder.CreateCall(dyn_cast<Value>(writeKernelTimeFunc),
        {kernelName, callClockStartKernel});
    }
    else
    {
        errs() << "Cannot instrument kernel launch for time measurement.\n";
    }
}
