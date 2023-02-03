#ifndef TRACEBACK_H
#define TRACEBACK_H

#include <vector>
#include <algorithm>
#include <exception>
#include "llvm/IR/InstVisitor.h"
#include "llvm/IR/Value.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/raw_ostream.h"

class AlreadyVisitedException : public std::runtime_error
{
public:
    AlreadyVisitedException() : std::runtime_error("PHINode already visited.") {}
};

class DataDependencyException : public std::runtime_error
{
public:
    DataDependencyException() : std::runtime_error("Data dependency detected.") {}
};

using BBIter = std::vector<llvm::BasicBlock*>::const_iterator;
unsigned getBasicBlockIndex(BBIter begin, BBIter end, llvm::BasicBlock* bb);
bool isNVVMIntrinsic(const llvm::StringRef functionName);
std::string getIntrinsicName(const llvm::StringRef functionName);
std::string predicateAsString(llvm::CmpInst::Predicate pred);
std::string convertConstantFPToString(llvm::ConstantFP* cFP);
std::string ifConstantToString(llvm::Value* val);


struct cudaIdx {
    const std::string threadIdx = "llvm.nvvm.read.ptx.sreg.tid.";
    const std::string blockDim  = "llvm.nvvm.read.ptx.sreg.ntid.";
    const std::string blockIdx  = "llvm.nvvm.read.ptx.sreg.ctaid.";
    const std::string gridDim   = "llvm.nvvm.read.ptx.sreg.nctaid.";
};

struct tracebackVisitor : public llvm::InstVisitor<tracebackVisitor>
{
    std::string valAsString;
    const std::vector<llvm::Value*> kernelArgs;
    const std::vector<llvm::BasicBlock*> basicBlocks;
    const std::vector<llvm::PHINode*> indvars;
    std::vector<llvm::PHINode*> visitedPHIs;
    cudaIdx idx;

    tracebackVisitor(const std::vector<llvm::Value*>& args,
                     const std::vector<llvm::BasicBlock*>& bbList,
                     const std::vector<llvm::PHINode*>& _indvars)
        : kernelArgs(args), basicBlocks(bbList), indvars(_indvars) {}

    tracebackVisitor(const tracebackVisitor &visitor)
        : kernelArgs (visitor.kernelArgs),
          basicBlocks (visitor.basicBlocks),
          indvars (visitor.indvars)
        {}

    void visitAllocaInst(llvm::AllocaInst& alloc)
    {
        // Check for the allocated type:
        // PointerType indicates data passed to the kernel
        //  -> should not be relevant for condition analysis!
        if (alloc.getAllocatedType()->isPointerTy())
        {
            return;
        }

        for (auto user : alloc.users())
        {
            if (auto store = llvm::dyn_cast<llvm::StoreInst>(user))
            {
                // Check if a value is stored to the allocated memory
                // and not vice versa.
                if (store->getValueOperand() == &(llvm::cast<llvm::Value>(alloc)))
                {
                    continue;
                }

                valAsString = tracebackValue(store->getValueOperand());
            }
        }
    }

    void visitPHINode(llvm::PHINode& phi)
    {
        checkIfVisited(&phi);
        visitedPHIs.push_back(&phi);

        std::string name = phi.getName().str();
        name.erase (std::remove(name.begin(), name.end(), '.'), name.end());
        name.erase (std::remove(name.begin(), name.end(), '-'), name.end());
        std::string valStrTemp = name + "PHI";
        auto indvarIt = std::find(indvars.begin(), indvars.end(), &phi);
        if (indvarIt != indvars.end())
        {
            valAsString = valStrTemp;
            visitedPHIs.pop_back();
            return;
        }

        unsigned idx = getBasicBlockIndex(basicBlocks.begin(), basicBlocks.end(), phi.getParent());
        valStrTemp += "(" + std::to_string(idx) + "$";

        for (unsigned i=0; i<phi.getNumIncomingValues(); ++i)
        {
            idx = getBasicBlockIndex(basicBlocks.begin(), basicBlocks.end(), phi.getIncomingBlock(i));

            valStrTemp += "(" + std::to_string(idx) + ":"
                    + tracebackValue(phi.getIncomingValue(i)) + ")";
        }

        valAsString = valStrTemp + ")";
        visitedPHIs.pop_back();
    }

    void visitLoadInst(llvm::LoadInst& ld)
    {
        valAsString = tracebackValue(ld.getPointerOperand());
    }

    void visitCmpInst(llvm::CmpInst& cmp)
    {
        std::string operatorName = predicateAsString(cmp.getPredicate());
        std::string op0Name = tracebackValue(cmp.getOperand(0));
        std::string op1Name = tracebackValue(cmp.getOperand(1));
        valAsString = "("+op0Name+" "+operatorName+" "+op1Name+")";
    }

    void visitBinaryOperator(llvm::BinaryOperator& binOp)
    {
        std::string operatorName = binOp.getOpcodeName();
        std::string op0Name = tracebackValue(binOp.getOperand(0));
        std::string op1Name = tracebackValue(binOp.getOperand(1));
        valAsString = "("+op0Name+" "+operatorName+" "+op1Name+")";
    }

    void visitCallInst(llvm::CallInst& call)
    {
        llvm::Function* func = call.getCalledFunction();
        llvm::StringRef funcName = func->getName();
        char coord = funcName.back();

        if (!func->isIntrinsic())
        {
            llvm::errs() << "Function " << funcName << " is not an intrinsic.\n"
                         << "Possibly increase inline threshold.\n";
            return;
        }


        if (funcName.find(idx.threadIdx) != std::string::npos)
        {
            valAsString = "ti";
            valAsString.push_back(coord);
        }
        else if (funcName.find(idx.blockIdx) != std::string::npos)
        {
            valAsString = "bi";
            valAsString.push_back(coord);
        }
        else if (funcName.find(idx.blockDim) != std::string::npos)
        {
            valAsString = "bd";
            valAsString.push_back(coord);
        }
        else if (funcName.find(idx.gridDim) != std::string::npos)
        {
            valAsString = "gd";
            valAsString.push_back(coord);
        }
        else
        {
            std::string valStrTemp = getIntrinsicName(funcName) + "(";
            for (unsigned i=0; i<call.getNumArgOperands(); ++i)
            {
                if (i>0)
                    valStrTemp += ',';

                llvm::Value* op = call.getArgOperand(i);
                std::string opStr = tracebackValue(op);
                valStrTemp += opStr;
            }
            valAsString = valStrTemp + ")";
        }
    }

    void visitCastInst(llvm::CastInst& cast)
    {
        valAsString = tracebackValue(cast.getOperand(0));
    }

    void visitSelectInst(llvm::SelectInst& slct)
    {
        std::string cond = tracebackValue(slct.getCondition());
        std::string trueVal = tracebackValue(slct.getTrueValue());
        std::string falseVal = tracebackValue(slct.getFalseValue());
        valAsString = "sel(" + cond + "#" + trueVal + "#" + falseVal + ")";
    }

    void visitGetElementPtrInst(llvm::GetElementPtrInst& gep)
    {
        throw DataDependencyException();
    }

    void clearValAsString() { valAsString.clear(); }

    void clearVisitedPHIs() { visitedPHIs.clear(); }

    void clearVisitor()
    {
        clearValAsString();
        clearVisitedPHIs();
    }

    std::string tracebackValue(llvm::Value* val)
    {
        std::string valStr = ifConstantToString(val);

        auto isArgIter = std::find(kernelArgs.begin(), kernelArgs.end(), val);
        if (isArgIter != kernelArgs.end())
        {
            if (val->getType()->isPointerTy())
            {
                throw DataDependencyException();
            }
            valStr += val->getName();
            return valStr;
        }

        if (llvm::Instruction* inst = llvm::dyn_cast<llvm::Instruction>(val))
        {
            this->visit(inst);
            valStr += valAsString;
        }

        return valStr;
    }

    void checkIfVisited(llvm::PHINode* phi) const
    {
        auto it = std::find(visitedPHIs.begin(), visitedPHIs.end(), phi);
        if (it != visitedPHIs.end())
        {
            throw AlreadyVisitedException();
        }
    }
};


#endif // TRACEBACK_H
