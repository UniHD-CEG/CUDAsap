#ifndef RECURRENCE_H
#define RECURRENCE_H

#include "llvm/IR/Value.h"
#include "llvm/Analysis/LoopInfo.h"

#include <vector>
#include <string>
#include <utility>
#include "traceback.h"


// Note: More generally spoken, the initial and the step value may also be
//       recurrences.
struct Recurrence
{
    llvm::Value* initVal;
    llvm::Value* stepVal;
    llvm::Instruction* stepInst;

    Recurrence() : initVal(nullptr), stepVal(nullptr), stepInst(nullptr) {}
    Recurrence(llvm::Value* _initVal, llvm::Value* _stepVal, llvm::Instruction* _stepInst)
    {
        initVal = _initVal;
        stepVal = _stepVal;
        stepInst = _stepInst;
    }

    virtual std::string getAsString(tracebackVisitor& visitor)
    {
        std::string recName;
        if (initVal != nullptr && stepVal != nullptr && stepInst != nullptr)
        {
            recName = "{" + visitor.tracebackValue(initVal)
                    + "," + stepInst->getOpcodeName()
                    + "," + visitor.tracebackValue(stepVal) + "}";
        }
        return recName;
    }
};


struct ExtendedRecurrence : Recurrence
{
    llvm::Value* finalVal;
    bool isFootControlled;
    std::string indvarName;
    std::pair<bool, bool> includeFinalVal;

    ExtendedRecurrence()
    {
        initVal = nullptr;
        finalVal = nullptr;
        stepVal = nullptr;
        stepInst = nullptr;
        isFootControlled = false;
        indvarName = "";
    }
    ExtendedRecurrence(llvm::Value* _initVal, llvm::Value* _finalVal,
                   llvm::Value* _stepVal, llvm::Instruction* _stepInst,
                   bool _footControlled, std::string _indvarName)
    {
        initVal = _initVal;
        finalVal = _finalVal;
        stepVal = _stepVal;
        stepInst = _stepInst;
        isFootControlled = _footControlled;
        indvarName = _indvarName;
    }

    std::string getAsString(tracebackVisitor& visitor) override;
    std::string getCorrection();
};


struct IndvarInfo
{
    llvm::PHINode* indvar;
    std::shared_ptr<ExtendedRecurrence> rec;

    IndvarInfo() : indvar(nullptr), rec(nullptr) {}
    IndvarInfo(llvm::PHINode* _indvar, std::shared_ptr<ExtendedRecurrence> _rec) : indvar(_indvar), rec(_rec) {}
};


// Maybe use inheritance instead.
struct LoopInfoCustom
{
    llvm::Loop* loop;
    std::vector<IndvarInfo> indvars;
    unsigned mainIndvarIndex;

    LoopInfoCustom() : loop(nullptr) {}

    LoopInfoCustom(llvm::Loop* _l, std::vector<IndvarInfo> _indvars, unsigned _mainIndvarIndex)
        : loop(_l), indvars(_indvars), mainIndvarIndex(_mainIndvarIndex) {}

    IndvarInfo getMainIndvar() const { return indvars[mainIndvarIndex]; }
};


bool isIncomingValue(const llvm::PHINode* phi, const llvm::Value* val);
bool isIndvarIncrementOperation(const llvm::BinaryOperator* binOp);
std::vector<llvm::PHINode*> getAllAuxiliaryInductionVariables(llvm::Loop* l);
bool isMainLoopIndvar(llvm::PHINode* indvar, llvm::Instruction* inst, unsigned recursionDepth = 0);
std::pair<llvm::Value *, std::pair<bool, bool>> getFinalIVValue(llvm::PHINode* indvar, llvm::Loop* l);
std::shared_ptr<ExtendedRecurrence> getRecurrenceForPHI(llvm::PHINode* indvar, llvm::Loop* l);


#endif  // RECURRENCE_H
