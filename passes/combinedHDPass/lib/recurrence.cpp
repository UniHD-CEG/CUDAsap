#include "recurrence.h"
#include "traceback.h"

using boolPair = std::pair<bool, bool>;

std::string ExtendedRecurrence::getAsString(tracebackVisitor& visitor)
{
    std::string condString;
    if (initVal == nullptr || stepInst == nullptr || stepVal == nullptr)
    {
        llvm::errs() << "Cannot determine recurrence.\n";
        // TODO: Throw exception!
        return condString;
    }

    if (finalVal == nullptr)
    {
        Recurrence pureRecurrence = static_cast<Recurrence>(*this);
        condString = pureRecurrence.getAsString(visitor);
    }
    else
    {
        std::string initValStr = visitor.tracebackValue(initVal);
        std::string finalValStr = visitor.tracebackValue(finalVal);
        std::string stepInstStr = stepInst->getOpcodeName();
        std::string stepValStr = visitor.tracebackValue(stepVal);
        std::string guardFlag = isFootControlled ? "1" : "0";

        // Append to finalValStr if the final value is included.
        finalValStr += getCorrection();
        condString = "{"  + initValStr
                   + ","  + stepInstStr
                   + ","  + stepValStr
                   + "}," + finalValStr
                   + "," + guardFlag;
    }
    return condString;
}


std::string ExtendedRecurrence::getCorrection()
{
    std::string correction = "";
    bool ifAdd = includeFinalVal.first;
    bool ifSub = includeFinalVal.second;
    switch (stepInst->getOpcode()) {
    case 13:    // add
    case 14:    // fadd
    case 17:    // mul
    case 18:    // fmul
    case 25:    // shl
        if (ifAdd)
            correction = "+1";
        break;
    case 15:    // sub
    case 16:    // fsub
    case 19:    // udiv
    case 20:    // sdiv
    case 21:    // fdiv
    case 26:    // lshr
        if (ifSub)
            correction = "-1";
        break;
    default:
        break;
    }
    return correction;
}


bool isIncomingValue(const llvm::PHINode* phi, const llvm::Value* val)
{
    for (const auto& inVal : phi->incoming_values())
    {
        if (inVal == val)
        {
            return true;
        }
    }
    return false;
}


bool isIndvarIncrementOperation(const llvm::BinaryOperator* binOp)
{
    bool isIncr = false;
    unsigned opcode = binOp->getOpcode();
    switch (opcode) {
    case 13: ;
    case 14: ;
    case 15: ;
    case 16: ;
    case 17: ;
    case 18: ;
    case 19: ;
    case 20: ;
    case 21: ;
    case 25: ;
    case 26: ;
    case 27:
        isIncr = true;
        break;
    default:
        break;
    }
    return isIncr;
}


// Purpose: Mimic the function Loop::isAuxiliaryInductionVariable w/o usage of SE.
// Resulting PHINode
//  1) is in header,
//  2) is not used outside the loop,
//  3) is incremented by a loop invariant step for each loop iteration and
//  4) has 'add' or 'sub' as step instruction.
// Note on 4): apparently, LLVM is not consistent here. For instance, the kernel
//             'reduce' of rodinia::srad-v1 contains a loop with bitshifting
//             the induction variable.
std::vector<llvm::PHINode*> getAllAuxiliaryInductionVariables(llvm::Loop* l)
{
    std::vector<llvm::PHINode*> result;
    llvm::BasicBlock* header = l->getHeader();
    // condition 1)
    for (auto& phi : header->phis())
    {
        for (auto user : phi.users())
        {
            llvm::Instruction* userInst = llvm::cast<llvm::Instruction>(user);
            // condition 2)
            if (!l->contains(userInst->getParent()))
            {
                break;
            }

            if (llvm::BinaryOperator* binOp = llvm::dyn_cast<llvm::BinaryOperator>(userInst))
            {
                // condition 4)
                if (!isIndvarIncrementOperation(binOp))
                {
                    continue;
                }

                // condition 3)
                if (l->isLoopInvariant(binOp->getOperand(0)) || l->isLoopInvariant(binOp->getOperand(1)))
                {
                    // check if the result of binOp is used as an incoming value
                    if (isIncomingValue(&phi, llvm::dyn_cast<llvm::Value>(binOp)))
                    {
                        result.push_back(&phi);
                    }
                }
            }
        }
    }
    return result;
}


bool isMainLoopIndvar(llvm::PHINode* indvar, llvm::Instruction* inst, unsigned recursionDepth)
{
    bool retVal = false;
    if (recursionDepth < 5)
    {
        for (llvm::Use &U : inst->operands())
        {
            llvm::Value* v = U.get();
            if (llvm::PHINode* phi = llvm::dyn_cast<llvm::PHINode>(v))
            {
                if (phi == indvar)
                {
                    retVal = true;
                    break;
                }
            }

            if (llvm::Instruction* nextInst = llvm::dyn_cast<llvm::Instruction>(v))
            {
                retVal = isMainLoopIndvar(indvar, nextInst, recursionDepth + 1);
            }
        }
    }

    return retVal;
}


unsigned getIndexOfLoopExitInBranch(llvm::BranchInst* br, llvm::Loop* l)
{
    unsigned exitIdx = -1;
    for (unsigned idx = 0; idx < br->getNumSuccessors(); ++idx)
    {
        llvm::BasicBlock* succ = br->getSuccessor(idx);
        if (l->contains(succ))
        {
            continue;
        }
        exitIdx = idx;
    }
    return exitIdx;
}


boolPair isFinalValIncluded(unsigned finalValIdx,
                            unsigned exitIdx,
                            std::string pred)
{
    bool ifAdd = false;
    bool ifSub = false;
    unsigned configID = 2*finalValIdx + exitIdx;
    switch (configID) {
    case 0:
        ifAdd = (pred == "<");
        ifSub = (pred == ">");
        break;
    case 1:
        ifAdd = (pred == ">=");
        ifSub = (pred == "<=");
        break;
    case 2:
        ifAdd = (pred == ">");
        ifSub = (pred == "<");
        break;
    case 3:
        ifAdd = (pred == "<=");
        ifSub = (pred == ">=");
        break;
    default:
        break;
    }

    return std::make_pair(ifAdd, ifSub);
}


// Returns a Value* if indvar is the main loop indvar, else nullptr.
std::pair<llvm::Value*, boolPair> getFinalIVValue(llvm::PHINode* indvar, llvm::Loop* l)
{
    llvm::Value* finalVal = nullptr;
    llvm::BasicBlock* latch = l->getLoopLatch();
    llvm::BasicBlock* header = l->getHeader();
    bool isFootControlled = l->isLoopExiting(latch);
    bool isHeaderControlled = l->isLoopExiting(header);
    llvm::BranchInst* loopBranch = nullptr;
    if (isFootControlled)
    {
        loopBranch = llvm::dyn_cast<llvm::BranchInst>(latch->getTerminator());
    }
    else if (isHeaderControlled)
    {
        loopBranch = llvm::dyn_cast<llvm::BranchInst>(header->getTerminator());
    }
    else
    {
        llvm::errs() << "couldn't find loop condition...\n";
        // maybe throw here
    }

    if (loopBranch->isUnconditional())
    {
        llvm::errs() << "branch is unconditional: couldn't find loop condition 2...\n";
        // maybe throw here
    }

    boolPair isIncluded;
    if (llvm::CmpInst* cmp = llvm::dyn_cast<llvm::CmpInst>(loopBranch->getCondition()))
    {
        for (unsigned i=0; i<2; ++i)
        {
            if (llvm::Instruction* inst = llvm::dyn_cast<llvm::Instruction>(cmp->getOperand(i)))
            {
                if (isMainLoopIndvar(indvar, inst))
                {
                    std::string pred = predicateAsString(cmp->getPredicate());
                    unsigned exitIdx = getIndexOfLoopExitInBranch(loopBranch, l);
                    unsigned finalValIdx = i ^ 1;
                    isIncluded = isFinalValIncluded(finalValIdx,
                                                    exitIdx,
                                                    pred);
                    finalVal = cmp->getOperand(finalValIdx);
                }
            }
        }
    }

    return std::make_pair(finalVal, isIncluded);
}


std::shared_ptr<ExtendedRecurrence> getRecurrenceForPHI(llvm::PHINode* indvar, llvm::Loop* l)
{
    llvm::BasicBlock* predecessor = l->getLoopPredecessor();
    llvm::BasicBlock* latch = l->getLoopLatch();
    ExtendedRecurrence rec;
    llvm::Value* initVal = indvar->getIncomingValueForBlock(predecessor);
    rec.initVal = initVal;
    rec.isFootControlled = l->isLoopExiting(latch);

    for (auto user : indvar->users())
    {
        llvm::Instruction* userInst = llvm::cast<llvm::Instruction>(user);

        if (llvm::BinaryOperator* binOp = llvm::dyn_cast<llvm::BinaryOperator>(userInst))
        {
            if (!isIndvarIncrementOperation(binOp))
            {
                continue;
            }
            if (!isIncomingValue(indvar, llvm::dyn_cast<llvm::Value>(binOp)))
            {
                continue;
            }

            unsigned opIndex = 0;
            for (unsigned i=0; i<2; ++i)
            {
                if (indvar == llvm::dyn_cast_or_null<llvm::PHINode>(binOp->getOperand(i)))
                {
                    continue;
                }
                opIndex = i;
            }
            llvm::Instruction* stepInst = binOp;
            llvm::Value* stepVal = binOp->getOperand(opIndex);
            rec.stepInst = stepInst;
            rec.stepVal = stepVal;
        }
    }
    std::pair<llvm::Value*, boolPair> finalInfo = getFinalIVValue(indvar, l);
    rec.finalVal = finalInfo.first;
    rec.includeFinalVal = finalInfo.second;

    return std::make_shared<ExtendedRecurrence>(rec);
}
