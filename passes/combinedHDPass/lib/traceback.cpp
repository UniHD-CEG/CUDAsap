#include "traceback.h"


unsigned getBasicBlockIndex(BBIter begin, BBIter end, llvm::BasicBlock* bb)
{
    auto it = std::find(begin, end, bb);
    if (it == end)
    {
        llvm::errs() << "didn't find the block. Something went wrong...\n";
    }
    unsigned idx = std::distance(begin, it);
    return idx;
}


bool isNVVMIntrinsic(const llvm::StringRef functionName)
{
    return functionName.find("nvvm") != llvm::StringRef::npos;
}


std::string getIntrinsicName(const llvm::StringRef functionName)
{
    // Usual intrinsics are of the form "llvm.<intrinsic>".
    // NVVM intrinsics are of the form "llvm.nvvm.<intrinc>".
    // Thus, the intrinsic name starts at position 5 for the former
    // and at position 10 for the latter.
    unsigned nameBegin = isNVVMIntrinsic(functionName) ? 10 : 5;
    unsigned nameEnd = functionName.find_first_of('.', nameBegin+1);
    std::string intrinsicName = functionName.slice(nameBegin, nameEnd).str();
    return intrinsicName;
}


std::string predicateAsString(llvm::CmpInst::Predicate pred)
{
    std::string cmp;

    switch (pred) {
    case llvm::CmpInst::Predicate::FCMP_OEQ:
    case llvm::CmpInst::Predicate::FCMP_UEQ:
    case llvm::CmpInst::Predicate::ICMP_EQ: cmp += "==";
                             break;
    case llvm::CmpInst::Predicate::FCMP_ONE:
    case llvm::CmpInst::Predicate::FCMP_UNE:
    case llvm::CmpInst::Predicate::ICMP_NE: cmp += "!=";
                             break;
    case llvm::CmpInst::Predicate::FCMP_OGT:
    case llvm::CmpInst::Predicate::FCMP_UGT:
    case llvm::CmpInst::Predicate::ICMP_SGT:
    case llvm::CmpInst::Predicate::ICMP_UGT: cmp += ">";
                              break;
    case llvm::CmpInst::Predicate::FCMP_OLT:
    case llvm::CmpInst::Predicate::FCMP_ULT:
    case llvm::CmpInst::Predicate::ICMP_SLT:
    case llvm::CmpInst::Predicate::ICMP_ULT: cmp += "<";
                              break;
    case llvm::CmpInst::Predicate::FCMP_OGE:
    case llvm::CmpInst::Predicate::FCMP_UGE:
    case llvm::CmpInst::Predicate::ICMP_SGE:
    case llvm::CmpInst::Predicate::ICMP_UGE: cmp += ">=";
                              break;
    case llvm::CmpInst::Predicate::FCMP_OLE:
    case llvm::CmpInst::Predicate::FCMP_ULE:
    case llvm::CmpInst::Predicate::ICMP_SLE:
    case llvm::CmpInst::Predicate::ICMP_ULE: cmp += "<=";
                              break;
    default:
        llvm::errs() << "Predicate not identified...\n";
        break;
    }

    return cmp;
}

std::string convertConstantFPToString(llvm::ConstantFP* cFP)
{
    std::string floatString;
    if (cFP->getType()->isDoubleTy())
    {
        floatString = std::to_string(cFP->getValueAPF().convertToDouble());
    }
    else
    {
        floatString = std::to_string(cFP->getValueAPF().convertToFloat());
    }
    return floatString;
}

std::string ifConstantToString(llvm::Value* val)
{
    std::string ret;
    if(llvm::isa<llvm::Constant>(val))
    {
        if (llvm::ConstantInt* cInt = llvm::dyn_cast<llvm::ConstantInt>(val))
        {
            ret = std::to_string(cInt->getSExtValue());
        }
        else if (llvm::ConstantFP* cFP = llvm::dyn_cast<llvm::ConstantFP>(val))
        {
            ret = convertConstantFPToString(cFP);
        }
        else
        {
            ret = val->getName().str();
        }
    }
    return ret;
}
