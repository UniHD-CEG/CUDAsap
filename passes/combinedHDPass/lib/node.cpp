#include "node.h"

using namespace llvm;


void findDependencies(std::string str, std::bitset<6>& dep, bool findBlockIdx)
{
    const std::string idx = findBlockIdx ? "bi" : "ti";
    const unsigned size = idx.size();
    const unsigned offset = findBlockIdx ? 0 : 3;

    std::size_t pos = str.find(idx);
    while (pos != std::string::npos)
    {
        unsigned next = pos+size;
        char coord = str[next];
        switch (coord) {
        case 'x':
            dep.set(0+offset);
            break;
        case 'y':
            dep.set(1+offset);
            break;
        case 'z':
            dep.set(2+offset);
            break;
        default:
            break;
        }

        pos = str.find(idx, next);
    }
}


std::bitset<6> dependsOnIdx(std::string str)
{
    std::bitset<6> dependencies;
    findDependencies(str, dependencies, true);
    findDependencies(str, dependencies, false);

    return dependencies;
}


void Node::print() const
{
    errs() << block->getName() << ' '
           << index << ' '
           << isLoopExit << '\n';
    errs() << "    ";
    for (auto suc : successorList)
    {
        errs() << suc->index << ' ';
    }
    errs() << '\n';
}


void Node::getSuccessorNode(std::vector<Node>& BBInfo, BasicBlock* suc)
{
    for (auto& x : BBInfo)
    {
        if (x.block == suc)
        {
            successorList.push_back(&x);
        }
    }
}


void Node::appendNewConditionInfo(std::string condString)
{
    std::bitset<6> dependencies = dependsOnIdx(condString);
    ConditionInfo newCond (condString, dependencies);
    condInfo.push_back(newCond);
}
