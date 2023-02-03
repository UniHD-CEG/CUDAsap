#ifndef NODE_H
#define NODE_H

#include "llvm/IR/BasicBlock.h"
#include "llvm/Support/raw_ostream.h"
#include <vector>
#include <bitset>
#include <string>


void findDependencies(std::string str, std::bitset<6>& dep, bool findBlockIdx);
std::bitset<6> dependsOnIdx(std::string str);


struct ConditionInfo
{
    std::string condition;
    std::bitset<6> dependencies;

    ConditionInfo() : condition(""), dependencies(0) {}
    ConditionInfo( std::string cond, std::bitset<6> dep) : condition(cond), dependencies(dep) {}
};


class Node
{
public:
    llvm::BasicBlock* block;
    unsigned index;
    std::vector<Node*> successorList;
    bool isLoopExit;
    std::vector<ConditionInfo> condInfo;

    Node(llvm::BasicBlock* bb_) : block(bb_), isLoopExit(false) {}
    void print() const;
    void getSuccessorNode(std::vector<Node>& BBInfo, llvm::BasicBlock* suc);
    void appendNewConditionInfo(std::string condString);
};

#endif // NODE_H
