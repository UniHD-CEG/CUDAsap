#ifndef KERNELINFO_H
#define KERNELINFO_H

#include "llvm/Pass.h"
#include "llvm/IR/Value.h"
#include "llvm/IR/Function.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/ScalarEvolution.h"

#include <vector>
#include <string>
#include <fstream>
#include <utility>
#include <stack>
#include "node.h"
#include "recurrence.h"
#include "traceback.h"


class KernelInfo
{
private:
    std::string name;
    std::vector<llvm::Value*> kernelArgs;
    std::vector<llvm::BasicBlock*> bbList;
    std::vector<Node> BBinfo;
    std::vector<std::string> adjMat;
    llvm::LoopInfo loopInfo;
    std::vector<LoopInfoCustom> loopList;
    std::vector<llvm::PHINode*> indvarList;
    tracebackVisitor* visitor;
public:
    KernelInfo(llvm::Function *kernel, llvm::LoopInfo &li, llvm::ScalarEvolution &SE);
    ~KernelInfo() { delete visitor; }

    void traverseCFG();
    void writeCFGToFile(bool fullConditions);
    void analyzeSwitchConditions(Node &node, llvm::SwitchInst *swInst);
    bool checkBranchForLoop(Node &node, std::string &condString);
    void writeBlockIDs(std::ofstream& block_data) const;
    std::string getName() const;
    void getExactAdjacencyMatrix();
};

#endif // KERNELINFO_H
