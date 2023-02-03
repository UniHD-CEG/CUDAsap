#include "kernelinfo.h"

using namespace llvm;

void initializeNodeVector(Function* kernel,
                          std::vector<Node>& BBinfo,
                          std::vector<BasicBlock*>& bbList)
{
    unsigned bbCounter = 0;
    for (auto& bb : kernel->getBasicBlockList())
    {
        Node bbNode(&bb);
        bbNode.index = bbCounter++;
        bbList[bbNode.index] = &bb;
        BBinfo.push_back(bbNode);
    }
}


// TODO: Make this function throw an exception upon failure!
void findSubLoop(Loop* l, std::vector<LoopInfoCustom>& loopList,
                 const std::vector<Value*> kernelArgs, ScalarEvolution &SE)
{
    std::vector<PHINode*> indvars = getAllAuxiliaryInductionVariables(l);
    std::vector<IndvarInfo> ivInfo (indvars.size());
    unsigned mainIndvarIndex = 0;
    for (unsigned i=0; i<indvars.size(); ++i)
    {
        std::shared_ptr<ExtendedRecurrence> rec = getRecurrenceForPHI(indvars[i], l);
        IndvarInfo iv(indvars[i], rec);
        ivInfo[i] = iv;
        if (rec->finalVal != nullptr)
        {
            mainIndvarIndex = i;
        }
    }

    LoopInfoCustom lic (l, ivInfo, mainIndvarIndex);
    loopList.push_back(lic);
    std::vector<Loop*> sub = l->getSubLoops();
    if (sub.empty())
    {
        return;
    }
    else
    {
        for (auto sl : sub)
        {
            findSubLoop(sl, loopList, kernelArgs, SE);
        }
    }
}


std::string truncateIndvarName(const IndvarInfo ivInfo)
{
    PHINode* indvar = ivInfo.indvar;
    if (indvar == nullptr)
        errs() << "indvar is null\n";
    std::string ivName = indvar->getName().str();
    ivName.erase (std::remove(ivName.begin(), ivName.end(), '.'), ivName.end());
    ivName.erase (std::remove(ivName.begin(), ivName.end(), '-'), ivName.end());

    return ivName;
}


KernelInfo::KernelInfo(Function *kernel, LoopInfo &li, ScalarEvolution &SE)
{
    name = kernel->getName();
    loopInfo = std::move(li);

    kernelArgs.reserve(kernel->arg_size());
    for (auto& arg : kernel->args())
    {
        kernelArgs.push_back(&arg);
    }

    std::size_t numBlocks = kernel->size();
    BBinfo.reserve(numBlocks);

    // To allow for a more thorough traceback analysis of PHINodes, we need
    // a list (or rather a std::vector) of BasicBlocks as well as one of the
    // indvars.
    std::vector<BasicBlock*> bbListTemp (numBlocks);

    // fill vector of Nodes
    initializeNodeVector(kernel, BBinfo, bbListTemp);
    bbList = bbListTemp;

    // find all loops and subloops
    // TODO: place try ... catch here!
    for (auto& loop : loopInfo)
    {
        findSubLoop(loop, loopList, kernelArgs, SE);
    }

    for (auto& lic : loopList)
    {
        for (auto& ivInfo : lic.indvars)
        {
            indvarList.push_back(ivInfo.indvar);
        }
    }

    visitor = new tracebackVisitor (kernelArgs, bbList, indvarList);
}


void KernelInfo::writeCFGToFile(bool fullConditions)
{
    std::string filesuffix = fullConditions ? ".out" : "_cfg.out";
    std::string filename = name + filesuffix;

    std::ofstream cfg_file(filename);
    assert(cfg_file.is_open() && "Cannot open file to store CFG.\n");
    for (std::size_t i=0; i<kernelArgs.size(); ++i)
    {
        if (kernelArgs[i]->getType()->isPointerTy())
            continue;

        std::string argName = kernelArgs[i]->getName().str();
        cfg_file << '[' << i << ']' << argName << ';';
    }
    cfg_file << '\n';

    // write all indvars and their recurrences
    for (unsigned loopID = 0; loopID<loopList.size(); ++loopID)
    {
        for (const auto& ivInfo : loopList[loopID].indvars)
        {
            auto rec = ivInfo.rec;
            if (rec.get() == nullptr)
                errs() << "rec is null\n";

            std::string ivName = truncateIndvarName(ivInfo);

            cfg_file << '[' << ivName << "," << loopID << ']'
                     << rec->getAsString(*visitor)
                     << ';';
        }
    }
    cfg_file << '\n';

    if (fullConditions)
    {
        const std::size_t numBlocks = BBinfo.size();
        for (unsigned i=0; i<adjMat.size(); ++i)
        {
            if (adjMat[i].empty())
                continue;

            unsigned row = i / numBlocks;
            unsigned col = i % numBlocks;
            cfg_file << row << ";" << col << ";" << adjMat[i] << '\n';
        }
    }
    else
    {
        for (auto node : BBinfo)
        {
            if (node.condInfo.empty())
            {
                continue;
            }

            unsigned i=0;
            for (auto suc : node.successorList)
            {
                std::string condition = node.condInfo[i].condition;
                if (condition.empty())
                {
                    condition = "1";
                }
                cfg_file << suc->index << ';'
                         << node.index << ';'
                         << condition << '\n';
                ++i;
            }
        }  // loop over BBinfo
    }
    cfg_file.close();
}


bool KernelInfo::checkBranchForLoop(Node &node, std::string &condString)
{
    BasicBlock* bb = node.block;
    unsigned loopDepth = loopInfo.getLoopDepth(bb);

    // Check if BB belongs to a loop.
    bool isLoopCond = false;
    if (loopDepth > 0)
    {
        for (auto lic : loopList)
        {
            Loop* sl = lic.loop;
            if (sl->contains(bb))
            {
                isLoopCond = sl->isLoopExiting(bb);

                // Check for exit block.
                // This is necessary for fillTransitionMatrix to work.
                for (auto& succ : node.successorList)
                {
                    succ->isLoopExit = !sl->contains(succ->block);
                }

                if (isLoopCond)
                {
                    auto rec = lic.getMainIndvar().rec;
                    condString = rec->getAsString(*visitor);
                    if (node.successorList[0]->isLoopExit)
                        std::swap(node.successorList[0], node.successorList[1]);
                }
            }
        }
    } // endif: BB is in loop
    return isLoopCond;
}


void KernelInfo::analyzeSwitchConditions(Node &node, SwitchInst *swInst)
{
    Value* cond = swInst->getCondition();
    BasicBlock* defaultDest = swInst->getDefaultDest();
    std::string negCond;
    unsigned numSucc = swInst->getNumSuccessors();

    node.condInfo.reserve(numSucc);
    ConditionInfo newCond;
    node.condInfo.push_back(newCond);
    node.getSuccessorNode(BBinfo, defaultDest);
    for (auto termCase : swInst->cases())
    {
        ConstantInt* val = termCase.getCaseValue();
        BasicBlock* succ = termCase.getCaseSuccessor();
        CmpInst* fullCond = CmpInst::Create(CmpInst::OtherOps::ICmp,
                                            CmpInst::Predicate::ICMP_EQ,
                                            cond,
                                            val);

        std::string condString = visitor->tracebackValue(fullCond);
        node.appendNewConditionInfo(condString);

        node.getSuccessorNode(BBinfo, succ);
        negCond += "(" + condString + " or ";
    }
    std::string parens (numSucc-1, ')');
    // Replace trailing string " or " with closing parentheses.
    negCond.replace(negCond.end()-4, negCond.end(), parens);
    negCond = '!' + negCond;
    std::bitset<6> dependencies = dependsOnIdx(negCond);
    node.condInfo[0].condition = negCond;
    node.condInfo[0].dependencies = dependencies;
}


// TODO: split this into several smaller functions
void KernelInfo::traverseCFG()
{
    for (auto& node : BBinfo)
    {
        BasicBlock* bb = node.block;
        Instruction* term = bb->getTerminator();

        if (BranchInst* branch = dyn_cast<BranchInst>(term))
        {
            for (unsigned i=0; i<branch->getNumSuccessors(); ++i)
            {
                BasicBlock* succ = branch->getSuccessor(i);
                node.getSuccessorNode(BBinfo, succ);
            }

            if (branch->isUnconditional())
            {
                ConditionInfo emptyCond;
                node.condInfo.push_back(emptyCond);
                continue;
            }

            Value* cond = branch->getCondition();
            std::string condString;

            bool isLoopCond = checkBranchForLoop(node, condString);

            if (!isLoopCond)
            {
                condString = visitor->tracebackValue(cond);
            }

            std::string negatedCondition = "!" + condString;
            node.appendNewConditionInfo(condString);
            node.appendNewConditionInfo(negatedCondition);
        }
        else if (SwitchInst* swInst = dyn_cast<SwitchInst>(term))
        {
            analyzeSwitchConditions(node, swInst);
        }

    }  // end traverse of CFG
}


void KernelInfo::writeBlockIDs(std::ofstream &block_data) const
{
    block_data << name << '\n';
    for (auto bb : BBinfo)
    {
        std::string bbName = bb.block->getName().str();
        block_data << bb.index << ' ' << bbName << '\n';
    }
    block_data << '\n';
}


std::string KernelInfo::getName() const
{
    return name;
}


void KernelInfo::getExactAdjacencyMatrix()
{
    errs() << "get matrix for kernel " << name << '\n';
    std::size_t numBlocks = BBinfo.size();
    std::vector<std::string> adjMatTemp (numBlocks*numBlocks);

    // iterative tree traversal with stack
    std::stack<const Node*> nodeStack;
    std::stack<std::vector<ConditionInfo>> condStack;

    const Node* root = &(BBinfo[0]);
    nodeStack.push(root);

    while (!nodeStack.empty())
    {
        auto topNode = nodeStack.top();
        nodeStack.pop();
        unsigned from = topNode->index;

        // We need the last inserted condition
        std::vector<ConditionInfo> lastCond;
        if (!condStack.empty())
        {
            lastCond = condStack.top();
            condStack.pop();
        }

        unsigned succIdx=0;
        for (auto succ : topNode->successorList)
        {
            unsigned to = succ->index;
            unsigned matInd = to*numBlocks + from;
            ConditionInfo topNodeCI = topNode->condInfo[succIdx];
            std::string cond;
            std::string brCond = topNodeCI.condition;
            bool notLoop = false;
            if (!brCond.empty())
            {
                // check for loop
                // We assume the loop to be index independent (->future work)
                if (brCond[0] != '{' && brCond[1] != '{')  // not a loop
                {
                    notLoop = true;

                    // check cross dependencies
                    for (auto& prev : lastCond)
                    {
                        if ((prev.dependencies & topNodeCI.dependencies) != 0)
                        {
                            cond += prev.condition + " && ";
                        }
                    }
                }

                cond += brCond;

                // insert cond only once
                if (adjMatTemp[matInd].find(cond) == std::string::npos)
                {
                    std::string connector = adjMatTemp[matInd].empty() ? "" : " || ";
                    adjMatTemp[matInd] += connector + cond;
                }
            }
            else // non-conditional transition
            {
                notLoop = true;
                if (adjMatTemp[matInd].empty())
                {
                    adjMatTemp[matInd] += "1";
                }
            }

            // be cautious with loops
            if (notLoop || succ->isLoopExit)
            {
                nodeStack.push(succ);
                if (notLoop)
                {
                    // improve!
                    std::vector<ConditionInfo> temp = lastCond;
                    temp.push_back(topNodeCI);
                    condStack.push(temp);
                }
            }

            ++succIdx;
        }

    }

    adjMat = adjMatTemp;
}
