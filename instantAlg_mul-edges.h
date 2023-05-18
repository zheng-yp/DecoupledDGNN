#ifndef InstantGNN_H
#define InstantGNN_H
#include<iostream>
#include <vector>
#include <iostream>
#include <algorithm>
#include <random>
#include <queue>
#include <unordered_map>
#include <math.h>
#include <cmath>
#include <limits.h>
#include <unistd.h>
#include <math.h>
#include <fstream>
#include <sstream>
#include <assert.h>
#include <thread>
#include <string>
#include <unistd.h>
#include <sys/time.h>
#include<Eigen/Dense>

#include "Graph.h"

using namespace std;
using namespace Eigen;
typedef unsigned int uint;

namespace propagation{
    class Instantgnn{
        Eigen::MatrixXd X;
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
        int NUMTHREAD=40;//Number of threads
    	uint edges, vert;
        Graph g;
        vector<vector<double>> R;
        //vector<vector<double>> P;
    	double rmax,alpha,t;
        string dataset_name;
        string updateFile;
		vector<double>rowsum_pos;
        vector<double>rowsum_neg;
        vector<int>random_w;
        vector<int>update_w;
        vector<double>Du;
        double initial_operation(string path, string dataset,uint mm,uint nn,double rmaxx,double alphaa,Eigen::Map<Eigen::MatrixXd> &feat);
        void ppr_push(int dimension, Eigen::Ref<Eigen::MatrixXd>feat, bool init,vector<queue<uint>>& candidate_sets,vector<vector<bool>>& isCandidates);
        void ppr_residue(Eigen::Ref<Eigen::MatrixXd>feats,int st,int ed, bool init,vector<queue<uint>>& candidate_sets,vector<vector<bool>>& isCandidates);
        void dynamic_operation(uint v_from, uint v_to, double rmaxx,double alphaa, Eigen::Map<Eigen::MatrixXd> &feat);
        void snapshot_operation(string updatefilename, double rmaxx,double alphaa, Eigen::Map<Eigen::MatrixXd> &feat);
        void overall_operation(double rmaxx,double alphaa, Eigen::Map<Eigen::MatrixXd> &feat);
        vector<vector<uint>> update_graph(string updatefilename, vector<uint>&affected_nodelsts);
    };
}


#endif // InstantGNN_H
