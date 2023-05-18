#ifndef GRAPH_H
#define GRAPH_H

#include <cstdlib>
#include <cstdio>
#include <iostream>
#include <fstream>
#include <time.h>
#include <vector>
using namespace std;

class Graph
{
public:
	uint n;	//number of nodes
	uint m;	//number of edges

	vector<vector<uint>> inAdj;
	vector<vector<uint>> outAdj;
	uint* indegree;
	uint* outdegree;
  vector<uint>indices;
  vector<uint>indptr;
	Graph()
	{
	}
	~Graph()
	{
	}

	void insertEdge(uint from, uint to) {
		outAdj[from].push_back(to);
		inAdj[to].push_back(from);
		outdegree[from]++;
		indegree[to]++;
	}

	void deleteEdge(uint from, uint to) {
		uint j;
		for (j=0; j < indegree[to]; j++) {
			if (inAdj[to][j] == from) {
				break;
			}
		}
		inAdj[to].erase(inAdj[to].begin()+j);
		indegree[to]--;

		for (j=0; j < outdegree[from]; j++) {
			if (outAdj[from][j] == to) {
				break;
			}
		}

		outAdj[from].erase(outAdj[from].begin() + j);
		outdegree[from]--;
	}

	int isEdgeExist(uint u, uint v) {
		for (uint j = 0; j < outdegree[u]; j++) {
			if (outAdj[u][j] == v) {
				return -1;
			}
		}
		return 1;
	}

	void inputGraph(string path, string dataset, uint nodenum, uint edgenum)
	{
    n = nodenum;
    m = edgenum;
    indices=vector<uint>(m);
    indptr=vector<uint>(n+1);
    //string dataset_el="data/"+dataset+"_adj_el.txt";
    string dataset_el=path+dataset+"_adj_el.txt";
    const char *p1=dataset_el.c_str();
    if (FILE *f1 = fopen(p1, "rb"))
    {
        size_t rtn = fread(indices.data(), sizeof indices[0], indices.size(), f1);
        if(rtn!=m)
            cout<<"Error! "<<dataset_el<<" Incorrect read!"<<endl;
        fclose(f1);
    }
    else
    {
        cout<<dataset_el<<" Not Exists."<<endl;
        exit(1);
    }
    string dataset_pl=path+dataset+"_adj_pl.txt";
    const char *p2=dataset_pl.c_str();

    if (FILE *f2 = fopen(p2, "rb"))
    {
        size_t rtn = fread(indptr.data(), sizeof indptr[0], indptr.size(), f2);
        if(rtn!=n+1)
            cout<<"Error! "<<dataset_pl<<" Incorrect read!"<<endl;
        fclose(f2);
    }
    else
    {
        cout<<dataset_pl<<" Not Exists."<<endl;
        exit(1);
    }
		indegree=new uint[n];
		outdegree=new uint[n];
        clock_t t1=clock();
		for(uint i=0;i<n;i++)
		{
			indegree[i] = indptr[i+1]-indptr[i];
            outdegree[i] = indptr[i+1]-indptr[i];
            vector<uint> templst(indices.begin() + indptr[i],indices.begin() + indptr[i+1]);
            outAdj.push_back(templst);
            inAdj.push_back(templst);
		}
		
		clock_t t2=clock();
		cout<<"m="<<m<<endl;
		cout<<"reading in graph takes "<<(t2-t1)/(1.0*CLOCKS_PER_SEC)<<" s."<<endl;
	}

	void inputGraph_fromedgelist(string path, string dataset, uint nodenum, uint edgenum)
	{
		m=edgenum;
		n=nodenum;
		string filename = path+dataset+".txt";
		ifstream infile(filename.c_str());

		indegree=new uint[n];
		outdegree=new uint[n];
		for(uint i=0;i<n;i++)
		{
			indegree[i]=0;
			outdegree[i]=0;
		}
		//read graph and get degree info
		uint from;
		uint to;
		while(infile>>from>>to)
		{
			outdegree[from]++;
			indegree[to]++;
		}

		cout<<"..."<<endl;

		for (uint i = 0; i < n; i++)
		{
			vector<uint> templst;
			inAdj.push_back(templst);
			outAdj.push_back(templst);
		}

		infile.clear();
		infile.seekg(0);

		clock_t t1=clock();

		while(infile>>from>>to)
		{
			outAdj[from].push_back(to);
			inAdj[to].push_back(from);
		}
		infile.close();
		clock_t t2=clock();
		cout<<"m="<<m<<endl;
		cout<<"reading in graph takes "<<(t2-t1)/(1.0*CLOCKS_PER_SEC)<<" s."<<endl;
	} 

	void inputWeightedGraph_fromedgelist(string path, string dataset, uint nodenum, uint edgenum)
	{
		m=edgenum;
		n=nodenum;
		string filename = path+dataset+".txt";
		//ifstream infile(filename.c_str());
		ifstream infile;
		infile.open(filename, ios::in);
		if(!infile.is_open())
		{
			cout<<"open file: " << filename <<" failed!!!"<<endl;
		}
		//
		vector<vector<vector<double>>> inWeight;  //inWeight[n][v_pos]=W_uv
		vector<vector<vector<double>>> outWeight;

		//initialize inAdj,outAdj,indegree,outdegree
		for (uint i = 0; i < n; i++)
		{
			vector<uint> templst;
			inAdj.push_back(templst);
			outAdj.push_back(templst);
			vector<vector<double>> temp;
			inWeight.push_back(temp);
			outWeight.push_back(temp);
		}
		indegree=new uint[n];//degree=outAdj[n].size()
		outdegree=new uint[n];
		for(uint i=0;i<n;i++)
		{
			indegree[i]=0;
			outdegree[i]=0;
		}

		vector<string> item;
		string temp;
		//read graph and get degree info
		uint from;
		uint to;
		// double weight;
		clock_t t1=clock();
		while(getline(infile, temp))
		{
			item.push_back(temp);
		}
		for(auto it = item.begin(); it!=item.end(); it++)
		{
			istringstream istr(*it);  //
			string str;
			int count = 0;
			vector<double> weights; //weights.size()
			while(istr>>str)
			{
				if(count == 0)
				{
					from = atoi(str.c_str());
				}
				if(count == 1)
				{
					to = atoi(str.c_str());
				}
				if(count >= 2)
				{
					double w_in = atof(str.c_str());
					weights.push_back(w_in);
				}
				count++;
			}
			// outWeight[from][to]=weights; //
			// inWeight[to][from]=weights;
			outAdj[from].push_back(to);
			inAdj[to].push_back(from);
			outWeight[from].push_back(weights);
			inWeight[to].push_back(weights);
			outdegree[from]++;
			indegree[to]++;
		}
		/*while(infile>>from>>to)
		{
			outdegree[from]++;
			indegree[to]++;
		}

		cout<<"..."<<endl;

		infile.clear();
		infile.seekg(0);

		while(infile>>from>>to)
		{
			outAdj[from].push_back(to);
			inAdj[to].push_back(from);
		}*/
		infile.close();
		clock_t t2=clock();
		cout<<"m="<<m<<endl;
		cout<<"reading in graph takes "<<(t2-t1)/(1.0*CLOCKS_PER_SEC)<<" s."<<endl;

		//test read graph
		cout<<"-------------------- test ouput -------------------"<<endl;
		int outD, inD;
		for (uint i = 0; i < n; i++)
		{
			outD = outAdj[i].size();
			cout<<"node "<<i<<"'s outdegree = "<<outD<<", and its outAdjs contains:"<<endl;
			for(int vn=0; vn<outD; vn++)
			{
				uint node_v = outAdj[i][vn];
				int dim_weight = outWeight[i][vn].size();
				cout<<vn<<"-th: "<<node_v<<", dim of weights = "<<dim_weight<<", weights: "<<endl;
				for(int ww=0; ww<dim_weight; ww++)
				{
					double weight_d = outWeight[i][vn][ww];
					cout<<weight_d<<" ;";
				}
				cout<<"==========="<<endl;
			}
			inD = inAdj[i].size();
			cout<<"node "<<i<<"'s indegree = "<<inD<<endl;
		}
	}

	int getInSize(int vert){
		return indegree[vert];
	}
	int getInVert(int vert, int pos){
		return inAdj[vert][pos];
	}
	int getOutSize(int vert){
		return outdegree[vert];
	}
	int getOutVert(int vert, int pos){
		return outAdj[vert][pos];
	}
  vector<uint> getOutAdjs(uint vert){
        return outAdj[vert];
    }

};


#endif
