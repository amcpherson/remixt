#ifndef SEQUENCES_H_
#define SEQUENCES_H_

#include "Common.h"

#include <map>
#include <string>
#include <iostream>

using namespace std;
using namespace boost;

class Sequences
{
public:
	explicit Sequences(int padding = 0) : mPadding(padding), mConcatSize(0) {}
	void Read(const string& fastaFilename);
	void ReadMappabilityBedGraph(const string& bedGraphFilename);
	const string& Get(const string& id) const;
	void Get(const string& id, int start, int end, string& sequence) const;
	void Get(const string& id, int start, int end, vector<uint8_t>& sequence) const;
	void GetWithDefault(const string& id, int start, int end, uint8_t dval, vector<uint8_t>& sequence) const;
	const char* Get(const string& id, int pos) const;
	void Get(const string& id, int strand, int& start, int& length, string& sequence) const;
	const vector<string>& GetNames() const;
	
private:
	int mPadding;
	vector<string> mNames;
	unordered_map<string,string> mSequences;
	int mConcatSize;
};

#endif
