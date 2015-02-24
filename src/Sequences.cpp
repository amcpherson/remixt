#include "Sequences.h"
#include "DebugCheck.h"

#include <fstream>
#include <algorithm>
#include <boost/unordered_map.hpp>
#include <boost/algorithm/string.hpp>

using namespace std;
using namespace boost;

void Sequences::Read(const string& fastaFilename)
{
	ifstream fastaFile(fastaFilename.c_str());
	CheckFile(fastaFile, fastaFilename);
	
	string id;
	string sequence;
	
	string line;
	while (getline(fastaFile, line))
	{
		if (line.size() == 0)
		{
			continue;
		}
		
		if (line[0] == '>')
		{
			if (!id.empty())
			{
				sequence.append(string(mPadding,'N'));
				sequence.reserve(0);
				mSequences[id] = sequence;
				mNames.push_back(id);
			}
			
			id = line.substr(1);
			
			vector<string> idFields;
			split(idFields, id, is_any_of(" "));
			id = idFields[0];
			
			sequence = string(mPadding,'N');
		}
		else
		{
			sequence.append(line);
		}
	}
	
	if (!id.empty())
	{
		sequence.append(string(mPadding,'N'));
		mSequences[id] = sequence;
		mNames.push_back(id);
	}
	
	mConcatSize = 0;
	for (unordered_map<string,string>::const_iterator seqIter = mSequences.begin(); seqIter != mSequences.end(); seqIter++)
	{
		mConcatSize += seqIter->second.size();
	}
}

void Sequences::ReadMappabilityBedGraph(const string& bedGraphFilename)
{
	ifstream bedGraphFile(bedGraphFilename.c_str());
	CheckFile(bedGraphFile, bedGraphFilename);
	
	vector<string> fields;
	int line = 1;
	while (ReadTSV(bedGraphFile, fields))
	{
		if (fields.size() < 4)
		{
			ReportFailure("Error: line " << line << " has too few fields");
		}
		
		const string& chromosome = fields[0];
		int start = SAFEPARSEFIELD(int, fields[1], bedGraphFilename, line);
		int end = SAFEPARSEFIELD(int, fields[2], bedGraphFilename, line);
		int value = SAFEPARSEFIELD(int, fields[3], bedGraphFilename, line);
		
		uint8_t valueApprox = (uint8_t)min(value, (int)numeric_limits<uint8_t>::max());
		
		if (mSequences[chromosome].size() < end + mPadding)
		{
			mSequences[chromosome].resize(end + mPadding, 0);
		}
		
		for (int pos = start + mPadding; pos < end + mPadding; pos++)
		{
			mSequences[chromosome][pos] = valueApprox;
		}
		
		line++;
	}
	
	mConcatSize = 0;
	for (unordered_map<string,string>::iterator seqIter = mSequences.begin(); seqIter != mSequences.end(); seqIter++)
	{
		mNames.push_back(seqIter->first);
		seqIter->second.resize(seqIter->second.size() + mPadding, 0);
		mConcatSize += seqIter->second.size();
	}
}

const string& Sequences::Get(const string& id) const
{
	if (mSequences.find(id) == mSequences.end())
	{
		ReportFailure("Error: Unable to find sequence " << id);
	}
	
	return mSequences.find(id)->second;
}

void Sequences::Get(const string& id, int start, int end, string& sequence) const
{
	if (mSequences.find(id) == mSequences.end())
	{
		ReportFailure("Error: Unable to find sequence " << id);
	}
	
	const string& fullSequence = mSequences.find(id)->second;
	int length = end - start + 1;
	
	sequence = fullSequence.substr(start - 1 + mPadding, length);
}

const char* Sequences::Get(const string& id, int pos) const
{
	if (mSequences.find(id) == mSequences.end())
	{
		ReportFailure("Error: Unable to find sequence " << id);
	}
	
	const string& fullSequence = mSequences.find(id)->second;
	
	return fullSequence.c_str() + pos - 1 + mPadding;
}

void Sequences::Get(const string& id, int strand, int& start, int& length, string& sequence) const
{
	if (mSequences.find(id) == mSequences.end())
	{
		ReportFailure("Error: Unable to find sequence " << id);
	}
	
	const string& fullSequence = mSequences.find(id)->second;
	
	int fullSequenceLength = fullSequence.length() - 2 * mPadding;
	
	int end = start + length - 1;
	
	start = max(1, start);
	end = min(fullSequenceLength, end);
	length = max(0, end - start + 1);
	
	sequence = fullSequence.substr(start - 1 + mPadding, length);
	
	if (strand == MinusStrand)
	{
		ReverseComplement(sequence);
	}
}

void Sequences::Get(const string& id, int start, int end, vector<uint8_t>& sequence) const
{
	if (mSequences.find(id) == mSequences.end())
	{
		ReportFailure("Error: Unable to find sequence " << id);
	}
	
	const string& fullSequence = mSequences.find(id)->second;
	
	sequence.resize(end - start + 1);
	
	start = start - 1 + mPadding;
	end = end - 1 + mPadding + 1;
	
	sequence = vector<uint8_t>(fullSequence.c_str() + start, fullSequence.c_str() + end);
}

void Sequences::GetWithDefault(const string& id, int start, int end, uint8_t dval, vector<uint8_t>& sequence) const
{
	if (mSequences.find(id) == mSequences.end())
	{
		ReportFailure("Error: Unable to find sequence " << id);
	}
	
	const string& fullSequence = mSequences.find(id)->second;
	
	sequence.clear();
	for (int idx = start - 1 + mPadding; idx <= end - 1 + mPadding; idx++)
	{
		if (idx >= 0 && idx < fullSequence.size())
		{
			sequence.push_back(fullSequence[idx]);
		}
		else
		{
			sequence.push_back(dval);
		}
	}
}

const vector<string>& Sequences::GetNames() const
{
	return mNames;
}

