#ifndef COMMON_H_
#define COMMON_H_

#include <iostream>
#include <vector>
#include <functional>
#include <numeric>
#include <boost/unordered_map.hpp>
#include <boost/unordered_set.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_int_distribution.hpp>
#include <boost/lexical_cast.hpp>

using namespace std;
using namespace boost;

enum Strand
{
	PlusStrand = 0,
	MinusStrand = 1,
};

int OtherReadEnd(int readEnd);

void ReverseComplement(string& sequence);

class RandomGenomicPositionGenerator
{
public:
	explicit RandomGenomicPositionGenerator(const vector<long>& chromosomeLengths, int seed = 2014)
	 : mChromosomeLengths(chromosomeLengths), mGenerator(seed)
	{
		mGenomeLength = accumulate(chromosomeLengths.begin(), chromosomeLengths.end(), 0L);
		mDistribution = boost::random::uniform_int_distribution<long>(1, mGenomeLength);
	}
	
	void Next(int& chrIdx, long& position)
	{
		position = mDistribution(mGenerator);
		
		chrIdx = 0;
		while (chrIdx < mChromosomeLengths.size())
		{
			if (position <= mChromosomeLengths[chrIdx])
			{
				break;
			}
			
			position -= mChromosomeLengths[chrIdx];
			
			chrIdx++;
		}
	}
	
private:
	const vector<long>& mChromosomeLengths;
	boost::random::mt19937 mGenerator;
	boost::random::uniform_int_distribution<long> mDistribution;
	long mGenomeLength;
};

void CheckFile(const ios& file, const string& filename);

template<typename TType>
TType SafeParseField(const string& field, const char* nameOfType, const char* codeFilename, int codeLine, const string& parseFilename = string(), int parseLine = 0)
{
    using boost::lexical_cast;
    using boost::bad_lexical_cast;
	
	try
	{
		return lexical_cast<TType>(field);
	}
	catch (bad_lexical_cast &)
	{
		if (parseFilename.empty())
		{
			cerr << "error interpreting '" << field << "' as " << nameOfType << endl;
		}
		else
		{
			cerr << "error interpreting '" << field << "' as " << nameOfType << " for " << parseFilename << ":" << parseLine << endl;
		}
		cerr << "parsing failed at " << codeFilename << ":" << codeLine << endl;
		exit(1);
	}
}

#define SAFEPARSE(type, field) SafeParseField<type>(field, #type, __FILE__, __LINE__)

#define SAFEPARSEFIELD(type, field, filename, line) SafeParseField<type>(field, #type, __FILE__, __LINE__, filename, line)

bool ReadTSV(istream& file, vector<string>& fields);

#endif
