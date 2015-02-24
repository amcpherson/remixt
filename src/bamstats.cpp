#include "Common.h"
#include "DebugCheck.h"
#include "external/bamtools/src/api/BamReader.h"

#include <fstream>
#include <iostream>
#include <string>
#include <map>
#include <set>
#include <tclap/CmdLine.h>
#include <boost/unordered_set.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/accumulators/accumulators.hpp>
#include <boost/accumulators/statistics/stats.hpp>
#include <boost/accumulators/statistics/min.hpp>
#include <boost/accumulators/statistics/max.hpp>
#include <boost/accumulators/statistics/mean.hpp>
#include <boost/accumulators/statistics/variance.hpp>

using namespace boost;
using namespace std;

using namespace BamTools;


int main(int argc, char* argv[])
{
	string bamFilename;
	int maxFragmentLength;
	string statsFilename;
	
	try
	{
		TCLAP::CmdLine cmd("Bam Stats");
		TCLAP::ValueArg<string> bamFilenameArg("b","bam","Bam Filename",true,"","string",cmd);
		TCLAP::ValueArg<int> maxFragmentLengthArg("f","flen","Maximum Fragment Length",true,0,"integer",cmd);
		TCLAP::ValueArg<string> statsFilenameArg("s","stats","Concordant Stats Filename",true,"","string",cmd);
		cmd.parse(argc,argv);
		
		bamFilename = bamFilenameArg.getValue();
		maxFragmentLength = maxFragmentLengthArg.getValue();
		statsFilename = statsFilenameArg.getValue();
	}
	catch (TCLAP::ArgException &e)
	{
		cerr << "error: " << e.error() << " for arg " << e.argId() << endl;
		exit(1);
	}
	
	int readCount = 0;
	unordered_set<int> readLengths;
	accumulators::accumulator_set<double, accumulators::stats<accumulators::tag::count, accumulators::tag::mean, accumulators::tag::variance > > fragmentLengthAcc;
	
	BamReader bamReader;
	if (!bamReader.Open(bamFilename))
	{
		cerr << "Error: Unable to open bam file " << bamFilename << endl;
		exit(1);
	}
	
	BamAlignment alignment;
	while (bamReader.GetNextAlignmentCore(alignment))
	{
		if (alignment.IsFirstMate())
		{
			readCount++;
		}
		
		if (alignment.IsProperPair() && alignment.IsFirstMate() && abs(alignment.InsertSize) <= maxFragmentLength)
		{
			fragmentLengthAcc((double)(abs(alignment.InsertSize)));
		}
		
		readLengths.insert(alignment.Length);
	}
	
	if (readCount == 0)
	{
		cerr << "Error: No reads" << endl;
		exit(1);
	}
	
	int concordantCount = accumulators::count(fragmentLengthAcc);
	
	if (concordantCount == 0)
	{
		cerr << "Error: No concordant reads" << endl;
		exit(1);
	}
	
	ofstream statsFile(statsFilename.c_str());
	
	CheckFile(statsFile, statsFilename);
	
	double fragmentMean = accumulators::mean(fragmentLengthAcc);
	double fragmentVariance = accumulators::variance(fragmentLengthAcc);
	double fragmentStdDev = pow(fragmentVariance, 0.5);
	int readLengthMin = *(min_element(readLengths.begin(), readLengths.end()));
	int readLengthMax = *(max_element(readLengths.begin(), readLengths.end()));
	stringstream readLengthListStream;
	copy(readLengths.begin(), readLengths.end(), ostream_iterator<int>(readLengthListStream, ","));
	string readLengthList = readLengthListStream.str().substr(0, readLengthListStream.str().size() - 1);
	
	statsFile << "read_count\t";
	statsFile << "concordant_count\t";
	statsFile << "fragment_mean\t";
	statsFile << "fragment_stddev\t";
	statsFile << "read_length_min\t";
	statsFile << "read_length_max\t";
	statsFile << "read_length_list" << endl;
	
	statsFile << readCount << "\t";
	statsFile << concordantCount << "\t";
	statsFile << fragmentMean << "\t";
	statsFile << fragmentStdDev << "\t";
	statsFile << readLengthMin << "\t";
	statsFile << readLengthMax << "\t";
	statsFile << readLengthList << endl;
}

