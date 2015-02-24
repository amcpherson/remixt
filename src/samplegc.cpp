#include "DebugCheck.h"
#include "Sequences.h"
#include "external/bamtools/src/api/BamReader.h"

#include <fstream>
#include <iostream>
#include <string>
#include <map>
#include <set>
#include <tclap/CmdLine.h>
#include <boost/lexical_cast.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/unordered_map.hpp>

using namespace boost;
using namespace std;
using namespace BamTools;


bool IsMappable(const Sequences& mappability, const string& chromosome, long start)
{
	vector<uint8_t> fragmentMappability;
	mappability.Get(chromosome, start, start, fragmentMappability);
	return fragmentMappability[0] != 0;
}

double CalculateGC(const Sequences& genomeSequences, const string& chromosome, long start, int fragmentLength, int positionOffset)
{
	const char* seqPtr = genomeSequences.Get(chromosome, start);
	
	double gcCount = 0.0;
	double ntCount = 0.0;
	for (int idx = positionOffset; idx < fragmentLength - positionOffset; idx++)
	{
		gcCount += (seqPtr[idx] == 'G' || seqPtr[idx] == 'C') ? 1.0 : 0.0;
		ntCount += 1.0;
	}
	
	return gcCount / ntCount;
}

int main(int argc, char* argv[])
{
	int numSamples;
	int fragmentLength;
	int positionOffset;
	string genomeFastaFilename;
	string mappabilityFilename;
	string bamFilename;
	
	try
	{
		TCLAP::CmdLine cmd("Sample Positions and Calculate GC");
		TCLAP::ValueArg<int> numSamplesArg("n","num","Number of Samples",true,0,"integer",cmd);
		TCLAP::ValueArg<int> fragmentLengthArg("f","frlen","Mean Fragment Length",true,0,"integer",cmd);
		TCLAP::ValueArg<int> positionOffsetArg("o","offset","Position Offset for GC",true,0,"integer",cmd);
		TCLAP::ValueArg<string> genomeFastaFilenameArg("g","genome","Genome Fasta",true,"","string",cmd);
		TCLAP::ValueArg<string> mappabilityFilenameArg("m","map","Mappability BedGraph Filename",true,"","string",cmd);
		TCLAP::ValueArg<string> bamFilenameArg("b","bam","Bam Filename",true,"","string",cmd);
		cmd.parse(argc,argv);
		
		numSamples = numSamplesArg.getValue();
		fragmentLength = fragmentLengthArg.getValue();
		positionOffset = positionOffsetArg.getValue();
		genomeFastaFilename = genomeFastaFilenameArg.getValue();
		mappabilityFilename = mappabilityFilenameArg.getValue();
		bamFilename = bamFilenameArg.getValue();
	}
	catch (TCLAP::ArgException &e)
	{
		cerr << "error: " << e.error() << " for arg " << e.argId() << endl;
		exit(1);
	}
	
	BamReader bamReader;
	if (!bamReader.Open(bamFilename))
	{
		cerr << "Error: Unable to open bam file " << bamFilename << endl;
		exit(1);
	}
	
	vector<long> chromosomeLengths;
	for (int chrIdx = 0; chrIdx < bamReader.GetReferenceCount(); chrIdx++)
	{
		chromosomeLengths.push_back(bamReader.GetReferenceData()[chrIdx].RefLength);
	}
	
	cerr << "Reading Genome" << endl;
	
	Sequences genomeSequences(1000);
	genomeSequences.Read(genomeFastaFilename);
	
	cerr << "Reading Mappability" << endl;
	
	Sequences mappability;
	mappability.ReadMappabilityBedGraph(mappabilityFilename);
	
	cerr << "Sampling Positions" << endl;
	
	RandomGenomicPositionGenerator randomPosition(chromosomeLengths);
	
	typedef pair<int,long> SamplePosition;
	typedef pair<double,int> GCCount;
	
	unordered_map<SamplePosition,GCCount> samples;
	
	while (samples.size() < numSamples)
	{
		// Sample 1 based position
		int chrIdx;
		long position;
		randomPosition.Next(chrIdx, position);
		
		const string& chromosome = bamReader.GetReferenceData()[chrIdx].RefName;
		
		if (!IsMappable(mappability, chromosome, position))
		{
			continue;
		}
		
		double gcPercent = CalculateGC(genomeSequences, chromosome, position, fragmentLength, positionOffset);
		
		samples[SamplePosition(chrIdx, position)] = GCCount(gcPercent, 0);
	}
	
	cerr << "Counting reads from bam" << endl;
	
	int rowCount = 0;
	
	BamAlignment alignment;
	while (bamReader.GetNextAlignmentCore(alignment))
	{
		if (++rowCount % 2000000 == 0)
		{
			cerr << ".";
			cerr.flush();
		}
		
		if (alignment.IsProperPair() && !alignment.IsReverseStrand() && alignment.MapQuality > 0 && alignment.InsertSize < 2*fragmentLength)
		{
			unordered_map<SamplePosition,GCCount>::iterator sampleIter = samples.find(SamplePosition(alignment.RefID, alignment.Position + 1));
			if (sampleIter != samples.end())
			{
				sampleIter->second.second++;
			}
		}
	}
	
	for (unordered_map<SamplePosition,GCCount>::const_iterator sampleIter = samples.begin(); sampleIter != samples.end(); sampleIter++)
	{
		cout << bamReader.GetReferenceData()[sampleIter->first.first].RefName << "\t";
		cout << sampleIter->first.second << "\t";
		cout << sampleIter->second.first << "\t";
		cout << sampleIter->second.second << endl;
	}
}

