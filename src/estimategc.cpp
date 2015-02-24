#include "DebugCheck.h"
#include "Sequences.h"
#include "external/bamtools/src/api/BamReader.h"

#include <fstream>
#include <iostream>
#include <string>
#include <map>
#include <set>
#include <tclap/CmdLine.h>
#include <boost/algorithm/string.hpp>
#include <boost/unordered_map.hpp>
#include <boost/math/distributions.hpp>
#include <boost/math/distributions/normal.hpp>

using namespace boost;
using namespace std;

using namespace BamTools;


class LoessCurve
{
public:
	LoessCurve() {}
	
	void Read(const string& curveFilename)
	{
		ifstream curveFile(curveFilename.c_str());
		CheckFile(curveFile, curveFilename);
		
		vector<string> fields;
		int line = 1;
		while (ReadTSV(curveFile, fields))
		{
			if (fields.size() < 1)
			{
				cerr << "Error: line " << line << " has too few fields" << endl;
				exit(1);
			}
			
			mCurveData.push_back(SAFEPARSE(double, fields[0]));
			
			line++;
		}
	}
	
	double Predict(double x)
	{
		int idx = max(0, min((int)mCurveData.size() - 1, (int)(x * (double)(mCurveData.size() - 1))));
		
		return max(mCurveData[idx], 0.0);
	}
	
private:
	vector<double> mCurveData;
};

struct RegionCounts
{
	string id;
	string chromosome[2];
	int strand[2];
	int position[2];
	int count;
};

void BuildIntervalSeqData(const RegionCounts& regionCounts, const Sequences& seqData, int fragmentLength, vector<uint8_t>& regionSeqData)
{
	int start = regionCounts.position[0] + 1;
	int end = regionCounts.position[1] - 1;
	
	if (start > end)
	{
		return;
	}
	
	seqData.GetWithDefault(regionCounts.chromosome[0], start, end, 0, regionSeqData);
}

void BuildReferenceSeqData(const RegionCounts& regionCounts, const Sequences& seqData, int fragmentLength, vector<uint8_t>& regionSeqData)
{
	int adjacentLength = max(0, 2*fragmentLength - (regionCounts.position[1] - regionCounts.position[0])) / 2;
	int start = regionCounts.position[0] - adjacentLength + 1;
	int end = regionCounts.position[1] + adjacentLength - 1;
	
	if (start > end)
	{
		return;
	}
	
	seqData.GetWithDefault(regionCounts.chromosome[0], start, end, 0, regionSeqData);
}

void BuildVariantSeqData(const RegionCounts& regionCounts, const Sequences& seqData, int fragmentLength, vector<uint8_t>& regionSeqData)
{
	for (int variantEnd = 0; variantEnd <= 1; variantEnd++)
	{
		int start = regionCounts.position[variantEnd] - fragmentLength + 1;
		int end = regionCounts.position[variantEnd];
		
		if (regionCounts.position[variantEnd] == MinusStrand)
		{
			start = regionCounts.position[variantEnd];
			end = regionCounts.position[variantEnd] + fragmentLength - 1;
		}
		
		if (start > end)
		{
			continue;
		}
		
		vector<uint8_t> variantEndMappability;
		seqData.GetWithDefault(regionCounts.chromosome[0], start, end, 0, variantEndMappability);
		
		if ((variantEnd == 0 && regionCounts.position[variantEnd] == MinusStrand) ||
			(variantEnd == 1 && regionCounts.position[variantEnd] == PlusStrand))
		{
			reverse(variantEndMappability.begin(), variantEndMappability.end());
		}
		
		regionSeqData.insert(regionSeqData.begin(), variantEndMappability.begin(), variantEndMappability.end());
	}
}

double GCIndicator(char nt)
{
	return (nt == 'G' || nt == 'g' || nt == 'C' || nt == 'c') ? 1.0 : 0.0;
}

struct PairedMappability
{
	PairedMappability(double fragmentMean, double fragmentStdDev, int alignLength)
	{
		int minFragmentLength = (int)(fragmentMean - 3.0 * fragmentStdDev);
		int maxFragmentLength = (int)(fragmentMean + 3.0 * fragmentStdDev);
		
		offset = minFragmentLength - alignLength;
		
		for (int fragmentLength = minFragmentLength; fragmentLength <= maxFragmentLength; fragmentLength++)
		{
			weights.push_back(math::pdf(math::normal(fragmentMean, fragmentStdDev), fragmentLength));
		}
	}
	
	double Calculate(const vector<uint8_t>& mappability, int position)
	{
		if (mappability[position] == 0)
		{
			return 0.0;
		}
		
		double pairedMappability = 0.0;
		for (int idx = 0; idx < min(weights.size(), mappability.size() - position - offset); idx++)
		{
			pairedMappability += (mappability[position + idx + offset] == 0) ? 0.0 : weights[idx];
		}
		
		return pairedMappability;
	}
	
	int offset;
	vector<double> weights;
};

int main(int argc, char* argv[])
{
	string gcCurveFilename;
	int alignLength;
	double fragmentMean;
	double fragmentStdDev;
	int positionOffset;
	string regionCountsFilename;
	string genomeFastaFilename;
	string mappabilityFilename;
	bool intervalType;
	bool referenceType;
	bool variantType;
	
	try
	{
		TCLAP::CmdLine cmd("Estimate Mean Given Mappability");
		TCLAP::ValueArg<string> gcCurveFilenameArg("l", "loessgc","GC Loess Curve Filename",true,"","string",cmd);
		TCLAP::ValueArg<int> alignLengthArg("a","alignlen","Aligned Length for Mappability",true,0,"integer",cmd);
		TCLAP::ValueArg<double> fragmentMeanArg("u","ufragment","Fragment Length Mean",true,0,"float",cmd);
		TCLAP::ValueArg<double> fragmentStdDevArg("s","sfragment","Fragment Length StdDev",true,0,"float",cmd);
		TCLAP::ValueArg<int> positionOffsetArg("o","offset","Position Offset for GC",true,0,"integer",cmd);
		TCLAP::ValueArg<string> regionCountsFilenameArg("c","counts","Region Counts Filename",true,"","string",cmd);
		TCLAP::ValueArg<string> genomeFastaFilenameArg("g","genome","Genome Fasta",true,"","string",cmd);
		TCLAP::ValueArg<string> mappabilityFilenameArg("m","map","Mappability BedGraph Filename",true,"","string",cmd);
		
		TCLAP::SwitchArg intervalTypeArg("i","interval","Interval Type Regions");
		TCLAP::SwitchArg referenceTypeArg("r","reference","Reference Type Regions");
		TCLAP::SwitchArg variantTypeArg("v","variant","Variant Type Regions");
		
		vector<TCLAP::Arg*> alignmentArgs;
		alignmentArgs.push_back(&intervalTypeArg);
		alignmentArgs.push_back(&referenceTypeArg);
		alignmentArgs.push_back(&variantTypeArg);
		cmd.xorAdd(alignmentArgs);
		
		cmd.parse(argc,argv);
		
		gcCurveFilename = gcCurveFilenameArg.getValue();
		alignLength = alignLengthArg.getValue();
		fragmentMean = fragmentMeanArg.getValue();
		fragmentStdDev = fragmentStdDevArg.getValue();
		positionOffset = positionOffsetArg.getValue();
		regionCountsFilename = regionCountsFilenameArg.getValue();
		genomeFastaFilename = genomeFastaFilenameArg.getValue();
		mappabilityFilename = mappabilityFilenameArg.getValue();
		intervalType = intervalTypeArg.getValue();
		referenceType = referenceTypeArg.getValue();
		variantType = variantTypeArg.getValue();
	}
	catch (TCLAP::ArgException &e)
	{
		cerr << "error: " << e.error() << " for arg " << e.argId() << endl;
		exit(1);
	}
	
	int fragmentLength = (int)fragmentMean;
	
	cerr << "Reading GC Loess Curve" << endl;
	
	LoessCurve gcLoessCurve;
	gcLoessCurve.Read(gcCurveFilename);
	
	cerr << "Reading Genome" << endl;
	
	Sequences genome(1000);
	genome.Read(genomeFastaFilename);
	
	cerr << "Reading Mappability" << endl;
	
	Sequences mappability;
	mappability.ReadMappabilityBedGraph(mappabilityFilename);
	
	ifstream regionCountsFile(regionCountsFilename.c_str());
	CheckFile(regionCountsFile, regionCountsFilename);
	
	PairedMappability pairedMappability(fragmentMean, fragmentStdDev, alignLength);
	
	vector<string> fields;
	int line = 1;
	while (ReadTSV(regionCountsFile, fields))
	{
		if (fields.size() < 8)
		{
			cerr << "Error: line " << line << " has too few fields" << endl;
			exit(1);
		}
		
		RegionCounts regionCounts;
		
		regionCounts.id = fields[0];
		
		int fieldIdx = 1;
		for (int regionEnd = 0; regionEnd <= 1; regionEnd++)
		{
			regionCounts.chromosome[regionEnd] = fields[fieldIdx++];
			regionCounts.position[regionEnd] = SAFEPARSE(int, fields[fieldIdx++]);
			regionCounts.strand[regionEnd] = (fields[fieldIdx++] == "+") ? PlusStrand : MinusStrand;
		}
		regionCounts.count = SAFEPARSE(int, fields[7]);
		
		vector<uint8_t> regionSequence;
		vector<uint8_t> regionMappability;
		if (intervalType)
		{
			BuildIntervalSeqData(regionCounts, genome, fragmentLength, regionSequence);
			BuildIntervalSeqData(regionCounts, mappability, fragmentLength, regionMappability);
		}
		else if (referenceType)
		{
			BuildReferenceSeqData(regionCounts, genome, fragmentLength, regionSequence);
			BuildReferenceSeqData(regionCounts, mappability, fragmentLength, regionMappability);
		}
		else if (variantType)
		{
			BuildVariantSeqData(regionCounts, genome, fragmentLength, regionSequence);
			BuildVariantSeqData(regionCounts, mappability, fragmentLength, regionMappability);
		}
		
		vector<int> gcIndicator;
		for (int position = 0; position < regionSequence.size(); position++)
		{
			gcIndicator.push_back(GCIndicator(regionSequence[position]));
		}
		
		vector<int> gcPartialSum(gcIndicator.size());
		partial_sum(gcIndicator.begin(), gcIndicator.end(), gcPartialSum.begin());
		
		double gcRegionSize = (double)(fragmentLength - 2 * positionOffset);
		
		double adjustedLength = 0.0;
		for (int position = 0; position < (int)regionMappability.size() - fragmentLength; position++)
		{
			int gcSum = gcPartialSum[position + fragmentLength - positionOffset - 1];
			if (positionOffset > 0)
			{
				gcSum -= gcPartialSum[position + positionOffset - 1];
			}
			
			double gcPrediction = gcLoessCurve.Predict((double)gcSum / gcRegionSize);
			double mappabilityPrediction = pairedMappability.Calculate(regionMappability, position);
			
			adjustedLength += gcPrediction * mappabilityPrediction;
		}
		
		for (int idx = 0; idx < fields.size(); idx++)
		{
			cout << fields[idx] << "\t";
		}
		cout << adjustedLength << endl;
	}
}

