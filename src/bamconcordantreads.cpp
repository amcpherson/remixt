#include "Common.h"
#include "DebugCheck.h"
#include "external/bamtools/src/api/BamReader.h"
#include "external/bamtools/src/utils/bamtools_pileup_engine.h"

#include <fstream>
#include <iostream>
#include <string>
#include <map>
#include <set>
#include <tclap/CmdLine.h>
#include <boost/algorithm/string.hpp>
#include <boost/unordered_map.hpp>
#include <boost/iostreams/filtering_stream.hpp>
#include <boost/iostreams/filter/gzip.hpp>

using namespace boost;
using namespace std;

using namespace BamTools;


inline int GetNumSoftClipped(const BamAlignment& alignment)
{
	int numSoftClipped = 0;
	for (vector<CigarOp>::const_iterator cigarOpIter = alignment.CigarData.begin(); cigarOpIter != alignment.CigarData.end(); cigarOpIter++)
	{
		if (cigarOpIter->Type == 'S')
		{
			numSoftClipped += cigarOpIter->Length;
		}
	}
	return numSoftClipped;
}

inline int GetReadEnd(const BamAlignment& alignment)
{
	return alignment.IsFirstMate() ? 0 : 1;
}

inline int GetOtherReadEnd(const BamAlignment& alignment)
{
	return OtherReadEnd(GetReadEnd(alignment));
}

inline bool IsReadPairDiscordant(const BamAlignment& alignment, double maxFragmentLength)
{
	return !(alignment.IsProperPair() &&
			 alignment.InsertSize != 0 &&
			 abs(alignment.InsertSize) <= maxFragmentLength);
}

inline bool IsReadValidConcordant(const BamAlignment& alignment, double maxSoftClipped)
{
	return (GetNumSoftClipped(alignment) <= maxSoftClipped &&
			!alignment.IsFailedQC() &&
			alignment.MapQuality > 0);
}

struct SNPInfo
{
	int position;
	char ref;
	char alt;
	
	bool operator<(const SNPInfo& snp) const
	{
		return position < snp.position;
	}
};

struct AlleleReader : PileupVisitor, DiscardAlignmentVisitor
{
	AlleleReader(BamReader& bamReader, ostream& readsFile, ostream& allelesFile, int maxFragmentLength, int maxSoftClipped)
		: mBamReader(bamReader),
		  mReadsFile(readsFile),
		  mAllelesFile(allelesFile),
		  mMaxFragmentLength(maxFragmentLength),
		  mMaxSoftClipped(maxSoftClipped),
		  mSNPStageRefID(-1),
		  mNextReadID(0)
	{
		// Fill list of reference names, create map from name to index
		for (int refID = 0; refID < bamReader.GetReferenceCount(); refID++)
		{
			mRefNames.push_back(bamReader.GetReferenceData()[refID].RefName);
			mRefIDMap[bamReader.GetReferenceData()[refID].RefName] = refID;
		}
		
		// SNPs table is indexed by bam ref id
		mSNPs.resize(mRefNames.size());
	}

	void ReadSNPs(const string& snpFilename)
	{
		// Read list of snps
		ifstream snpFile(snpFilename.c_str());
		CheckFile(snpFile, snpFilename);
		
		// SNPs table is indexed by bam ref id
		mSNPs.clear();
		mSNPs.resize(mRefNames.size());
		
		vector<string> fields;
		while (ReadTSV(snpFile, fields))
		{
			if (fields.size() < 4)
			{
				cerr << "Error: expected chr,pos,ref,alt in file " << snpFilename << endl;
				exit(1);
			}
			
			const string& chromosome = fields[0];
			
			unordered_map<string,int>::const_iterator refIDIter = mRefIDMap.find(chromosome);
			if (refIDIter == mRefIDMap.end())
			{
				continue;
			}
			
			SNPInfo snp;
			snp.position = SAFEPARSE(int, fields[1]);
			snp.ref = SAFEPARSE(char, fields[2]);
			snp.alt = SAFEPARSE(char, fields[3]);
			
			// Convert to 0-based position
			snp.position -= 1;
			
			mSNPs[refIDIter->second].push_back(snp);
		}
		
		// Sorting required for streaming 
		for (int refID = 0; refID < mSNPs.size(); refID++)
		{
			sort(mSNPs[refID].begin(), mSNPs[refID].end());
		}
	}
	
	void ProcessBam()
	{
		PileupEngine pileupEngine;
		
		pileupEngine.AddVisitor(dynamic_cast<PileupVisitor*>(this));
		pileupEngine.AddVisitor(dynamic_cast<DiscardAlignmentVisitor*>(this));
		
		BamAlignment alignment;
		while (mBamReader.GetNextAlignment(alignment))
		{
			// Classify reads pairs as discordant and ignore
			if (IsReadPairDiscordant(alignment, mMaxFragmentLength))
			{
				continue;
			}
			
			// Classify remaining reads as valid concordant reads
			bool valid = IsReadValidConcordant(alignment, mMaxSoftClipped);
			
			// Add valid concordant alignments to the queue
			if (valid)
			{
				mReadQueue.push_back(alignment);
			}
			
			// Pair up reads and classify as concordant, give passing reads an index
			unordered_map<string,BamAlignment>::iterator otherEndIter = mReadBuffer[GetOtherReadEnd(alignment)].find(alignment.Name);
			if (otherEndIter != mReadBuffer[GetOtherReadEnd(alignment)].end())
			{
				BamAlignment& alignment1 = alignment;
				BamAlignment& alignment2 = otherEndIter->second;
				
				bool valid1 = valid;
				bool valid2 = IsReadValidConcordant(alignment2, mMaxSoftClipped);
				bool validPair = valid1 && valid2;
				
				if (validPair)
				{
					// Calculate start and end of fragment alignment
					uint32_t fragmentStart = min(alignment1.Position, alignment2.Position);
					uint16_t fragmentLength = abs(alignment1.InsertSize);
					
					// Write out read alignment info
					mReadsFile.write(reinterpret_cast<const char*>(&fragmentStart), sizeof(uint32_t));
					mReadsFile.write(reinterpret_cast<const char*>(&fragmentLength), sizeof(uint16_t));
					
					// Store read id for snp stage
					mReadID[0].insert(make_pair(alignment.Name, mNextReadID));
					mReadID[1].insert(make_pair(alignment.Name, mNextReadID));
					mNextReadID++;
				}
				
				// Set status for alignments in the queue
				if (valid1)
				{
					mReadStatus[GetReadEnd(alignment1)].insert(make_pair(alignment1.Name, validPair));
				}
				if (valid2)
				{
					mReadStatus[GetReadEnd(alignment2)].insert(make_pair(alignment2.Name, validPair));
				}
				
				mReadBuffer[GetOtherReadEnd(alignment)].erase(otherEndIter);
			}
			else
			{
				mReadBuffer[GetReadEnd(alignment)].insert(make_pair(alignment.Name, alignment));
			}
			
			// Process concordant reads from the queue
			while (!mReadQueue.empty())
			{
				BamAlignment& nextAlignment = mReadQueue.front();
				
				unordered_map<string,bool>::const_iterator readStatusIter = mReadStatus[GetReadEnd(nextAlignment)].find(nextAlignment.Name);
				
				// Check for existance of read pair status
				if (readStatusIter != mReadStatus[GetReadEnd(nextAlignment)].end())
				{
					// Add valid reads to the pileup
					if (readStatusIter->second)
					{
						pileupEngine.AddAlignment(nextAlignment);
					}
					
					// Remove read status
					mReadStatus[GetReadEnd(nextAlignment)].erase(readStatusIter);
				}
				// Check for an unmatched read stuck in the queue
				else if (alignment.Position - nextAlignment.Position > 2.0 * mMaxFragmentLength)
				{
					cerr << "Warning: Could not match read " << nextAlignment.Name << endl;
				}
				// Read pair status unavailable but read not yet considered stuck
				else
				{
					break;
				}
				
				// Remove read from the queue
				mReadQueue.pop_front();
			}
		}
		
		// Process remaining concordant reads from the queue
		while (!mReadQueue.empty())
		{
			BamAlignment& nextAlignment = mReadQueue.front();
			
			unordered_map<string,bool>::const_iterator readStatusIter = mReadStatus[GetReadEnd(nextAlignment)].find(nextAlignment.Name);
			
			// Check for existance of read pair status
			if (readStatusIter != mReadStatus[GetReadEnd(nextAlignment)].end())
			{
				// Add valid reads to the pileup
				if (readStatusIter->second)
				{
					pileupEngine.AddAlignment(nextAlignment);
				}
				
				// Remove read status
				mReadStatus[GetReadEnd(nextAlignment)].erase(readStatusIter);
			}
			// Check for an unmatched read stuck in the queue
			else
			{
				cerr << "Warning: Could not match read " << nextAlignment.Name << endl;
			}
			
			// Remove read from the queue
			mReadQueue.pop_front();
		}
		
		pileupEngine.Flush();
	}
	
	void Visit(const PileupPosition& pileupData)
	{
		// Check if we have switched chromosomes
		if (pileupData.RefId != mSNPStageRefID)
		{
			mSNPIter = mSNPs[pileupData.RefId].begin();
			mSNPIterEnd = mSNPs[pileupData.RefId].end();
			
			mSNPStageRefID = pileupData.RefId;
		}
		
		// Catch up to the current pileup position
		while (mSNPIter != mSNPIterEnd && mSNPIter->position < pileupData.Position)
		{
			mSNPIter++;
		}
		
		// Return if we passed the current pileup position
		if (mSNPIter == mSNPIterEnd || mSNPIter->position > pileupData.Position)
		{
			return;
		}
		
		assert(mSNPIter->position == pileupData.Position);
		
		// Label each read as reference or alternate
		for (vector<PileupAlignment>::const_iterator pileupIter = pileupData.PileupAlignments.begin(); pileupIter != pileupData.PileupAlignments.end(); ++pileupIter)
		{
			const PileupAlignment& pileupAlignment = (*pileupIter);
			const BamAlignment& alignment = pileupAlignment.Alignment;
			
			if (pileupAlignment.IsCurrentDeletion)
			{
				continue;
			}
			
			char base = toupper(alignment.QueryBases.at(pileupAlignment.PositionInAlignment));
			
			uint8_t isAlt = 0;
			if (base == mSNPIter->alt)
			{
				isAlt = 1;
			}
			else if (base != mSNPIter->ref)
			{
				continue;
			}
			
			uint32_t readID = mReadID[GetReadEnd(alignment)][alignment.Name];
			
			// Output 1-based positions
			uint32_t position = mSNPIter->position + 1;
			
			// Write out snp info
			mAllelesFile.write(reinterpret_cast<const char*>(&readID), sizeof(uint32_t));
			mAllelesFile.write(reinterpret_cast<const char*>(&position), sizeof(uint32_t));
			mAllelesFile.write(reinterpret_cast<const char*>(&isAlt), sizeof(uint8_t));
		}
	}
	
	void Visit(const BamAlignment& alignment)
	{
		mReadID[GetReadEnd(alignment)].erase(alignment.Name);
	}
	
	BamReader& mBamReader;
	ostream& mReadsFile;
	ostream& mAllelesFile;
	
	int mMaxFragmentLength;
	int mMaxSoftClipped;
	
	vector<string> mRefNames;
	unordered_map<string,int> mRefIDMap;
	
	deque<BamAlignment> mReadQueue;
	unordered_map<string,BamAlignment> mReadBuffer[2];
	unordered_map<string,bool> mReadStatus[2];
	unordered_map<string,int> mReadID[2];
	
	uint32_t mNextReadID;
	int mSNPStageRefID;
	
	vector<vector<SNPInfo> > mSNPs;
	vector<SNPInfo>::const_iterator mSNPIter;
	vector<SNPInfo>::const_iterator mSNPIterEnd;
};


int main(int argc, char* argv[])
{
	string bamFilename;
	string snpFilename;
	int maxFragmentLength;
	int maxSoftClipped;
	string chromosome;
	string readsFilename;
	string allelesFilename;
	
	try
	{
		TCLAP::CmdLine cmd("Bam Concordant Read Extractor");
		TCLAP::ValueArg<string> bamFilenameArg("b","bam","Bam Filename",true,"","string",cmd);
		TCLAP::ValueArg<string> snpFilenameArg("s","snp","SNP Filename",false,"","string",cmd);
		TCLAP::ValueArg<int> maxFragmentLengthArg("","flen","Maximum Fragment Length",false,1000,"integer",cmd);
		TCLAP::ValueArg<int> maxSoftClippedArg("","clipmax","Maximum Allowable Soft Clipped",false,8,"integer",cmd);
		TCLAP::ValueArg<string> chromosomeArg("c","chr","Restrict to Specific Chromosome",true,"","string",cmd);
		TCLAP::ValueArg<string> readsFilenameArg("r","reads","Read Alignments Filename",true,"","string",cmd);
		TCLAP::ValueArg<string> allelesFilenameArg("a","alleles","Read Alleles Filename",true,"","string",cmd);
		cmd.parse(argc,argv);
		
		bamFilename = bamFilenameArg.getValue();
		snpFilename = snpFilenameArg.getValue();
		maxFragmentLength = maxFragmentLengthArg.getValue();
		maxSoftClipped = maxSoftClippedArg.getValue();
		chromosome = chromosomeArg.getValue();
		readsFilename = readsFilenameArg.getValue();
		allelesFilename = allelesFilenameArg.getValue();
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
	
	if (!bamReader.LocateIndex())
	{
		cerr << "Error: Unable to find index for bam file " << bamFilename << endl;
		exit(1);
	}
	
	int refID = bamReader.GetReferenceID(chromosome);

	if (refID < 0)
	{
		string altChromosome;

		if (chromosome.substr(0, 3) == "chr")
		{
			altChromosome = chromosome.substr(3);
		}
		else
		{
			altChromosome = "chr" + chromosome;
		}

		refID = bamReader.GetReferenceID(altChromosome);

		if (refID < 0)
		{
			cerr << "Error: Unable to find chromosome " << chromosome << " or " << altChromosome << endl;
			exit(1);
		}
	}

	bamReader.SetRegion(BamRegion(refID, 0, refID+1, 1));
	
	ofstream readsFile(readsFilename.c_str(), ios_base::binary);
	CheckFile(readsFile, readsFilename);
	iostreams::filtering_stream<iostreams::output> readsFilter;
	readsFilter.push(iostreams::gzip_compressor());
	readsFilter.push(readsFile);
	
	ofstream allelesFile(allelesFilename.c_str(), ios_base::binary);
	CheckFile(allelesFile, allelesFilename);
	iostreams::filtering_stream<iostreams::output> allelesFilter;
	allelesFilter.push(iostreams::gzip_compressor());
	allelesFilter.push(allelesFile);
	
	// BUG: boost bug with empty files
	char dummy = 'b';
	readsFilter.component<iostreams::gzip_compressor>(0)->write(readsFile, &dummy, 0);
	allelesFilter.component<iostreams::gzip_compressor>(0)->write(allelesFile, &dummy, 0);
	
	AlleleReader alleleReader(bamReader, readsFilter, allelesFilter, maxFragmentLength, maxSoftClipped);

	if (!snpFilename.empty())
	{
		alleleReader.ReadSNPs(snpFilename);
	}
	
	alleleReader.ProcessBam();
}


