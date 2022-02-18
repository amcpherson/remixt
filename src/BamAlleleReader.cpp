#include "BamAlleleReader.h"
#include "external/bamtools/src/api/BamReader.h"
#include "external/bamtools/src/utils/bamtools_pileup_engine.h"

#include <fstream>
#include <iostream>
#include <string>
#include <map>
#include <set>
#include <sstream>
#include <limits>
#include <stdexcept>
#include <algorithm>

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

inline int OtherReadEnd(int readEnd)
{
	return (1 - readEnd);
}

inline int GetOtherReadEnd(const BamAlignment& alignment)
{
	return OtherReadEnd(GetReadEnd(alignment));
}

inline bool IsReadPairDiscordant(const BamAlignment& alignment, double maxFragmentLength, bool checkProperPair)
{
	return !((alignment.IsProperPair() || !checkProperPair) &&
	         alignment.InsertSize != 0 &&
	         abs(alignment.InsertSize) <= maxFragmentLength);
}

inline bool IsReadValidConcordant(const BamAlignment& alignment, double maxSoftClipped)
{
	return (GetNumSoftClipped(alignment) <= maxSoftClipped &&
	        alignment.IsMapped() &&
	        !alignment.IsFailedQC());
}

AlleleReader::AlleleReader(const string& bamFilename,
                           const string& snpFilename,
                           const string& chromosome,
                           int maxFragmentLength,
                           int maxSoftClipped,
                           bool checkProperPair)
	: mChromosome(chromosome),
	  mMaxFragmentLength(maxFragmentLength),
	  mMaxSoftClipped(maxSoftClipped),
	  mRefID(-1),
	  mNextFragmentID(0),
	  mCheckProperPair(checkProperPair)
{
	if (!mBamReader.Open(bamFilename))
	{
		throw invalid_argument("Unable to open bam file " + bamFilename);
	}
	
	if (!mBamReader.LocateIndex())
	{
		throw invalid_argument("Unable to find index for bam file " + bamFilename);
	}

	// Alternate chromosome, ucsc or ensembl
	if (mChromosome.substr(0, 3) == "chr")
	{
		mAlternateChromosome = mChromosome.substr(3);
	}
	else
	{
		mAlternateChromosome = "chr" + mChromosome;
	}

	// Querty bam for either chromosome
	mRefID = mBamReader.GetReferenceID(mChromosome);

	if (mRefID < 0)
	{
		mRefID = mBamReader.GetReferenceID(mAlternateChromosome);

		if (mRefID < 0)
		{
			throw out_of_range("Unable to find chromosome " + mChromosome + " or " + mAlternateChromosome);
		}
	}

	// Set region in bam
	mBamReader.SetRegion(BamRegion(mRefID, 0, mRefID+1, 1));

	if (!snpFilename.empty())
	{
		ReadSNPs(snpFilename);
	}
	
	mPileupEngine.AddVisitor(dynamic_cast<PileupVisitor*>(this));
	mPileupEngine.AddVisitor(dynamic_cast<DiscardAlignmentVisitor*>(this));
}

void AlleleReader::ReadSNPs(const string& snpFilename)
{
	// Read list of snps
	ifstream snpFile(snpFilename.c_str());
	if (!snpFile.good())
	{
		throw ios_base::failure("Error: Unable to open " + snpFilename);
	}
	
	// clear SNPs table
	mSNPs.clear();
	
	string line;
	while (getline(snpFile, line))
	{
		istringstream lineStream(line);

		string chromosome;
		int position;
		string ref;
		string alt;

		lineStream >> chromosome >> position >> ref >> alt;

		if (chromosome != mChromosome && chromosome != mAlternateChromosome)
		{
			continue;
		}

		if (ref.size() > 1)
		{
			throw invalid_argument("expected nucletide got " + ref);
		}

		if (alt.size() > 1)
		{
			throw invalid_argument("expected nucletide got " + alt);
		}

		SNPInfo snp;

		// Convert to 0-based position
		snp.position = position - 1;
		
		snp.ref = ref[0];
		snp.alt = alt[0];
		
		mSNPs.push_back(snp);
	}
	
	// Sorting required for streaming
	sort(mSNPs.begin(), mSNPs.end());

	// Initialize iterators for sequential access
	mSNPIter = mSNPs.begin();
}

bool AlleleReader::ReadAlignments(int maxAlignments)
{
	mFragmentData.clear();
	mAlleleData.clear();

	bool finishedGetAlignments = false;
	for (int idx = 0; idx < maxAlignments; idx++)
	{
		// Get next alignment if one is available
		BamAlignment alignment;
		finishedGetAlignments = !mBamReader.GetNextAlignment(alignment);

		// Skip all secondary alignments
		if (!alignment.IsPrimaryAlignment())
		{
			continue;
		}

		// Break out if finished getting alignments
		if (finishedGetAlignments)
		{
			break;
		}

		// Classify reads pairs as discordant and ignore
		if (IsReadPairDiscordant(alignment, mMaxFragmentLength, mCheckProperPair))
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
		map<string,BamAlignment>::iterator otherEndIter = mReadBuffer[GetOtherReadEnd(alignment)].find(alignment.Name);
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
				int fragmentStart = min(alignment1.Position, alignment2.Position);
				int fragmentEnd = fragmentStart + abs(alignment1.InsertSize);

				// Set as duplicate if either is duplicate
				int isDuplicate = (int)(alignment1.IsDuplicate() || alignment2.IsDuplicate());

				// Fragment mapping quality as minimum of read mapping qualities
				int mappingQuality = min(alignment1.MapQuality, alignment2.MapQuality);

				// Integer ID for the fragment based on order in which they appear in the BAM
				int fragmentID = mNextFragmentID;
				if (fragmentID == numeric_limits<int>::max())
				{
					throw out_of_range("Fragment ID overflow");
				}
				mNextFragmentID++;

				// Store fragment id for snp stage
				mFragmentID[0].insert(make_pair(alignment.Name, fragmentID));
				mFragmentID[1].insert(make_pair(alignment.Name, fragmentID));

				// Save out read alignment info
				mFragmentData.push_back(FragmentData(fragmentID, fragmentStart, fragmentEnd, mappingQuality, isDuplicate));
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
			
			map<string,bool>::iterator readStatusIter = mReadStatus[GetReadEnd(nextAlignment)].find(nextAlignment.Name);
			
			// Check for existance of read pair status
			if (readStatusIter != mReadStatus[GetReadEnd(nextAlignment)].end())
			{
				// Add valid reads to the pileup
				if (readStatusIter->second)
				{
					mPileupEngine.AddAlignment(nextAlignment);
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

	// Cleanup after finished getting alignments
	if (finishedGetAlignments)
	{
		// Process remaining concordant reads from the queue
		while (!mReadQueue.empty())
		{
			BamAlignment& nextAlignment = mReadQueue.front();
			
			map<string,bool>::iterator readStatusIter = mReadStatus[GetReadEnd(nextAlignment)].find(nextAlignment.Name);
			
			// Check for existance of read pair status
			if (readStatusIter != mReadStatus[GetReadEnd(nextAlignment)].end())
			{
				// Add valid reads to the pileup
				if (readStatusIter->second)
				{
					mPileupEngine.AddAlignment(nextAlignment);
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
		
		mPileupEngine.Flush();
	}

	return !mFragmentData.empty() || !mAlleleData.empty();
}

void AlleleReader::Visit(const PileupPosition& pileupData)
{
	// Check if we are on the correct chromosome
	if (pileupData.RefId != mRefID)
	{
		return;
	}

	// Catch up to the current pileup position
	while (mSNPIter != mSNPs.end() && mSNPIter->position < pileupData.Position)
	{
		mSNPIter++;
	}
	
	// Return if we passed the current pileup position
	if (mSNPIter == mSNPs.end() || mSNPIter->position > pileupData.Position)
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
		
		int isAlt = 0;
		if (base == mSNPIter->alt)
		{
			isAlt = 1;
		}
		else if (base != mSNPIter->ref)
		{
			continue;
		}
		
		int fragmentID = mFragmentID[GetReadEnd(alignment)][alignment.Name];
		
		// Output 1-based positions
		int position = mSNPIter->position + 1;
		
		// Save out snp info
		mAlleleData.push_back(AlleleData(fragmentID, position, isAlt));
	}
}

void AlleleReader::Visit(const BamAlignment& alignment)
{
	mFragmentID[GetReadEnd(alignment)].erase(alignment.Name);
}

