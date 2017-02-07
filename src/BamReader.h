#ifndef BAMREADER_H_
#define BAMREADER_H_

#include <string>
#include <deque>
#include <map>

#include "external/bamtools/src/api/BamReader.h"
#include "external/bamtools/src/utils/bamtools_pileup_engine.h"


struct FragmentData
{
	FragmentData(
		int fragmentID,
		int fragmentStart,
		int fragmentEnd,
		int mappingQuality,
		int isDuplicate
	):
			fragmentID(fragmentID),
			fragmentStart(fragmentStart),
			fragmentEnd(fragmentEnd),
			mappingQuality(mappingQuality),
			isDuplicate(isDuplicate)
	{}

	int fragmentID;
	int fragmentStart;
	int fragmentEnd;
	int mappingQuality;
	int isDuplicate;
};

struct AlleleData
{
	AlleleData(
		int fragmentID,
		int position,
		int isAlt
	):
		fragmentID(fragmentID),
		position(position),
		isAlt(isAlt)
	{}

	int fragmentID;
	int position;
	int isAlt;
};

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

struct AlleleReader : BamTools::PileupVisitor, BamTools::DiscardAlignmentVisitor
{
	AlleleReader(const std::string& bamFilename,
	             const std::string& snpFilename,
	             const std::string& chromosome,
	             int maxFragmentLength,
	             int maxSoftClipped,
	             bool checkProperPair);

	void ReadSNPs(const std::string& snpFilename);

	bool ReadAlignments(int maxAlignments);

	void Visit(const BamTools::PileupPosition& pileupData);
	
	void Visit(const BamTools::BamAlignment& alignment);
	
	BamTools::BamReader mBamReader;
	const std::string& mChromosome;
	std::string mAlternateChromosome;

	std::vector<FragmentData> mFragmentData;
	std::vector<AlleleData> mAlleleData;
	
	int mMaxFragmentLength;
	int mMaxSoftClipped;
	bool mCheckProperPair;
	
	std::deque<BamTools::BamAlignment> mReadQueue;
	std::map<std::string,BamTools::BamAlignment> mReadBuffer[2];
	std::map<std::string,bool> mReadStatus[2];
	std::map<std::string,int> mFragmentID[2];
	
	int mRefID;
	int mNextFragmentID;
	
	std::vector<SNPInfo> mSNPs;
	std::vector<SNPInfo>::const_iterator mSNPIter;

	BamTools::PileupEngine mPileupEngine;
};

#endif
