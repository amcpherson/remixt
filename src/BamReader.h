#ifndef BAMREADER_H_
#define BAMREADER_H_

#import <string>

extern void ExtractReads(
	const std::string& bamFilename,
	const std::string& snpFilename,
	int maxFragmentLength,
	int maxSoftClipped,
	const std::string& chromosome,
	const std::string& readsFilename,
	const std::string& allelesFilename,
	bool removeDuplicates,
	int mapQualThreshold);

#endif
