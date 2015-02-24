#include "Common.h"
#include "DebugCheck.h"

#include <iostream>
#include <fstream>
#include <boost/algorithm/string.hpp>


using namespace std;
using namespace boost;


int OtherStrand(int strand)
{
	return (1 - strand);
}

int OtherReadEnd(int readEnd)
{
	return (1 - readEnd);
}

int OtherClusterEnd(int clusterEnd)
{
	return (1 - clusterEnd);
}

void ReverseComplement(string& sequence)
{
	reverse(sequence.begin(), sequence.end());
	
	for (int seqIndex = 0; seqIndex < sequence.size(); seqIndex++)
	{
		char nucleotide = sequence[seqIndex];
		
		switch (nucleotide)
		{
			case 'A': nucleotide = 'T'; break;
			case 'C': nucleotide = 'G'; break;
			case 'T': nucleotide = 'A'; break;
			case 'G': nucleotide = 'C'; break;
			case 'a': nucleotide = 't'; break;
			case 'c': nucleotide = 'g'; break;
			case 't': nucleotide = 'a'; break;
			case 'g': nucleotide = 'c'; break;
		}
		
		sequence[seqIndex] = nucleotide;
	}
}

double normalpdf(double x, double mu, double sigma)
{
	double coeff = 1.0 / ((double)sigma * sqrt(2 * M_PI));
	
	double dist = (((double)x - (double)mu) / (double)sigma);
	double exponent = -0.5 * dist * dist;
	
	return coeff * exp(exponent);
}

int InterpretStrand(const string& strand)
{
	DebugCheck(strand == "+" || strand == "-");

	if (strand == "+")
	{
		return PlusStrand;
	}
	else
	{
		return MinusStrand;
	}
}

void CheckFile(const ios& file, const string& filename)
{
	if (!file.good())
	{
		cerr << "Error: Unable to open " << filename << endl;
		exit(1);
	}	
}

bool ReadTSV(istream& file, vector<string>& fields)
{
	string line;
	if (!getline(file, line))
	{
		return false;
	}
	
	split(fields, line, is_any_of("\t"));
	
	return true;
}

