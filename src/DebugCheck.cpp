#include "DebugCheck.h"
#include "stacktrace.h"

#include <iostream>
#include <stdlib.h>

void _ReportFailure(const char* expr, const char* file, int line)
{
	std::cerr << "Error: " << expr << " failed on line: " << line << " of " << file << std::endl;
	print_stacktrace();
	exit(1);
}
