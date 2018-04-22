#ifndef TSV_FILE_H__ 
#define TSV_FILE_H__

#include <fstream>
#include <iomanip>
#include <string>
#include <boost/iostreams/filtering_streambuf.hpp>
#include "defs.h"

using namespace std;

/*
   A tab-separated file. The first line is a header that is discarded.
  
   Synopsis:
  
   TSVFile file(filename);
 
   while (true) {
       int i;
       double ev;
       file >> i >> ev;
 
       if (file.eof()) {
           break;
       }
 
       // ... Do something with i, ev ...
   }
 
   file.close();

 */
class TSVFile {
public:
    TSVFile(string filename);
    ~TSVFile();

    TSVFile& operator >> (int& x);
    TSVFile& operator >> (double& x);
    string getline();

    // Are we there yet.
    bool eof();

    void close();

private:
    ifstream file;
    boost::iostreams::filtering_streambuf<boost::iostreams::input> filter;
    istream* in;
};

#endif // TSV_FILE_H__
