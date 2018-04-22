#include <iostream>
#include <stdlib.h>
#include <boost/iostreams/filter/bzip2.hpp>
#include "TSVFile.h"

TSVFile::TSVFile(string filename) : in(NULL) {
    file.open(filename.c_str());

    if (!file) {
        cout << "Cannot open TSV file '" << filename << "'" << endl;
        exit(1);
    }

    if (filename.find(".bz2") != string::npos) {
        // It's a compressed file so decompress it on th fly
        filter.push(boost::iostreams::bzip2_decompressor());
    }

    filter.push(file);
    in = new istream(&filter);

    // Read and discard the header
    getline();
}

TSVFile::~TSVFile() {
    if (in != NULL) {
        delete in;
        in = NULL;
    }

    close();
}

TSVFile& TSVFile::operator >> (int& x) {
    (*in) >> x;
    return *this;
}

TSVFile& TSVFile::operator >> (double& x) {
    (*in) >> x;
    return *this;
}

string TSVFile::getline() {
    string line;
    std::getline(*in, line);
    return line;
}

bool TSVFile::eof() {
    return in->eof();
}

void TSVFile::close() {
    file.close();
}

