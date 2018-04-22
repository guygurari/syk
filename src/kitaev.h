#ifndef KITAEV_H__ 
#define KITAEV_H__

#include <string>
#include <fstream>
#include "KitaevHamiltonian.h"

using namespace std;

class TwoPointOutput {
public:
    TwoPointOutput();
    virtual ~TwoPointOutput();
    virtual void write(double T, double t, cpx correlator) = 0;
    virtual void close() = 0;
};

class TwoPointFileOutput : public TwoPointOutput {
public:
    TwoPointFileOutput(string filename);
    virtual ~TwoPointFileOutput();
    virtual void write(double T, double t, cpx correlator);
    virtual void close();
private:
    ofstream file;
};

class TwoPointNullOutput : public TwoPointOutput {
public:
    TwoPointNullOutput();
    virtual void write(double T, double t, cpx correlator);
    virtual void close();
};

#endif // KITAEV_H__

