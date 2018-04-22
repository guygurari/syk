#include <iostream>
#include <cmath>
#include <stdio.h>
#include "Timer.h"

using namespace std;

Timer::Timer() {
    reset();
}

void Timer::reset() {
    begin = clock();
}

double Timer::seconds() {
    clock_t end = clock();
    double seconds = (double) (end - begin) / (double) CLOCKS_PER_SEC;
    return seconds;
}

double Timer::msecs() {
    return seconds() * 1000.;
}

void Timer::print(string title) {
    printf("%sThat took %.1f seconds\n", title.c_str(), seconds());
}

void Timer::print_msec(string title) {
    cout << title << "That took " << msecs() << " msec" << endl;
}

void Timer::print_msec(int iterations) {
    double ms = msecs();
    cout << "That took " << ms << " msec ("
         << ms / iterations << " msec / iteration)"
         << endl;
}
