#ifndef TIMER_H__ 
#define TIMER_H__

#include <ctime>
#include <string>

using namespace std;

class Timer {
public:
    Timer();

    // Reset the timer
    void reset();

    // How many seconds passed since construction / reset
    double seconds();

    // How many msecs passed since construction / reset
    double msecs();

    // Print the elapsed time in seconds
    void print(string title = "");

    // Print the elapsed time in milliseconds
    void print_msec(string title = "");

    // Print the elapsed time in milliseconds, and the time per
    // iteration
    void print_msec(int iterations);

private:
    clock_t begin;
};

#endif // TIMER_H__
