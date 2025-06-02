#ifndef __CPU_TIMER_H__
#define __CPU_TIMER_H__

#include <chrono>

class CpuTimer
{
public:
    CpuTimer() = default;

    void Start()
    {
        start = std::chrono::steady_clock::now();
    }

    void Stop()
    {
        end = std::chrono::steady_clock::now();
    }

    // ms
    float Elapsed() const
    {
        return std::chrono::duration<float, std::milli>(end - start).count();
    }

private:
    std::chrono::steady_clock::time_point start, end;
};

#endif /* __CPU_TIMER_H__ */