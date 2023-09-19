#include "State.h"
#include "Event.h"

Event::Event(double time)
    : time_(time)
{
    
}

void Event::process(State & state)
{
    // Time update
    std::cout << "Event.cpp - start" << std::endl;
    state.predict(time_);
    std::cout << "Event.cpp - predict" << std::endl;

    // Event-specific implementation
    update(state);
    std::cout << "Event.cpp - end" << std::endl;
}