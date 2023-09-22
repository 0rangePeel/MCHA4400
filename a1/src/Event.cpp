#include "State.h"
#include "Event.h"

Event::Event(double time)
    : time_(time)
{
    
}

void Event::process(State & state)
{
    std::cout << "Before Predict: muSize   = " << state.density.mean().size() << std::endl;
    std::cout << "Before Predict: sqrtSize = " << state.density.sqrtCov().rows() << " " << state.density.sqrtCov().cols() << std::endl;
    // Time update
    state.predict(time_);
    std::cout << "After Predict: muSize   = " << state.density.mean().size() << std::endl;
    std::cout << "After Predict: sqrtSize = " << state.density.sqrtCov().rows() << " " << state.density.sqrtCov().cols() << std::endl;
    
    // Event-specific implementation
    update(state);
}