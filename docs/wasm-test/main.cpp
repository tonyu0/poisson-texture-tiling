#include <string>
#include <algorithm>
#include <emscripten/bind.h>

using namespace emscripten;

std::string toUpperCase(std::string input)
{
    std::transform(input.begin(), input.end(), input.begin(), ::toupper);
    return input;
}

// Bind so that JS can use
EMSCRIPTEN_BINDINGS(my_module)
{
    function("toUpperCase", &toUpperCase);
}