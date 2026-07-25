#include "app/Application.hpp"

#include <cstdlib>
#include <iostream>

using namespace vkrt::app;

int main()
{
    try
    {
        Application app;
        app.run();
    }
    catch (const std::exception& e)
    {
        std::cerr << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    return 0;
}
