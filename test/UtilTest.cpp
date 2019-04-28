//============================================================================
// Name : LayerTest.cpp
// Author : David Nogueira
//============================================================================

#include "Layer.h"
#include "Sample.h"
#include "Utils.h"

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <sstream>
#include <fstream>
#include <vector>
#include <algorithm>
#include "microunit.h"
#include "easylogging++.h"

INITIALIZE_EASYLOGGINGPP

UNIT(ErrorFunctionManagerTest){
    LOG(INFO) << "Test of ErrorFunctionManager";

    utils::functionTwoArgDeriv error =
            utils::ErrorFunctionsManager::Singleton()
            .GetErrorFunctionPair("error");

    // test against some numbers
    for( double x = -100; x < 100; x += 0.1 )
    {
        for( double y = -100; y < 100; y += 0.1 )
        {
            ASSERT_TRUE( error.first(x,y) == utils::quadratic_error( x, y ) );
            ASSERT_TRUE( error.second(x,y) == utils::deriv_quadratic_error( x, y ) );
        }
    }

}

UNIT(ActivationFunctionsManagerTest){

    LOG(INFO) << "Test of the class ActivationFunctionsManager";

    // get some of the functions
    utils::functionWithDeriv sigmoid =
            utils::ActivationFunctionsManager::Singleton()
            .GetActivationFunctionPair("sigmoid");
    utils::functionWithDeriv tanh =
            utils::ActivationFunctionsManager::Singleton()
            .GetActivationFunctionPair("tanh");
    utils::functionWithDeriv linear =
            utils::ActivationFunctionsManager::Singleton()
            .GetActivationFunctionPair("linear");
    utils::functionWithDeriv relu =
            utils::ActivationFunctionsManager::Singleton()
            .GetActivationFunctionPair("relu");


    // test against some numbers
    for( double x = -100; x < 100; x += 0.1 )
    {
        ASSERT_TRUE( sigmoid.first(x) == utils::sigmoid( x ) );
        ASSERT_TRUE( sigmoid.second(x) == utils::deriv_sigmoid( x ) );
        ASSERT_TRUE( tanh.first(x) == utils::hyperbolic_tan( x ) );
        ASSERT_TRUE( tanh.second(x) == utils::deriv_hyperbolic_tan( x ) );
        ASSERT_TRUE( linear.first(x) == utils::linear( x ) );
        ASSERT_TRUE( linear.second(x) == utils::deriv_linear( x ) );
        ASSERT_TRUE( relu.first(x) == utils::relu( x ) );
        ASSERT_TRUE( relu.second(x) == utils::deriv_relu( x ) );
    }

}




int main(int argc, char* argv[]) {
  START_EASYLOGGINGPP(argc, argv);
  // Load configuration from file
  el::Configurations conf("easylog.conf");
  el::Loggers::reconfigureAllLoggers(conf);
  microunit::UnitTester::Run();
  return 0;
}
