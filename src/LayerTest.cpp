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

int main(int argc, char* argv[]) {
  START_EASYLOGGINGPP(argc, argv);
  microunit::UnitTester::Run();
  return 0;
}