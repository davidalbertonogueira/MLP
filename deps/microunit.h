/**
* @file microunit.h
* @version 0.2
* @author Sebastiao Salvador de Miranda (ssm)
* @brief Tiny library for cpp unit testing. Should work on any c++11 compiler.
*
* Simply include this header in your test implementation file (e.g., main.cpp)
* and call microunit::UnitTester::Run() in the function main(). To register
* a new unit test case, use the macro UNIT (See the example below). Inside the
* test case body, you can use the following macros to control the result
* of the test.
*
* @li PASS() : Pass the test and return.
* @li FAIL() : Fail the test and return.
* @li ASSERT_TRUE(condition) : If the condition does not hold, fail and return.
* @li ASSERT_FALSE(condition) : If the condition holds, fail and return.
*
* @code{.cpp}
*  UNIT(Test_Two_Plus_Two) {
*    ASSERT_TRUE(2 + 2 == 4);
*  };
*  // ...
*  int main(){
*    return microunit::UnitTester::Run() ? 0 : -1;
*  }
* @endcode
*
* @copyright Copyright (c) 2016-2017, Sebastiao Salvador de Miranda.
*            All rights reserved. See licence below.
*
* Redistribution and use in source and binary forms, with or without
* modification, are permitted provided that the following conditions are
* met:
*
* (1) Redistributions of source code must retain the above copyright
* notice, this list of conditions and the following disclaimer.
*
* (2) Redistributions in binary form must reproduce the above copyright
* notice, this list of conditions and the following disclaimer in
* the documentation and/or other materials provided with the
* distribution.
*
* (3) The name of the author may not be used to
* endorse or promote products derived from this software without
* specific prior written permission.
*
* THIS SOFTWARE IS PROVIDED BY THE AUTHOR ``AS IS'' AND ANY EXPRESS OR
* IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
* WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
* DISCLAIMED. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY DIRECT,
* INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
* (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
* SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
* HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
* STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING
* IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
* POSSIBILITY OF SUCH DAMAGE.
*/

#ifndef _MICROUNIT_MICROUNIT_H_
#define _MICROUNIT_MICROUNIT_H_
#include <string.h>
#include <map>
#include <string>
#include <vector>
#include <iostream>

/**
* @brief Helper macros to get current logging filename
*/
#if defined(_WIN32)
#define __FILENAME__ (strrchr(__FILE__, '\\') ?                                \
strrchr(__FILE__, '\\') + 1 : __FILE__)
#else
#define __FILENAME__ (strrchr(__FILE__, '/') ?                                 \
strrchr(__FILE__, '/') + 1 : __FILE__)
#endif
#define MICROUNIT_SEPARATOR "----------------------------------------"         \
                            "----------------------------------------"
#if defined(_WIN32)
#include "windows.h"
#endif

namespace microunit {
  const static int COLORCODE_GREY{ 7 };
  const static int COLORCODE_GREEN{ 10 };
  const static int COLORCODE_RED{ 12 };
  const static int COLORCODE_YELLOW{ 14 };

  /**
  * @brief Helper class to convert from color codes to ansi escape codes
  *        Used to print color in non-win32 systems.
  */
  std::string ColorCodeToANSI(const int color_code) {
    switch (color_code) {
    case COLORCODE_GREY: return "\033[22;37m";
    case COLORCODE_GREEN: return "\033[01;32m";
    case COLORCODE_RED: return "\033[01;31m";
    case COLORCODE_YELLOW: return "\033[01;33m";
    default: return "";
    }
  }

  /**
  * @brief Helper function to change the current terminal color.
  * @param [in] color_code Input color code.
  */
  void SetTerminalColor(int color_code) {
#if defined(_WIN32)
    HANDLE handler = GetStdHandle(STD_OUTPUT_HANDLE);
    CONSOLE_SCREEN_BUFFER_INFO buffer_info;
    GetConsoleScreenBufferInfo(handler, &buffer_info);
    SetConsoleTextAttribute(handler, ((buffer_info.wAttributes & 0xFFF0) |
      (WORD)color_code));
#else
    std::cout << ColorCodeToANSI(color_code);
#endif
  }

  /**
  * @brief Helper class to be used as a iostream manipulator and change the
  *        terminal color.
  */
  class Color {
  public:
    Color(int code) : code_(code) {}
    void Set() const {
      SetTerminalColor(code_);
    }
    int code() const { return code_; }
  private:
    int code_;
  };

  const static Color Grey{ COLORCODE_GREY };
  const static Color Green{ COLORCODE_GREEN };
  const static Color Red{ COLORCODE_RED };
  const static Color Yellow{ COLORCODE_YELLOW };

  /**
  * @brief Helper class to be used in a cout streaming statement. Resets to
  *        the default terminal color upon statement completion.
  */
  class SaveColor {
  public:
    ~SaveColor() {
      SetTerminalColor(Grey.code());
    };
  };

  /**
  * @brief Helper class to be used in a cout streaming statement. Puts a line
  *        break upon statement completion.
  */
  class EndingLineBreak {
  public:
    ~EndingLineBreak() {
      std::cout << std::endl;
    };
  };
}

/** @brief Operator to allow using SaveColor class with an ostream */
inline std::ostream& operator<<(std::ostream& os,
  const microunit::SaveColor& obj) {
  return os;
}

/** @brief Operator to allow using EndingLineBreak class with an ostream */
inline std::ostream& operator<<(std::ostream& os,
  const microunit::EndingLineBreak& obj) {
  return os;
}

/** @brief Operator to allow using Color class with an ostream */
inline std::ostream& operator<<(std::ostream& os,
  const microunit::Color& color) {
  color.Set();
  return os;
}

/**
* @brief Macro for writing to the terminal an INFO-level log
*/
#define TERMINAL_INFO std::cout << microunit::SaveColor{} <<                   \
microunit::EndingLineBreak{} << microunit::Yellow << "[    ] "
#define LOG_INFO TERMINAL_INFO << __FILENAME__ << ":" << __LINE__ << ": "

/**
* @brief Macro for writing to the terminal a BAD-level log
*/
#define TERMINAL_BAD std::cout << microunit::SaveColor{} <<                    \
microunit::EndingLineBreak{} << microunit::Red << "[    ] "
#define LOG_BAD TERMINAL_BAD << __FILENAME__ << ":" << __LINE__ << ": "

/**
* @brief Macro for writing to the terminal a GOOD-level log
*/
#define TERMINAL_GOOD std::cout << microunit::SaveColor{} <<                   \
microunit::EndingLineBreak{} << microunit::Green << "[    ] "
#define LOG_GOOD TERMINAL_GOOD << __FILENAME__ << ":" << __LINE__ << ": "

namespace microunit {
  /**
  * @brief Result of a unit test.
  */
  struct UnitFunctionResult {
    bool success{ true };
  };

  /**
  * @brief Unit test function type.
  */
  typedef void(*UnitFunction)(UnitFunctionResult*);

  /**
  * @brief Main class for unit test management. This class is a singleton
  *        and maintains a list of all registered unit test cases.
  */
  class UnitTester {
  public:
    /**
    * @brief Run all the registered unit test cases.
    * @returns True if all tests pass, false otherwise.
    */
    static bool Run() {
      std::vector<std::string> failures, sucesses;

      TERMINAL_INFO
        << "Will run " << Instance().unitfunction_map_.size()
        << " test cases";

      // Iterate all registered unit tests
      for (auto& unit : Instance().unitfunction_map_) {
        std::cout << MICROUNIT_SEPARATOR << std::endl;
        TERMINAL_GOOD << "Test case '" << unit.first << "'";

        // Run the unit test
        UnitFunctionResult result;
        unit.second(&result);

        if (!result.success) {
          TERMINAL_BAD << "Failed test";
          failures.push_back(unit.first);
        }
        else {
          TERMINAL_GOOD << "Passed test";
          sucesses.push_back(unit.first);
        }
      }
      std::cout
        << MICROUNIT_SEPARATOR << std::endl
        << MICROUNIT_SEPARATOR << std::endl;

      TERMINAL_GOOD << "Passed " << sucesses.size()
        << " test cases:";
      for (const auto& success_t : sucesses) {
        TERMINAL_GOOD << success_t;
      }
      std::cout << MICROUNIT_SEPARATOR << std::endl;

      // Output result summary
      if (failures.empty()) {
        TERMINAL_GOOD << "All tests passed";
        std::cout << MICROUNIT_SEPARATOR << std::endl;
        return true;
      }
      else {
        TERMINAL_BAD << "Failed " << failures.size()
          << " test cases:";
        for (const auto& failure : failures) {
          TERMINAL_BAD << failure;
        }
        std::cout << MICROUNIT_SEPARATOR << std::endl;
        return false;
      }
    }

    /**
    * @brief Register a unit test case function. In regular library client usage,
    *        this doesn't need to be called, and the macro UNIT should be used
    *        instead.
    * @param [in] name  Name of the unit test case.
    * @param [in] function  Pointer to unit test case function.
    * @returns True if all tests pass, false otherwise.
    */
    static void RegisterFunction(const std::string &name,
      UnitFunction function) {
      Instance().unitfunction_map_.emplace(name, function);
    }

    /**
    * @brief Helper class to register a unit test in construction time. This is
    *        used to call RegisterFunction in the construction of a static
    *        helper object. Used by the REGISTER_UNIT macro, which in turn is
    *        used by the UNIT macro.
    * @returns True if all tests pass, false otherwise.
    */
    class Registrator {
    public:
      Registrator(const std::string &name,
        UnitFunction function) {
        UnitTester::RegisterFunction(name, function);
      };
      Registrator(const Registrator&) = delete;
      Registrator(Registrator&&) = delete;
      ~Registrator() {};
    };

    ~UnitTester() {};
    UnitTester(const UnitTester&) = delete;
    UnitTester(UnitTester&&) = delete;

  private:
    UnitTester() {};
    static UnitTester& Instance() {
      static UnitTester instance;
      return instance;
    }
    std::map<std::string, UnitFunction> unitfunction_map_;
  };
}

#define MACROCAT_NEXP(A, B) A ## B
#define MACROCAT(A, B) MACROCAT_NEXP(A, B)

/**
* @brief Register a unit function using a helper static Registrator object.
*/
#define REGISTER_UNIT(FUNCTION)                                                \
  static microunit::UnitTester::Registrator                                    \
  MACROCAT(MICROUNIT_REGISTRATION, __COUNTER__)(#FUNCTION, FUNCTION);

/**
* @brief Define a unit function body. This macro is the one which should be used
*        by client code to define unit test cases.
* @code{.cpp}
*  UNIT(Test_Two_Plus_Two) {
*    ASSERT_TRUE(2 + 2 == 4);
*  };
* @endcode
*/
#define UNIT(FUNCTION)                                                         \
void FUNCTION(microunit::UnitFunctionResult*);                                 \
REGISTER_UNIT(FUNCTION);                                                       \
void FUNCTION(microunit::UnitFunctionResult *__microunit_testresult)

/**
* @brief Pass the test and return from the test case.
*/
#define PASS() {                                                               \
LOG_GOOD << "Test stopped: Pass";                                              \
__microunit_testresult->success = true;                                        \
return;                                                                        \
}

/**
* @brief Fail the test and return from the test case.
*/
#define FAIL() {                                                               \
LOG_BAD << "Test stopped: Fail";                                               \
__microunit_testresult->success = false;                                       \
return;                                                                        \
}

/**
* @brief Check a particular test condition. If the condition does not hold,
*        fail the test and return.
*/
#define ASSERT_TRUE(condition) if(!(condition)) {                              \
LOG_BAD << "Assert-true failed: " #condition;                                  \
FAIL();                                                                        \
}

/**
* @brief Check a particular test condition. If the condition holds, fail the
*        test and return.
*/
#define ASSERT_FALSE(condition) if((condition)) {                              \
LOG_BAD << "Assert-false failed: " #condition << std::endl;                    \
FAIL();                                                                        \
}
#endif