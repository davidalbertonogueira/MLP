cmake_minimum_required(VERSION 3.7)
project(MLP VERSION 1.0)

set(SHARED_LIB_NAME ${PROJECT_NAME})
set(STATIC_LIB_NAME ${PROJECT_NAME}_static)
set(MLP_SOURCE_FILES "${CMAKE_CURRENT_SOURCE_DIR}/src/MLP.cpp")
set(MLP_HEADER_FILES "${CMAKE_CURRENT_SOURCE_DIR}/src/MLP.h"
					"${CMAKE_CURRENT_SOURCE_DIR}/src/Layer.h"
					"${CMAKE_CURRENT_SOURCE_DIR}/src/Node.h"
					"${CMAKE_CURRENT_SOURCE_DIR}/src/Sample.h"
					"${CMAKE_CURRENT_SOURCE_DIR}/src/Utils.h")


#### STATIC LIBRARY
add_library(${STATIC_LIB_NAME} STATIC ${MLP_HEADER_FILES} ${MLP_SOURCE_FILES})
set_target_properties(${STATIC_LIB_NAME} PROPERTIES VERSION ${PROJECT_VERSION})
set_target_properties(${STATIC_LIB_NAME} PROPERTIES SOVERSION 1)
set_target_properties(${STATIC_LIB_NAME} PROPERTIES PUBLIC_HEADER ${MLP_HEADER_FILES})
target_include_directories(${STATIC_LIB_NAME} PRIVATE ".")
target_include_directories(${STATIC_LIB_NAME} PRIVATE "src")
target_include_directories(${STATIC_LIB_NAME} PRIVATE "deps")


#### SHARED LIBRARY
add_library(${SHARED_LIB_NAME} SHARED ${MLP_HEADER_FILES} ${MLP_SOURCE_FILES})
set_target_properties(${SHARED_LIB_NAME} PROPERTIES VERSION ${PROJECT_VERSION})
set_target_properties(${SHARED_LIB_NAME} PROPERTIES SOVERSION 1)
set_target_properties(${SHARED_LIB_NAME} PROPERTIES PUBLIC_HEADER ${MLP_HEADER_FILES})
target_include_directories(${SHARED_LIB_NAME} PRIVATE ".")
target_include_directories(${SHARED_LIB_NAME} PRIVATE "src")
target_include_directories(${SHARED_LIB_NAME} PRIVATE "deps")

if(UNIX) ## Install shared lib
	include(GNUInstallDirs)
	install(TARGETS ${SHARED_LIB_NAME}
		LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
		PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
endif(UNIX)
