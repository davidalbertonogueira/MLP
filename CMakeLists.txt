cmake_minimum_required(VERSION 3.7)
project(MLP VERSION 1.0)

######################################
############### OPTIONS ##############
######################################
option(BUILD_SHARED_LIB "Building shared library" ON)
option(BUILD_STATIC_LIB "Building static library" ON)
option(MLP_EASYLOGGING "Enable logging" ON)

if (MLP_EASYLOGGING)
	add_compile_definitions("MLP_EASYLOGGING")
endif()

set(MLP_SHARED_LIB_NAME ${PROJECT_NAME})
set(MLP_STATIC_LIB_NAME ${PROJECT_NAME}_static)
set(MLP_SOURCE_FILES "${CMAKE_CURRENT_SOURCE_DIR}/src/MLP.cpp")
set(MLP_HEADER_FILES "${CMAKE_CURRENT_SOURCE_DIR}/src/MLP.h"
					 "${CMAKE_CURRENT_SOURCE_DIR}/src/Layer.h"
					 "${CMAKE_CURRENT_SOURCE_DIR}/src/Node.h"
					 "${CMAKE_CURRENT_SOURCE_DIR}/src/Sample.h"
					 "${CMAKE_CURRENT_SOURCE_DIR}/src/Utils.h")

######################################
######### BUILDING LIBRARIES #########
######################################
#### STATIC LIBRARY
if (BUILD_STATIC_LIB)
	add_library(${MLP_STATIC_LIB_NAME} STATIC ${MLP_HEADER_FILES} ${MLP_SOURCE_FILES})
	set_target_properties(${MLP_STATIC_LIB_NAME} PROPERTIES VERSION ${PROJECT_VERSION})
	set_target_properties(${MLP_STATIC_LIB_NAME} PROPERTIES SOVERSION 1)
	set_target_properties(${MLP_STATIC_LIB_NAME} PROPERTIES PUBLIC_HEADER ${MLP_HEADER_FILES})
	target_include_directories(${MLP_STATIC_LIB_NAME} PRIVATE ".")
	target_include_directories(${MLP_STATIC_LIB_NAME} PRIVATE "src")
	target_include_directories(${MLP_STATIC_LIB_NAME} PRIVATE "deps")
endif()

#### SHARED LIBRARY
if (BUILD_SHARED_LIB)
	add_library(${MLP_SHARED_LIB_NAME} SHARED ${MLP_HEADER_FILES} ${MLP_SOURCE_FILES})
	set_target_properties(${MLP_SHARED_LIB_NAME} PROPERTIES VERSION ${PROJECT_VERSION})
	set_target_properties(${MLP_SHARED_LIB_NAME} PROPERTIES SOVERSION 1)
	set_target_properties(${MLP_SHARED_LIB_NAME} PROPERTIES PUBLIC_HEADER ${MLP_HEADER_FILES})
	target_include_directories(${MLP_SHARED_LIB_NAME} PRIVATE ".")
	target_include_directories(${MLP_SHARED_LIB_NAME} PRIVATE "src")
	target_include_directories(${MLP_SHARED_LIB_NAME} PRIVATE "deps")

	if(UNIX) ## Install shared lib
		include(GNUInstallDirs)
		install(TARGETS ${MLP_SHARED_LIB_NAME}
			LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
			PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
	endif(UNIX)
endif()