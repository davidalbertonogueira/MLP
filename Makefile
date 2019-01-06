#!/bin/bash
# Makefile for MLP
CC = g++
DEBUG = -g3
PROJNAME = mlp

HEADERPATH = ./src
SOURCEPATH = ./src

LOCALDEPSINCLUDES = ./deps
AUXINCLUDES = 
AUXLIBS = 

INCLUDES = -I$(LOCALDEPSINCLUDES) -I$(AUXINCLUDES)  
LIBS = -L$(AUXLIBS) 
#LIBS += -L/usr/local/lib/
#rlunaro: removed optimization for tests: -O3
CFLAGS = -std=gnu++11 -std=c++11 -Wall -fmessage-length=0 -fPIC $(INCLUDES)
CFLAGS += $(DEBUG)
LFLAGS = $(LIBS)
#For verbosity
LFLAGS += -v
LDFLAGS = -shared

HDRS  = $(shell find $(HEADERPATH) $(AUXINCLUDES) $(LOCALDEPSINCLUDES) -name '*.h')
HDRS += $(shell find $(HEADERPATH) $(AUXINCLUDES) $(LOCALDEPSINCLUDES) -name '*.h++')
SRCS  = $(shell find $(SOURCEPATH) -name '*.cpp')
SRCS += $(shell find $(SOURCEPATH) -name '*.c')
OBJS = $(SRCS:.cpp=.o)
TXTS = $(wildcard *.txt)
SCRIPTS = $(wildcard *.sh)

all : IrisDatasetTest MLPTest LayerTest NodeTest $(PROJNAME).a $(PROJNAME).so

$(PROJNAME).a : $(SOURCEPATH)/MLP.o
	@echo Creating static lib $@
	ar rcs $@ $(SOURCEPATH)/MLP.o

$(PROJNAME).so : $(SOURCEPATH)/MLP.o
	@echo Creating dynamic lib $@
	$(CC) -o $@ $(SOURCEPATH)/MLP.o $(LDFLAGS) $(LFLAGS) 

%.o: %.cpp $(HDRS)
	$(CC) -c $(CFLAGS) $(LFLAGS) -o $@ $<

IrisDatasetTest: $(SOURCEPATH)/IrisDatasetTest.o  $(SOURCEPATH)/MLP.o
	@echo Compiling program $@
	$(CC)  $^ $(CFLAGS) $(LFLAGS) -o $@

MLPTest: $(SOURCEPATH)/MLPTest.o  $(SOURCEPATH)/MLP.o
	@echo Compiling program $@
	$(CC)  $^ $(CFLAGS) $(LFLAGS) -o $@
	
LayerTest: $(SOURCEPATH)/LayerTest.o  $(SOURCEPATH)/MLP.o
	@echo Compiling program $@
	$(CC)  $^ $(CFLAGS) $(LFLAGS) -o $@
	
NodeTest: $(SOURCEPATH)/NodeTest.o  $(SOURCEPATH)/MLP.o
	@echo Compiling program $@
	$(CC)  $^ $(CFLAGS) $(LFLAGS) -o $@
clean:
	@echo Clean
	rm -f *~ $(SOURCEPATH)/*.o *~
	@echo Success

cleanall:
	@echo Clean All
	rm -f *~ $(SOURCEPATH)/*.o *~ $(PROJNAME).a $(PROJNAME).so IrisDatasetTest MLPTest LayerTest NodeTest
	@echo Success