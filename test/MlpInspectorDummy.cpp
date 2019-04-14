/*
 * MlpInspectorDummy.cpp
 *
 * rluna
 * Apr 8, 2019
 *
 */

#include "MlpInspectorDummy.h"

MlpInspectorDummy::MlpInspectorDummy()
{

}

MlpInspectorDummy::~MlpInspectorDummy()
{

}

void MlpInspectorDummy::onEnterTraining( std::vector<Layer> layers ){
    ++callsOnEnter;
}

void MlpInspectorDummy::onBeforeTrainingSample( std::vector<Layer> layers )
{
    ++callsOnBeforeTraining;
}

void MlpInspectorDummy::onAfterTrainingSample( std::vector<Layer> layers )
{
    ++callsOnAfterTraining;
}


void MlpInspectorDummy::onEndTraining( std::vector<Layer> layers ){
    ++callsOnEnd;
}

long MlpInspectorDummy::getOnEnter(){
    return callsOnEnter;
}

long MlpInspectorDummy::getOnBefore(){
    return callsOnBeforeTraining;
}

long MlpInspectorDummy::getOnAfter(){
    return callsOnAfterTraining;
}

long MlpInspectorDummy::getOnEnd(){
    return callsOnEnd;
}



