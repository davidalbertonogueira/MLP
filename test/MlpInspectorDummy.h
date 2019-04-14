/*
 * MlpInspectorDummy.h
 *
 * rluna
 * Apr 8, 2019
 *
 */

#ifndef TEST_MLPINSPECTORDUMMY_H_
#define TEST_MLPINSPECTORDUMMY_H_

#include "easylogging++.h"

#include "MlpInspector.h"


class MlpInspectorDummy : public MlpInspector
{
public:
    MlpInspectorDummy();
    void onEnterTraining( std::vector<Layer> layers );
    void onBeforeTrainingSample( std::vector<Layer> layers );
    void onAfterTrainingSample( std::vector<Layer> layers );
    void onEndTraining( std::vector<Layer> layers );

    long getOnEnter();
    long getOnBefore();
    long getOnAfter();
    long getOnEnd();
    ~MlpInspectorDummy();

private:
    long callsOnEnter = 0;
    long callsOnEnd = 0;
    long callsOnBeforeTraining = 0;
    long callsOnAfterTraining = 0;
};

#endif /* TEST_MLPINSPECTORDUMMY_H_ */



