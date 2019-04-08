/*
 * MlpInspector.h
 *
 * rluna
 * Mar 22, 2019
 *
 */

#ifndef SRC_MLPINSPECTOR_H_
#define SRC_MLPINSPECTOR_H_

#include "Layer.h"

/**
 * You can create a class based on this interface to
 * inspect the internal works of the MLP network.
 *
 *
 */
class MlpInspector
{
public:
    virtual void afterCreateNetwork( std::vector<Layer> layers );
    virtual void beforeDestroyNetwork( std::vector<Layer> layers );
    virtual void onEnterTraining( std::vector<Layer> layers );
    virtual void onEndTraining( std::vector<Layer> layers );

    virtual ~MlpInspector() {};
};

#endif /* SRC_MLPINSPECTOR_H_ */


