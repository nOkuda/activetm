from __future__ import division
import tech.sampler.slda

factory = {
    'slda': tech.sampler.slda.SamplingSLDA
}

def build(rng, settings):
    if settings['model'] == 'slda':
        tech.sampler.slda.set_seed(int(settings['cseed']))
        NUM_TOPICS = int(settings['numtopics'])
        ALPHA = float(settings['alpha'])
        BETA = float(settings['beta'])
        VAR = float(settings['var'])
        NUM_TRAIN = int(settings['numtrain'])
        NUM_SAMPLES_TRAIN = int(settings['numsamplestrain'])
        TRAIN_BURN = int(settings['trainburn'])
        TRAIN_LAG = int(settings['trainlag'])
        NUM_SAMPLES_PREDICT = int(settings['numsamplespredict'])
        PREDICT_BURN = int(settings['predictburn'])
        PREDICT_LAG = int(settings['predictlag'])
        return factory['slda'](rng, NUM_TOPICS, ALPHA, BETA, VAR,
                NUM_TRAIN, NUM_SAMPLES_TRAIN, TRAIN_BURN, TRAIN_LAG,
                NUM_SAMPLES_PREDICT, PREDICT_BURN, PREDICT_LAG)
    else:
        raise Exception('Unknown model "'+settings['model']+'"; aborting')

