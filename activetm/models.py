"""Factory for building models"""
from .tech import anchor
from .tech.sampler import slda


FACTORY = {
    'slda': slda.SamplingSLDA,
    'ridge_anchor': anchor.RidgeAnchor,
    'gp_anchor': anchor.GPAnchor,
    'semi_ridge_anchor': anchor.SemiRidgeAnchor,
    'semi_gp_anchor': anchor.SemiGPAnchor
}


def build(rng, settings):
    """Build a model according to the settings"""
    if settings['model'] == 'slda':
        slda.set_seed(int(settings['cseed']))
        num_topics = int(settings['numtopics'])
        alpha = float(settings['alpha'])
        beta = float(settings['beta'])
        var = float(settings['var'])
        num_train = int(settings['numtrain'])
        num_samples_train = int(settings['numsamplestrain'])
        train_burn = int(settings['trainburn'])
        train_lag = int(settings['trainlag'])
        num_samples_predict = int(settings['numsamplespredict'])
        predict_burn = int(settings['predictburn'])
        predict_lag = int(settings['predictlag'])
        return FACTORY['slda'](rng,
                               num_topics,
                               alpha,
                               beta,
                               var,
                               num_train,
                               num_samples_train,
                               train_burn,
                               train_lag,
                               num_samples_predict,
                               predict_burn,
                               predict_lag)
    elif 'anchor' in settings['model'] and settings['model'] in FACTORY:
        num_topics = int(settings['numtopics'])
        num_train = int(settings['numtrain'])
        return FACTORY[settings['model']](rng, num_topics, num_train)
    else:
        raise Exception('Unknown model "'+settings['model']+'"; aborting')

