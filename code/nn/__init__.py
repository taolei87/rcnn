
'''
    Import classes and methods from other .py files
'''

from utils import say
#from .initialization import default_srng, default_rng, USE_XAVIER_INIT
#from .initialization import set_default_rng_seed, random_init, create_shared
#from .initialization import ReLU, sigmoid, tanh, softmax, linear, get_activation_by_name
#from .advanced import StrCNN, AttentionLayer, BilinearAttentionLayer

from .initialization import *
from .basic import *
from .advanced import *
from .optimization import create_optimization_updates

