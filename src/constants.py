from pathlib import Path

#### Paths ####
# Update this to wherever you want to store data
DATADIR = Path("/home/alex/data/mapai") 

# INRIA
INRIA_DATA = DATADIR/'external'/'inria_aerialimagelabeling'
INRIA_PATCHES = INRIA_DATA/'AerialImageDataset'/'train'/'png'
INRIA_PREDS = INRIA_PATCHES/'predictions'

# MapAI
MAPAI_TRAIN = DATADIR/'data'/'train'
MAPAI_MASKS = MAPAI_TRAIN/'masks'
MAPAI_PATCHES = MAPAI_TRAIN/'patches'

MAPAI_VAL = DATADIR/'data'/'validation'
MAPAI_VAL_MASKS = MAPAI_VAL/'masks'
MAPAI_VAL_PREDS = MAPAI_VAL/'predictions'