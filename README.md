# Elo Merchant Category Recommendation Kaggle Competition

## How to train on Colab

```python
import sys
from google.colab import drive
drive.mount('/content/gdrive')
%matplotlib inline

!rm -rf elo-competition
!git clone https://github.com/tianhaoz95/elo-competition.git

sys.path.append('./elo-competition')

import util

dataset_root = 'gdrive/My Drive/Development/elo-competition/workspace/dataset'
util.common_routine(dataset_root, 500, 500, 50, viz=True, output_dir=dataset_root, train=True, test=True, load_trained_model=False)
```

## Phases
* Have a complete feature set for training. No new feature will be needed after this phase theoretically.
* Model optimization where a couple of models are manually crafted and tested.
* Automatic model exploration. Read papers on auto model space exploration, source code from auto keras, etc
