# Elo Merchant Category Recommendation Kaggle Competition

## How to train on Colab

```python
import pandas as pd
import numpy as np
import sys
from google.colab import drive
drive.mount('/content/gdrive')

!rm -rf elo-competition
!git clone https://github.com/tianhaoz95/elo-competition.git

sys.path.append('./elo-competition')

import util

dataset_root = 'gdrive/My Drive/Development/elo-competition/workspace/dataset'
```
