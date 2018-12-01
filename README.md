# Elo Merchant Category Recommendation Kaggle Competition

## How to train on Colab

```python
import sys
from google.colab import drive
drive.mount('/content/gdrive')

!rm -rf elo-competition
!git clone https://github.com/tianhaoz95/elo-competition.git

sys.path.append('./elo-competition')

import util

dataset_root = 'gdrive/My Drive/Development/elo-competition/workspace/dataset'
dataset = util.common_routine(dataset_root, 10000, 10000, 100)
```

## Roadmap
- [x] Infrastructure and sanity check model with only `feature_1, feature_1, feature_1`
- [ ] Make use of the `category_1 category_2 category_3` in `new_merchant_transactions.csv`
- [ ] Make use of the `state_id` in `new_merchant_transactions.csv`
- [ ] Make use of the `subsector_id` in `new_merchant_transactions.csv`