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
dataset = util.common_routine(dataset_root, 10000, 10000, 100)
```
## Phases
* Have a complete feature set for training. No new feature will be needed after this phase theoretically.
* Model optimization where a couple of models are manually crafted and tested.
* Automatic model exploration. Read papers on auto model space exploration, source code from auto keras, etc

## Roadmap
- [x] Infrastructure and sanity check model with only `feature_1, feature_1, feature_1`
- [x] Normalize `target`
- [x] Add `first_activate_month` into features used
- [ ] Make use of the `category_1 category_2 category_3` in `new_merchant_transactions.csv`
- [ ] Make use of the `state_id` in `new_merchant_transactions.csv`
- [ ] Make use of the `subsector_id` in `new_merchant_transactions.csv`
