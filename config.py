dataset_meta = {
    'historical_transactions': {
        'filename': 'historical_transactions.csv'
    },
    'merchants': {
        'filename': 'merchants.csv'
    },
    'new_merchant_transactions': {
        'filename': 'new_merchant_transactions.csv'
    },
    'sample_submission': {
        'filename': 'sample_submission.csv'
    },
    'test': {
        'filename': 'test.csv'
    },
    'train': {
        'filename': 'train.csv'
    }
}

index_features = [
    'feature_1',
    'feature_2',
    'feature_3',
    'target',
    'card_id',
    'first_active_month'
]

id_features = [
    'city_id',
    'category_1',
    'category_2',
    'category_3',
    'installments'
]

date_format = '%Y-%m'