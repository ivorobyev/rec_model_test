import argparse
import numpy as np
from keras.models import load_model
import pandas as pd

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('-u', '--user_id', help='user_id', type = int)
    parser.add_argument('-p', '--products', help='tnumber of products to recommend', type=int, default=10)
    args = parser.parse_args()

    model = load_model('ncf.h5')
    users = pd.read_csv('users_meta.csv')
    products = pd.read_csv('products_meta.csv')
    user_num = users['user_id_num'].loc[users['user_id'] == args.user_id].values[0]
    products_num = np.array(products.product_id_num.to_list())
    users_num = np.array([user_num] * len(products_num))
    ratings = model.predict([users_num, products_num])
    products['rating'] = ratings.flatten()
    print(products.sort_values(by = 'rating'))
