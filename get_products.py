import argparse
import numpy as np
from numpy import savetxt
from keras.models import load_model
import pandas as pd
import csv

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--products', help='number of products to recommend', type=int, default=10)
    args = parser.parse_args()

    model = load_model('ncf.h5')
    users = pd.read_csv('users_meta.csv')
    products = pd.read_csv('products_meta.csv')
    products_num = np.array(products.product_id_num.to_list())

    recs = []
    users_count = len(users)
    for ind, a in users.iterrows():
        user_num = a['user_id_num']
        ratings = model.predict([np.array([user_num]*len(products_num)), 
                                 products_num])
        products['rating'] = ratings.flatten()
        user_products = products.sort_values(by = 'rating', ascending=False).head(10)
        user_products = user_products['product_id'].to_list()
        user_products = [a['user_id']] + user_products
        recs.append(user_products)
        print('{0} of {1} users proceeded'.format(ind+1, users_count))
        with open('submission.csv', 'a') as fh:
            print(' '.join(str(x) for x in user_products), file=fh)

    print('done')