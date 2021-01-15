import numpy as np
from keras.models import load_model
import pandas as pd
import csv
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import OneHotEncoder

def get_similar_products(user_id, products, g, number):
    user_profile = np.dot(products[g[g.user_id == user_id]['product_id'].values-1].T, 
                          g[g.user_id == user_id]['product_id_count'].values)
    C = cosine_similarity(np.atleast_2d(user_profile), products)
    R = np.argsort(C)[:, ::-1]
    return R[0][:number]

if __name__ == '__main__':

    print('collecting data')
    model = load_model('ncf.h5')
    users = pd.read_csv('users_meta.csv')
    products = pd.read_csv('products.csv')
    transactions_df =  pd.read_csv('/content/drive/My Drive/transactions.csv')

    print('product encoding')
    one_hot_encoder = OneHotEncoder(sparse=False)
    aisle_enc = one_hot_encoder.fit_transform(products.aisle_id.values.reshape(-1,1))
    department_enc = one_hot_encoder.fit_transform(products.department_id.values.reshape(-1,1))
    prod_enc = np.hstack([aisle_enc, department_enc])

    user_product = transactions_df.groupby(['user_id', 'product_id'])['product_id'].count()
    user = transactions_df.groupby(['user_id'])['product_id'].count()

    print('compute produts ratings')
    g = pd.DataFrame(user_product.div(user, level='user_id')).add_suffix('_count').reset_index()

    recs = []
    users_count = len(users)

    for ind, a in users.iterrows():
        user_num = a['user_id_num']

        #take 100 similar products to user, content-base filtration
        s_prods = get_similar_products(a['user_id'], prod_enc, g, 100)
        s_prods = list(map(lambda x: x+1, s_prods))

        #predict by NCF
        ratings = model.predict([np.array([user_num]*len(s_prods)), np.array(s_prods)])
        prod_rating = list(zip(s_prods, ratings.flatten()))
        prod_rating.sort(key = lambda x: x[1], reverse = True)
        prod_rating = [a['user_id']] + [row[0] for row in prod_rating]

        #write to file
        print('{0} of {1} users proceeded'.format(ind+1, users_count))
        with open('submission.csv', 'a') as fh:
            print(str(prod_rating[0]) + ',' +' '.join(str(x) for x in prod_rating[1:11]), file=fh)

    print('done')