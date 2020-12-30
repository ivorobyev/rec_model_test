import argparse

if __name__ == '__main__':
    start_time: float = time.time()

    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('-u', '--user_id', help='user_id')
    group.add_argument('-p', '--products', help='number of products to recommend')
    args = parser.parse_args()