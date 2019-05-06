import argparse
import json
import sys
import time
from collections import defaultdict

import nltk

sys.path.append('./')

nltk.download('punkt')


def main(args):
    rel2id = json.loads(open('./preproc/{}_relation2id.json'.format(args.data)).read())

    print('Constructing training bags...')
    print('Counting number of training bags...')
    data_counts = defaultdict(int)
    with open('./data/{}_train.json'.format(args.data)) as f:
        for i, line in enumerate(f):
            data = json.loads(line.strip())

            _id = f'{data["sub"]}_{data["obj"]}'
            data_counts[_id] += 1

            if (i + 1) % args.log_steps == 0:
                print('Processed {}, {}'.format(i + 1, time.strftime("%d_%m_%Y %H:%M:%S")))
            if not args.FULL and i > args.sample_size:
                break

    print('Combining and saving training bags...')
    train_data = defaultdict(lambda: {'rels': defaultdict(list)})
    data_counts_proc = defaultdict(int)
    with open('./data/{}_train.json'.format(args.data)) as f,\
            open('./data/{}_train_bags.json'.format(args.data), 'w') as fout:
        for i, line in enumerate(f):
            data = json.loads(line.strip())

            _id = f'{data["sub"]}_{data["obj"]}'
            data_counts_proc[_id] += 1

            train_data[_id]['sub_id'] = data['sub_id']
            train_data[_id]['obj_id'] = data['obj_id']
            train_data[_id]['sub'] = data['sub']
            train_data[_id]['obj'] = data['obj']

            train_data[_id]['rels'][rel2id.get(data['rel'], rel2id['NA'])].append({
                'corenlp': data['corenlp'],
                'rsent': data['rsent'],
                'openie': data['openie'],
            })

            if data_counts_proc[_id] == data_counts[_id]:
                data = train_data[_id]
                for rel, sents in data['rels'].items():
                    entry = {
                        'sub': data['sub'],
                        'obj': data['obj'],
                        'sub_id': data['sub_id'],
                        'obj_id': data['obj_id'],
                        'sents': sents,
                        'rel': [rel],
                    }
                    fout.write(json.dumps(entry) + '\n')
                del train_data[_id]

            if (i + 1) % args.log_steps == 0:
                print('Completed {}, {}'.format(i + 1, time.strftime("%d_%m_%Y %H:%M:%S")))
            if not args.FULL and i > args.sample_size:
                break
    del train_data, data_counts, data_counts_proc

    print('Constructing test bags...')
    print('Counting number of test bags...')
    data_counts = defaultdict(int)
    with open('./data/{}_test.json'.format(args.data)) as f:
        for i, line in enumerate(f):
            data = json.loads(line.strip())

            _id = f'{data["sub"]}_{data["obj"]}'
            data_counts[_id] += 1

            if (i + 1) % args.log_steps == 0:
                print('Processed {}, {}'.format(i + 1, time.strftime("%d_%m_%Y %H:%M:%S")))
            if not args.FULL and i > args.sample_size:
                break

    print('Combining and saving test bags...')
    test_data = defaultdict(lambda: {'rels': defaultdict(list)})
    data_counts_proc = defaultdict(int)
    with open('./data/{}_test.json'.format(args.data)) as f, \
            open('./data/{}_test_bags.json'.format(args.data), 'w') as fout:
        for i, line in enumerate(f):
            data = json.loads(line.strip())

            _id = f'{data["sub"]}_{data["obj"]}'
            data_counts_proc[_id] += 1

            test_data[_id]['sub_id'] = data['sub_id']
            test_data[_id]['obj_id'] = data['obj_id']
            test_data[_id]['sub'] = data['sub']
            test_data[_id]['obj'] = data['obj']

            test_data[_id]['rels'][rel2id.get(data['rel'], rel2id['NA'])].append({
                'corenlp': data['corenlp'],
                'rsent': data['rsent'],
                'openie': data['openie'],
            })

            if data_counts_proc[_id] == data_counts[_id]:
                data = test_data[_id]
                for rel, sents in data['rels'].items():
                    entry = {
                        'sub': data['sub'],
                        'obj': data['obj'],
                        'sub_id': data['sub_id'],
                        'obj_id': data['obj_id'],
                        'sents': sents,
                        'rel': [rel],
                    }
                    fout.write(json.dumps(entry) + '\n')
                del test_data[_id]

            if (i + 1) % args.log_steps == 0:
                print('Completed {}, {}'.format(i + 1, time.strftime("%d_%m_%Y %H:%M:%S")))
            if not args.FULL and i > args.sample_size:
                break
    del test_data, data_counts, data_counts_proc


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--data', default='riedel')
    parser.add_argument('--log-steps', default=20000, type=int,
                        help='Logging frequency in steps')
    parser.add_argument('--sample', dest='FULL', action='store_false',
                        help='To process the entire data or a sample of it')
    parser.add_argument('--sample-size', dest='sample_size', default=500, type=int,
                        help='Sample size to use for testing processing script')
    args = parser.parse_args()

    main(args)
