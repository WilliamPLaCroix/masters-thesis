import pandas as pd

def main():

    try:
        sample = pd.read_csv('./data/wikilarge/wiki_train_labelled.csv')
        print('Resuming labelling process from checkpoint...')
    except FileNotFoundError:
        print('Starting new labelling process...')
        data = pd.read_csv('./data/wikilarge/wiki_train.csv')
        sample = data.sample(1000, random_state=42).reset_index(drop=True)
        sample['label'] = -1
        sample.to_csv('./data/wikilarge/wiki_train_labelled.csv', index=False)

    for i, row in sample.iterrows():
        print(i)
        print('src:', row['src'])
        print('tgt:', row['tgt'])
        if row['label'] in {0,1}:
            print('label:', row['label'])
            print()
            continue
        while True:
            try:
                label: int = int(input('Label: '))
            except ValueError:
                print('Invalid input. Please enter 0 or 1.')
                continue
            sample['label'][i] = label
            if label in {0, 1}:
                sample.to_csv('./data/wikilarge/wiki_train_labelled.csv', index=False)
                break
        print()

if __name__ == '__main__':
    main()
