import pandas as pd
from collections import Counter

def counts(dataset):
    sess = (list(dataset['Session']))
    cn = len(list(dict.fromkeys(sess)))
    return cn


def main():
    # Simple calculations for conversation length (includes agent utterances, and actions such as user-left)
    file_paths = ['removed for anonymity']

    all_session_ids = []
    nms = []
    for file_path in file_paths:
        # Read files
        feat_df = pd.read_excel(file_path)

        # Filter rows with join/rejoin/left actions
        feat_df = feat_df[~feat_df['Action'].isin(['removed for anonymity'])]

        # Collect sessionIds
        session_ids = feat_df['Session'].tolist()

        # Append to one list
        all_session_ids.extend(session_ids)

        # Counts amount of conversations (without conversations with only technical actions removed)
        print(file_path, counts(feat_df))
        nms.append(counts(feat_df))

    # count occurences
    session_counts = Counter(all_session_ids)
     # calculate average
    average_count = sum(session_counts.values()) / len(session_counts)

    print(average_count)

    print(sum(nms))


if __name__ == '__main__':
    main()
