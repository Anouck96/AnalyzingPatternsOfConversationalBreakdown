import pandas as pd
from collections import Counter

def main():
    # Simple calculations for conversation length (includes agent utterances)
    file_paths = ['removed for anonynimity']

    all_session_ids = []

    for file_path in file_paths:
        # Read files
        feat_df = pd.read_excel(file_path)

        # Filter rows with user joining/leaving actions
        feat_df = feat_df[~feat_df['Action'].isin(['removed for anonynimity'])]

        # Collect sessionIds
        session_ids = feat_df['Session'].tolist()

        # Append to one list
        all_session_ids.extend(session_ids)

    # count occurences
    session_counts = Counter(all_session_ids)
     # calculate average
    average_count = sum(session_counts.values()) / len(session_counts)

    print(average_count)


if __name__ == '__main__':
    main()
