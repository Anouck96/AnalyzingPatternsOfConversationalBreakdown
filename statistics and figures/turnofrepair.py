import statistics
import pandas as pd
from collections import Counter

def main():
    # Simple calculations for conversation length (includes agent utterances, and actions such as a user leaving the conversation)
    file_paths = ['removed for anonymity']

    amount_of_turns = []

    for file_path in file_paths:
        # Read files
        feat_df = pd.read_excel(file_path)
        # Group by Session ID
        grouped = feat_df.groupby('Session')
        # Loop over each group
        for session_id, group in grouped:
            # Check for welcome as starting point of conversation
            default_welcome_rows = group[group['Msg Text'].str.contains('removed for anonymity', na=False)]
            # Check for repair row
            row_num_sorry = group[group['Msg Attachment'] == 'removed for anonymity'].index.tolist()
            row_num_helaas = group[group['Msg Attachment'] == 'removed for anonymity'].index.tolist()

            if not default_welcome_rows.empty:
                row_welcome = default_welcome_rows.index.tolist()
            if row_num_sorry:
                for item in row_num_sorry:
                    turn = item - row_welcome[0]
                amount_of_turns.append(turn)
            if row_num_helaas:
                for item in row_num_helaas:
                    turn = item - row_welcome[0]
                amount_of_turns.append(turn)

    # calculate mean, mode and median over the turns
    mean_turns = sum(amount_of_turns) / len(amount_of_turns)
    print(mean_turns)
    mode_turns = statistics.mode(amount_of_turns) if amount_of_turns else 0
    median_turns = statistics.median(amount_of_turns) if amount_of_turns else 0
    print(mode_turns)
    print(median_turns)

if __name__ == '__main__':
    main()
