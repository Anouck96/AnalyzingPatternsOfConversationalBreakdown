import pandas as pd

def main():

    file_paths = ['removed for anonymity']
    session_ids = set()

    # Get sessions for rows with message (create set so duplicates are removed)
    for fi in file_paths:
        data = pd.read_excel(fi)
        filtered_rows = data[data['Msg Attachment'] == 'removed for anonymity']
        session_ids.update(filtered_rows['Session'])
    print(len(session_ids))


if __name__ == '__main__':
    main()
