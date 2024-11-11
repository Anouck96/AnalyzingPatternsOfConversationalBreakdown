import pandas as pd

def counts(dataset):
    data = pd.read_excel(dataset)
    sess = (list(data['Session']))
    cn = len(list(dict.fromkeys(sess)))
    return cn

def main():
    logs = ['removed for anonymity']
    nms = []
    for log in logs:
        print(log, counts(log))
        nms.append(counts(log))

    print(sum(nms))
if __name__=='__main__':
    main()
