import pandas as pd
import numpy as np
df = pd.read_excel('SampleInput.xlsx')
df.fillna(0,inplace=True)

del df['TicketNumber']
del df['Client']
del df['Staff']
del df['Source']
del df['DataOn']
del df['DueDate']
del df['ClosedDate']
del df['ModifiedDate']
del df['KYC']
to_be_removed = []
for i in range(df.shape[0]):
    x = df.iloc[i][0]
    if type(x)==int:
        to_be_removed.append(i)
    elif 'Status' in x:
        to_be_removed.append(i)
    elif 'Ticket' in x:
        to_be_removed.append(i)
    elif 'Collaborators' in x:
        to_be_removed.append(i)
    elif 'Collaborator' in x:
        to_be_removed.append(i)
df.drop(df.index[to_be_removed],inplace=True)

df['approved'] = 0
df.to_csv('finalInput.csv')

