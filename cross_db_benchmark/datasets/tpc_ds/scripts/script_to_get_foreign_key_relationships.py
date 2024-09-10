filename='foreign_key_relationships.txt'
with open(filename) as f:
    lines=f.readlines()
l=[]
for line in lines:
    table1=line.split('alter table')[1].split('add')[0].strip()
    column1=line.split('foreign key')[1].split('references')[0].strip()[1:-1]
    table2=line.split('references')[1].split('(')[0].strip()
    column2=line.split('references')[1].split('(')[1].strip()[:-2]
    l.append([table1, column1, table2, column2])
print(l)