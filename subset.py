x=[1,2,3]
le=len(x)
for i in range(1,(1<<le)):
    for j in range(le):
        if i&(1<<j):
            print(x[j],end=' ')
    print()