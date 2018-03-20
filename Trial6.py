x=12765432345678909876543123234567;

s="12765432345678909876543123234567";

def gcd(a,b):
    if b==0:
        return a
    return gcd(b,a%b)

ans=0

for i in range(len(s)-1):
    x1=0
    x2=0
    for j in range(i+1):
        x1*=10
        x1+=int(s[j])

    for j in range(len(s)-i-1):
        x2*=10
        x2+=int(s[i+j+1])
    print(x1, x2, gcd(x1, x2))
    if ans<gcd(x1,x2):
        ans=gcd(x1,x2);
print(ans)