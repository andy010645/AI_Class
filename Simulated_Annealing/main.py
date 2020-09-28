from functions import func
f=open('input.txt','r')
lines=f.readlines()
#lines=lines.replace(","," ")
lines[0]=lines[0].replace(","," ")
lines[0]=lines[0].replace("\0"," ")
print(int(lines[0]))

def Brute_force():
    max=10000
    for a in range():
        for b in range(0,1):
            print(b)
            if (func(a,b)<max):
                max=func(a,b)
                ans_a=a
                ans_b=b
        print(a)
    
    return ans_a,ans_b,round(func(ans_a,ans_b),4)
#def SA():    
print(Brute_force())