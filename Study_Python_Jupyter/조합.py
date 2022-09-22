
sample = ["a", "b", "c"]

# def YesDuple(list):

element = []
all = []
for a in sample:
    for b in sample:
        for c in sample:
            to_list = a+b+c+""
            all.append(list(to_list))
            
print(all)