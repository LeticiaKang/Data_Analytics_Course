


def YesDuple(sample):
    
    all = [] #묶음 집합

    num = 0
    to_list = ""
    while True:
        for a in sample[num: ]:
            to_list = to_list + a 
            all.append(list(to_list))
            num += 1
        
        if len(sample) == num:
            break

    return all

test = [1,2,3]
YesDuple(test)

            


