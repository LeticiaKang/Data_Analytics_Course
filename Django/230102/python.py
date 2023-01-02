# list1 = ['alpha', 'bravo', 'delta', 'echo', 'foxtrot', 'golf',
#          'hotal', 'india']

# 1번 풀이
# for a in list1:
#     if len(a)==5:
#         print(a)
        
# print([a for a in list1 if len(a)==5])

         
# 문제2. 자동판매기
# ihave, cost= map(int, input().split())
# balance = ihave - cost
# obec = balance//500
# bec = (balance%500)//100
# print(f"""거스름돈을 확인하세요.
# 잔돈 : {balance}원
# 500원 : {obec}개
# 100원 : {bec}개""")

# # 문제3. 화폐단위 

# # # 방법1
# money = int(input("금액을 입력하세요 : "))

# moneys = {'50000원':50000, '10000원': 10000, '5000원': 5000, '1000원': 1000,
#                 '500원': 500, '100원': 100, '50원': 50, '10원': 10}
# counts = {}
    
# for a, b in denominations.items():
#     counts[a] = money//denominations[a]
#     money = money - counts[a]*denominations[a]
        
# print(f"""
# 오만원권 {counts['50000원']}매
# 만원권 {counts['10000원']}매
# 오천원권 {counts['5000원']}매
# 천원권 {counts['1000원']}매
# 오백원 {counts['500원']}개
# 백원 {counts['100원']}개
# 오십원 {counts['50원']}개
# 십원 {counts['10원']}개
# """)

# # # 방법2
# money = int(input("금액을 입력하세요 : "))
# moneys = [50000,10000,5000,1000,500,100,50,10]
# list2=['오만', '만원', '오천', '천원', '오백', '백', '오십', '십원']

# for i in range(len(moneys)):
#     count = money // moneys[i]
#     money = money % moneys[i]
#     if money >= 1000:
#         print(f"{list2[i]}원권 : {count}매")
#     else:
#         print(f"{list2[i]}원 : {count}개")

# # # 방법3
# # money=int(input("금액을 입력하세요 : "))
# list= [50000, 10000, 5000, 1000, 500, 100, 50, 10]
# list2=['오만', '만원', '오천', '천원', '오백', '백', '오십', '십원']
# for i in range(len(list)):
#     if money >=1000:
#         count = int(money/list[i])
#         money = money % list[i]
#         print(f"{list2[i]}원권 : {count}매") 
#     else:
#         count = int(money/list[i])
#         money = money % list[i]
#         print(f"{list2[i]}원 : {count}개")

#방법4
cost = int(input('금액을 입력하세요. : '))
print('입력받은 금액:',cost,'원')
korean = {50000:'오만원권',10000:'만원권',5000:'오천원권',1000:'천원권',500:'오백원',100:'백원',50:'오십원',10:'십원'}
for i in korean:
    tmp,cost = divmod(cost,i)
    if i>=1000:
        print(f'{korean[i]} : {tmp}매')
    else:
        print(f'{korean[i]} : {tmp}개')