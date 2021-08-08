import time
import requests
import json

id = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31]
obsazeno = [True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True]
now_time = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]

obsazeno_frame = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
neobsazeno_frame = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]

# Délka obsazenosti parkovacího místa
# cas = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
# cas_last = 0
# cas_real = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]

# den = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
# den_hod = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
# hodina = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
# hod_min = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
# minuta = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
# min_sec = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
# sekunda = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]

# url = "http://167.71.38.238:3080/api/data"

for i in range(32):
    Obsazeno_User = input("Je obsazeno? ")
    if Obsazeno_User == "t":
        obsazeno_frame[i] = obsazeno_frame[i] + 1
        if obsazeno_frame[i] == 1:
            now_time[i] = time.ctime()
            # cas[i] = time.time()
        print(str(i) + " Počet obsazeno frame: " + str(obsazeno_frame[i]))
        neobsazeno_frame[i] = 0
        obsazeno[i] = True
    else:
        neobsazeno_frame[i] = neobsazeno_frame[i] + 1
        print(str(i) + " Počet neobsazeno frame: " + str(neobsazeno_frame[i]))
        if neobsazeno_frame[i] >= 8:
            obsazeno_frame[i] = 0
            obsazeno[i] = False
            now_time[i] = 0
            # cas[i] = 0

# cas_last = time.time()
# for x in range(32):
#     cas_real[x] = cas_last - cas[x]
#     now_time[x] = cas_real[x]

#     if cas_real[x] >= 60:
#         min_sec[x] = cas_real[x] / 60
#         minuta[x] = int(min_sec[x])
#         sekunda[x] = cas_real[x] % 60
#         if minuta[x] >= 60:
#             hod_min[x] = minuta[x] / 60
#             hodina[x] = int(hod_min[x])
#             minuta[x] = minuta[x] % 60
#             if hodina[x] >=24:
#                 den_hod[x] = hodina[x] / 24
#                 den[x] = int(den_hod[x])
#                 hodina[x] = hodina[x] % 24
#     else:
#         sekunda[x] = cas_real[x]

# print(sekunda)
# print(minuta)
# print(hodina)
# print(den)

JSON = [
{
    "id": id[0],
    "obsazeno": obsazeno[0],
    "datum": now_time[0]
    # "sekunda": sekunda[0],
    # "minuta": minuta[0],
    # "hodina": hodina[0],
    # "den": den[0]
},
{
    "id": id[1],
    "obsazeno": obsazeno[1],
    "datum": now_time[1]
},
{
    "id": id[2],
    "obsazeno": obsazeno[2],
    "datum": now_time[2]
},
{
    "id": id[3],
    "obsazeno": obsazeno[3],
    "datum": now_time[3]
},
{
    "id": id[4],
    "obsazeno": obsazeno[4],
    "datum": now_time[4]
},
{
    "id": id[5],
    "obsazeno": obsazeno[5],
    "datum": now_time[5]
},
{
    "id": id[6],
    "obsazeno": obsazeno[6],
    "datum": now_time[6]
},
{
    "id": id[7],
    "obsazeno": obsazeno[7],
    "datum": now_time[7]
},
{
    "id": id[8],
    "obsazeno": obsazeno[8],
    "datum": now_time[8]
},
{
    "id": id[9],
    "obsazeno": obsazeno[9],
    "datum": now_time[9]
},
{
    "id": id[10],
    "obsazeno": obsazeno[10],
    "datum": now_time[10]
},
{
    "id": id[11],
    "obsazeno": obsazeno[11],
    "datum": now_time[11]
},
{
    "id": id[12],
    "obsazeno": obsazeno[12],
    "datum": now_time[12]
},
{
    "id": id[13],
    "obsazeno": obsazeno[13],
    "datum": now_time[13]
},
{
    "id": id[14],
    "obsazeno": obsazeno[14],
    "datum": now_time[14]
},
{
    "id": id[15],
    "obsazeno": obsazeno[15],
    "datum": now_time[15]
},
{
    "id": id[16],
    "obsazeno": obsazeno[16],
    "datum": now_time[16]
},
{
    "id": id[17],
    "obsazeno": obsazeno[17],
    "datum": now_time[17]
},
{
    "id": id[18],
    "obsazeno": obsazeno[18],
    "datum": now_time[18]
},
{
    "id": id[19],
    "obsazeno": obsazeno[19],
    "datum": now_time[19]
},
{
    "id": id[20],
    "obsazeno": obsazeno[20],
    "datum": now_time[20]
},
{
    "id": id[21],
    "obsazeno": obsazeno[21],
    "datum": now_time[21]
},
{
    "id": id[22],
    "obsazeno": obsazeno[22],
    "datum": now_time[22]
},
{
    "id": id[23],
    "obsazeno": obsazeno[23],
    "datum": now_time[23]
},
{
    "id": id[24],
    "obsazeno": obsazeno[24],
    "datum": now_time[24]
},
{
    "id": id[25],
    "obsazeno": obsazeno[25],
    "datum": now_time[25]
},
{
    "id": id[26],
    "obsazeno": obsazeno[26],
    "datum": now_time[26]
},
{
    "id": id[27],
    "obsazeno": obsazeno[27],
    "datum": now_time[27]
},
{
    "id": id[28],
    "obsazeno": obsazeno[28],
    "datum": now_time[28]
},
{
    "id": id[29],
    "obsazeno": obsazeno[29],
    "datum": now_time[29]
},
{
    "id": id[30],
    "obsazeno": obsazeno[30],
    "datum": now_time[30]
},
{
    "id": id[31],
    "obsazeno": obsazeno[31],
    "datum": now_time[31]
}
]
# print("Data z detekce: " + str(JSON))
print("________________________________________________________________________________________________________________________")
data = json.dumps(JSON)
print("Data z detekce: " + str(data))
# response = requests.post(url, data=data, headers={"Content-Type": "application/json"})
# print("odpověď serveru/cloudu: " + str(response.json()))