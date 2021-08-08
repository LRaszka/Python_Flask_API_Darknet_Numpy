import numpy as np

array1 = np.array([1,2,3], dtype='int16') #dtype - nastavení datového typu pole
array2 = np.array([[3,2,1],[1,2,3]])

array_long = np.array([[7,6,5,4,3,2,1],[1,2,3,4,5,6,7]])
print(array_long)

print(array_long[1,5]) #druhé pole, šestý prvek
print(array_long[1,-2]) #druhé pole, šestý prvek

array_long[1,5]=15 #změna prvku druhého pole, šestého prvek na 15
print(array_long)

print(array_long[0, :]) #všechny prvky prvního pole

print(array_long[:, 2]) #všechny řádky, třetí prvek

print(array2)

np.save("a.npy", array2)
array3 = np.load("a.npy")

print(array3)

print(array3.ndim) #počet dimenzí pole

print(array3.shape) #počet prvků pole

print(array1.dtype) #datový typ pole

print(array1.itemsize) #počet byte

print(np.zeros(5)) #vytvoření jednorozměrného pole o 5 prvcích - samé nuly 
print(np.zeros((2,3))) #dvě dimenze, tři prvky
print(np.zeros((2,3,3))) #tři dimenze
#print(np.zeros((2,3,3,2,2,2))) #wtf

print()
print(np.ones(5)) #vytvoření jednorozměrného pole o 5 prvcích - samé jedničky
print(np.ones((5), dtype='int32')) #vytvoření jednorozměrného pole o 5 prvcích - samé jedničky - int

print(np.random.rand(4,2)) #pole náhodných číles (float) - dvourozměrné - 4 řádky a dva prvky
print(np.random.randint(100, size=(3,3))) #pole náhodných číles (float) - dvourozměrné - 3 řádky a tři prvky; první čáslo znamená hranici čísel

math = np.array([1,2,3])
print(math)
math += 2
print(math)
math -= 2
print(math)
math *= 2
print(math)
math = math / 2
print(math)

math2 = np.array([2,4,6])
print(math+math2)


a = 3
print(a**2) # druhá mocnina

print(np.min(math)) #min
print(np.max(math)) #max
print(np.sum(math)) #součet


filedata = np.genfromtxt('data.txt', delimiter=',') #načtení dat z txt
filedata = filedata.astype('int32') #převod na celé čísla 
print(filedata)

print(filedata>50) #jsou větší jak 50?
print(filedata==5) #jsou 5?

velka_cisla = filedata[filedata>1] #jen čísla větší než 1
print(velka_cisla)
