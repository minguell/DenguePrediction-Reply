import csv
import numpy
import sys


inputList = []
metData = {2017:[],2018:[],2019:[],2020:[],2021:[],2022:[]}

output = []

demographyDict = {}
neighbours = {}

inputMetData = ["dataMet/INMET_POA_2017_MONTH.csv","dataMet/INMET_POA_2018_MONTH.csv","dataMet/INMET_POA_2019_MONTH.csv","dataMet/INMET_POA_2020_MONTH.csv","dataMet/INMET_POA_2021_MONTH.csv","dataMet/INMET_POA_2022_MONTH.csv"]


def readNeighbours():
    with open("data/neighbours.csv",newline='',encoding="ISO-8859-1") as csvfile:
        reader = csv.reader(csvfile,delimiter=";")
        rows = list(reader)
        for row in rows:
            neighbours[row[0]] = row[1:]

def readDemographyData():
    with open("data/demografia.csv",newline='',encoding="ISO-8859-1") as csvfile:
        reader = csv.reader(csvfile,delimiter=";")
        rows = list(reader)
        for row in rows:
            demographyDict[row[0]] = int(row[1])

def readMetData():
    year = 2017
    for inp in inputMetData:
        with open(inp,newline='',encoding="ISO-8859-1") as csvfile:
            reader = csv.reader(csvfile,delimiter=";")
            rows = list(reader)
            for i in range(1,len(rows)):
                metData[year].append(rows[i])

        year+=1

def addToInputList(yearDiagnosis, distList):
    year = 2017
    month = 1
    #print("distrito","dengue_diagnosis","ano","mes","precipitacao","temperatura","umidade")
    output.append(["nome_distrito","dengue_diagnosis","ano","mes","precipitacao (mm)","temperatura (°C)","umidade ar (%)","Populacao","t-1","t-2","t-3","precipitacao (mm)-1","temperatura (°C)-1","umidade ar (%)-1","zika-1","chikungunya-1","sum_vizinhos_t-1","qtd_cnes","liraa"])
    for dist in distList:
        year = 2017
        month = 1

        



        for i in range(0,len(yearDiagnosis)):

            sum_neighbours_1 = 0

            for neighbour in neighbours[dist]:
                if neighbour == '':
                    continue
                if i == 0 and year == 2017:
                    sum_neighbours_1 += yearDiagnosis[i][neighbour]
                else:
                    sum_neighbours_1 += yearDiagnosis[i-1][neighbour]

            dengue_1 = 0
            dengue_2 = 0
            dengue_3 = 0
            precip_1 = 0
            temp_1 = 0
            humid_1 = 0
            if i == 1 and year == 2017:
                dengue_2 = yearDiagnosis[i-1][dist]
            else:
                dengue_2 = yearDiagnosis[i-2][dist]
            if i == 2 and year == 2017:
                dengue_3 = yearDiagnosis[i-2][dist]
            else:
                dengue_3 = yearDiagnosis[i-3][dist]
            if i == 0 and year == 2017:
                precip_1,temp_1,humid_1 = [metData[year][month-1][1],metData[year][month-1][2],metData[year][month-1][3]]
                dengue_1 = yearDiagnosis[i][dist]
            elif month == 1:
                precip_1,temp_1,humid_1 = [metData[year-1][11][1],metData[year-1][11][2],metData[year-1][11][3]]
                dengue_1 = yearDiagnosis[i-1][dist]
            else:
                precip_1,temp_1,humid_1 = [metData[year][month-2][1],metData[year][month-2][2],metData[year][month-2][3]]
                dengue_1 = yearDiagnosis[i-1][dist]
            
            
            
            output.append([dist,yearDiagnosis[i][dist],year,month, metData[year][month-1][1],metData[year][month-1][2],metData[year][month-1][3],demographyDict[dist], dengue_1,dengue_2,dengue_3,precip_1, temp_1, humid_1,dengue_1,dengue_2,sum_neighbours_1,0,0])
            month+=1
            if month == 13:
                year+=1
                month = 1
            

def writeOutput():  
    with open("dengue_input.csv",'w') as csvfilew: #write final data (month-by-month)
        writer = csv.writer(csvfilew,delimiter=";")

        for row in output:
            writer.writerow(row)

yearDiagnosis = []
for i in range(1,len(sys.argv)):
    with open(sys.argv[i],newline='',encoding="ISO-8859-1") as csvfile: #Read raw dengue data
        reader = csv.reader(csvfile,delimiter=";")
        rows = list(reader)
        
        
        dengueDiagnosis = {'CENTRO':0,'CENTRO SUL':0,'CRISTAL':0, 'CRUZEIRO':0, 'EIXO BALTAZAR':0, 'EXTREMO SUL':0, 'GLORIA':0, 'HUMAITA NAVEGANTES':0, 'ILHAS':0, 'LESTE':0, 'LOMBA DO PINHEIRO':0, 'NORDESTE':0, 'NOROESTE':0, 'NORTE':0, 'PARTENON':0, 'RESTINGA':0,'SUL':0}
        
        for i in range(1,len(rows)):
            district = rows[i][0]
            if district == "(Sem Ref.)":
                continue
            if district == '-':
                yearDiagnosis.append(dengueDiagnosis)
                dengueDiagnosis = {'CENTRO':0,'CENTRO SUL':0,'CRISTAL':0, 'CRUZEIRO':0, 'EIXO BALTAZAR':0, 'EXTREMO SUL':0, 'GLORIA':0, 'HUMAITA NAVEGANTES':0, 'ILHAS':0, 'LESTE':0, 'LOMBA DO PINHEIRO':0, 'NORDESTE':0, 'NOROESTE':0, 'NORTE':0, 'PARTENON':0, 'RESTINGA':0,'SUL':0}
                continue
            dengueDiagnosis[district]+=int(rows[i][1]) #Notifications

readNeighbours()
readDemographyData()
readMetData()
addToInputList(yearDiagnosis,dengueDiagnosis)

writeOutput()