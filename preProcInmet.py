import csv
import numpy
import sys


dateDelimiter = '-'


newData = []
lastDay = []
newData.append(["Date","Precipitation","Temperature","Relative humidity"])
with open(sys.argv[1],newline='',encoding="ISO-8859-1") as csvfile:
    reader = csv.reader(csvfile,delimiter=";")

    rows = list(reader)
    #start on 9th row
    cnt = 0
    data = numpy.zeros(4,dtype=float)


    disc = 0
    for i in range(9,len(rows)):
        if rows[i][2] == "" or rows[i][7] == "" or rows[i][15] == "" or float(rows[i][2].replace(',','.')) < 0 or float(rows[i][15].replace(',','.')) < 0:
            disc+=1
        else:
            data[0] = data[0] + float(rows[i][2].replace(',','.'))  #Precipitation
            data[1] = data[1] + float(rows[i][7].replace(',','.'))  #Temperature
            data[2] = data[2] + float(rows[i][15].replace(',','.')) #Relative humidity

        cnt+=1
        if cnt == 24:
            date = rows[i][0]
            #Take the mean
            
            if data[0] == 0 and data[1] == 0 and data[2] == 0:
                local = [date, lastDay[0],lastDay[1],lastDay[2]]
            else:
                data[1] /= 24-disc
                data[2] /= 24-disc
                local = [date,str(data[0]),str(data[1]),str(data[2])]

            lastDay = [str(data[0]),str(data[1]),str(data[2])]
            #print(local)
            newData.append(local)
            data = numpy.zeros(4,dtype=float)
            cnt = 0
            disc=0


with open(sys.argv[2],'w') as csvfilew:
    writer = csv.writer(csvfilew,delimiter=";")

    for row in newData:
        writer.writerow(row)

newData = []
newData.append(["Month","Precipitation","Temperature","Relative humidity"])
with open(sys.argv[2]) as csvfile:
    reader = csv.reader(csvfile,delimiter=";")

    rows = list(reader)
    data = numpy.zeros(4,dtype=float)
    days = 0
    for i in range(1,len(rows)-1):
        data[0] = data[0] + float(rows[i][1])  #Precipitation
        data[1] = data[1] + float(rows[i][2])  #Temperature
        data[2] = data[2] + float(rows[i][3]) #Relative humidity
        days+=1
        if rows[i][0].split(dateDelimiter)[1] != rows[i+1][0].split(dateDelimiter)[1]:
            newData.append([rows[i][0].split(dateDelimiter)[0] + '/' + rows[i][0].split(dateDelimiter)[1],round(data[0],2),round(data[1]/days,2),round(data[2]/days,2)])
            data = numpy.zeros(4,dtype=float)
            days = 0 
    newData.append([rows[i][0].split(dateDelimiter)[0] + '/' + rows[i][0].split(dateDelimiter)[1],round(data[0],2),round(data[1]/days,2),round(data[2]/days,2)])


with open(sys.argv[3],'w') as csvfilew:
    writer = csv.writer(csvfilew,delimiter=";")

    for row in newData:
        writer.writerow(row)