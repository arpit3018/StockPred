import csv

file1 = "./companylist.csv"
file2 = "./ind_nifty500list.csv"

# Symbol, Name, Sector

with open(file1) as csv_file:
    csv_reader = csv.DictReader(csv_file, delimiter=',')
    line_count = 0
    for row in csv_reader:
        if line_count == 0:
            line_count += 1
        else:
            print(row["Symbol"], row["Name"], row["Sector"], sep='\t')

with open(file2) as csv_file:
    csv_reader = csv.DictReader(csv_file, delimiter=',')
    line_count = 0
    for row in csv_reader:
        if line_count == 0:
            line_count += 1
        else:
            print(row["Symbol"], row["Name"], row["Industry"], sep='\t')