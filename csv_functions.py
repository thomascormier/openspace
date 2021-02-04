import csv

def create_csv(name, field1, field2):
    """
    Create a csv called number_of_people.csv with the following headers :     
    number_of_people, nb_person_threshold
    """
    with open(name, 'w', newline='') as f:
        fieldnames = [field1, field2]
        writer = csv.DictWriter(f, fieldnames=fieldnames)

        writer.writeheader()
    f.close()

def update_csv(nb_pers,threshold):
    """
    Add a new row to the csv file number_of_people.csv
    """
    with open('number_of_people.csv', 'a', newline='') as f:
        fieldnames = ['number_of_people', 'nb_person_threshold']
        writer = csv.DictWriter(f, fieldnames=fieldnames)

        writer.writerow({'number_of_people': nb_pers, 'nb_person_threshold': threshold})
    f.close()
