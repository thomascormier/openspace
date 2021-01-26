import csv

def create_csv():
    """
    Create a csv called number_of_people.csv with the following headers :     
    number_of_people, nb_person_threshold
    """
    with open('number_of_people.csv', 'w', newline='') as f:
        fieldnames = ['number_of_people', 'nb_person_threshold']
        writer = csv.DictWriter(f, fieldnames=fieldnames)

        writer.writeheader()
    f.close()

def update_csv(nb_pers):
    """
    Add a new row to the csv file number_of_people.csv
    """
    with open('number_of_people.csv', 'a', newline='') as f:
        fieldnames = ['number_of_people', 'nb_person_threshold']
        writer = csv.DictWriter(f, fieldnames=fieldnames)

        writer.writerow({'number_of_people': nb_pers, 'nb_person_threshold': 3})
    f.close()
