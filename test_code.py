import pickle 

def load_file(): 
    file = open("output/preprocessing/data_face68.pkl", "rb")

    data = pickle.load(file)

    print(type(data))
    print("len data: ", len(data))
    print(type(data[0]))
    print(data[0]) 

if __name__ == "__main__":
    load_file() 