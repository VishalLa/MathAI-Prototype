import sys
sys.path.append(r"C:\Users\visha\OneDrive\Desktop\MathAI")

label_to_index = {
    '0': 0,
    '1': 1, 
    '2': 2, 
    '3': 3, 
    '4': 4, 
    '5': 5, 
    '6': 6, 
    '7': 7, 
    '8': 8, 
    '9': 9, 
    'add': 10,
    'sub': 11, 
    'mul': 12,
    # 'div': 12, 
    # 'eq': 13, 
    # 'sub': 15,
    # # '(': 16, 
    # # ')': 17, 
    # 'x': 16,  
    # 'y': 17, 
    # # 'z': 20,
}

index_to_label = {v: k for k, v in label_to_index.items()}


classes = 13
learning_rate = 0.0001
iterations = 20