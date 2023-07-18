import random
import math
import matplotlib.pyplot as plt

def relax(t):
    numerator = 4 * (1 - math.exp(-1.5 * t))
    denominator = 4 * (1 + math.exp(-1.5 * t))
    result = numerator / denominator
    return result

def save_list_to_file(data_list, file_path):
    try:
        with open(file_path, 'w') as file:
            for item in data_list:
                file.write(str(item) + '\n')
    except IOError as e:
        print(f"Error writing to the file: {e}")
    else:
        print("List successfully saved to the file.")

def relu(x):
    alpha = 0.01
    return max(alpha * x, x)

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def elu(x):
    alpha = 1.0
    if x >= 0:
        return x
    else:
        return alpha * (math.exp(x) - 1)

# a is top left pixel, b is bottom left pixel, c is top right pixel, d is bottom right pixel

global individual_runtime
global population
global iterations
global mut
global preservation
individual_runtime = int(input("Individual runtime: "))
population = int(input("Batch population: "))
iterations = int(input("Number of batches until completion: "))
mut = float(input("Mutation constant: "))
preservation = int(input("Mutation momentum: "))

def mse(y_true, y_pred):
    return (y_true - y_pred) ** 2

def calculate_cost(vertical, horizontal, diagonal, solid, pattern):
    if pattern == 0 or pattern == 1:
        return relax(mse(1, vertical) + mse(0, horizontal) + mse(0, diagonal) + mse(0, solid))
    elif pattern == 2 or pattern == 3:
        return relax(mse(1, horizontal) + mse(0, vertical) + mse(0, diagonal) + mse(0, solid))
    elif pattern == 4 or pattern == 5:
        return relax(mse(1, diagonal) + mse(0, vertical) + mse(0, horizontal) + mse(0, solid))
    elif pattern == 6 or pattern == 7:
        return relax(mse(1, solid) + mse(0, vertical) + mse(0, horizontal) + mse(0, diagonal))
    else:
        raise ValueError("Invalid pattern...")

def patterngen():
    global pattern
    pattern = random.randint(0, 3)
    if pattern == 0:
        # vertical
        return 1, 1, 0, 0, 1
    elif pattern == 1:
        # horizontal
        return 1, 0, 1, 0, 2
    elif pattern == 2:
        # diagonal
        return 1, 0, 0, 1, 3
    elif pattern == 3:
        # solid
        return 0, 0, 0, 0, 4

def epoch():
    global basew
    global basebias
    global record
    global mut
    global preservation
    global records
    global recordless
    global population
    global panswer
    blibrary = []
    wlibrary = []
    batchrecord = 4
    for i in range(population):
        w = [i + random.uniform(-(mut), mut) if -1 <= i <= 1 else i + random.uniform(-(mut), 0) if i > 1 else i + random.uniform(0, mut) if i < -1 else i for i in basew]
        bias = [i + random.uniform(-(mut), mut) if -1 <= i <= 1 else i + random.uniform(-(mut), 0) if i > 1 else i + random.uniform(0, mut) if i < -1 else i for i in basebias]
        rawcost = 0
        for i in range(individual_runtime):
            a, b, c, d, answer = patterngen()
            h1a = elu(a * w[1] + b * w[2] + c * w[3] + d * w[4] + bias[1])
            h1b = elu(a * w[5] + b * w[6] + c * w[7] + d * w[8] + bias[2])
            h1c = elu(a * w[9] + b * w[10] + c * w[11] + d * w[12] + bias[3])
            h1d = elu(a * w[13] + b * w[14] + c * w[15] + d * w[16] + bias[4])
            h2a = relu(h1a * w[17] + h1b * w[18] + h1c * w[19] + h1d * w[20] + bias[5])
            h2b = relu(h1a * w[21] + h1b * w[22] + h1c * w[23] + h1d * w[24] + bias[6])
            h2c = relu(h1a * w[25] + h1b * w[26] + h1c * w[27] + h1d * w[28] + bias[7])
            h2d = relu(h1a * w[29] + h1b * w[30] + h1c * w[31] + h1d * w[32] + bias[8])
            h3a = elu(h2a * w[33] + h2b * w[34] + h2c * w[35] + h2d * w[36] + bias[9])
            h3b = elu(h2a * w[37] + h2b * w[38] + h2c * w[39] + h2d * w[40] + bias[10])
            h3c = elu(h2a * w[41] + h2b * w[42] + h2c * w[43] + h2d * w[44] + bias[11])
            h3d = elu(h2a * w[45] + h2b * w[46] + h2c * w[47] + h2d * w[48] + bias[12])
            h4a = relu(h3a * w[48] + h3b * w[49] + h3c * w[50] + h3d * w[51] + bias[13])
            h4b = relu(h3a * w[52] + h3b * w[53] + h3c * w[54] + h3d * w[55] + bias[14])
            h4c = relu(h3a * w[56] + h3b * w[57] + h3c * w[58] + h3d * w[59] + bias[15])
            h4d = relu(h3a * w[60] + h3b * w[61] + h3c * w[62] + h3d * w[63] + bias[16])
            h4e = relu(h3a * w[64] + h3b * w[65] + h3c * w[66] + h3d * w[67] + bias[17])
            h4f = relu(h3a * w[68] + h3b * w[69] + h3c * w[70] + h3d * w[71] + bias[18])
            h5a = relu(h4a * w[72] + h4b * w[73] + h4c * w[74] + h4d * w[75] + h4e * w[76] + h4f * w[77] + bias[19])
            h5b = relu(h4a * w[78] + h4b * w[79] + h4c * w[80] + h4d * w[81] + h4e * w[82] + h4f * w[83] + bias[20])
            h5c = relu(h4a * w[84] + h4b * w[85] + h4c * w[86] + h4d * w[87] + h4e * w[88] + h4f * w[89] + bias[21])
            h6a = relu(h5a * w[90] + h5b * w[91] + h5c * w[92] + bias[22])
            h6b = relu(h5a * w[93] + h5b * w[94] + h5c * w[95] + bias[23])
            h6c = relu(h5a * w[96] + h5b * w[97] + h5c * w[98] + bias[0])
            vertical = sigmoid(h6a * w[99] + h6b * w[100] + h6c * w[101])
            horizontal = sigmoid(h6a * w[102] + h6b * w[103] + h6c * w[104])
            diagonal = sigmoid(h6a * w[105] + h6b * w[106] + h6c * w[107])
            solid = sigmoid(h6a * w[108] + h6b * w[109] + h6c * w[0])
            rawcost += calculate_cost(vertical, horizontal, diagonal, solid, pattern)
        cost = rawcost / individual_runtime
        if cost < record:
            record = cost
            basew = w
            basebias = bias
            mut = mut - (mut/preservation)
            records += 1
            recordless = 0
        elif cost < batchrecord:
            pvertical = vertical
            phorizontal = horizontal
            pdiagonal = diagonal
            psolid = solid
            batchrecord = cost
            panswer = answer
    print(str(panswer))
    print(str(round(pvertical, 2)), ", ", str(round(phorizontal, 2)), ", ", str(round(pdiagonal, 2)), ", ", str(round(psolid, 2)), ", Cost:", str(round(record, 2)), "    ", mut, "|", records)
    if recordless > 3:
        if mut < 1:
            mut = mut + (mut/preservation)
        elif mut == 1:
            if recordless > 10:
                population += population
        else:
            mut = 1
        recordless = 0
    else:
        recordless += 1

def final():
    global basew
    global basebias
    global cost
    for i in range(20):
        a, b, c, d, answer = patterngen()
        w = basew
        bias = basebias
        h1a = elu(a * w[1] + b * w[2] + c * w[3] + d * w[4] + bias[1])
        h1b = elu(a * w[5] + b * w[6] + c * w[7] + d * w[8] + bias[2])
        h1c = elu(a * w[9] + b * w[10] + c * w[11] + d * w[12] + bias[3])
        h1d = elu(a * w[13] + b * w[14] + c * w[15] + d * w[16] + bias[4])
        h2a = relu(h1a * w[17] + h1b * w[18] + h1c * w[19] + h1d * w[20] + bias[5])
        h2b = relu(h1a * w[21] + h1b * w[22] + h1c * w[23] + h1d * w[24] + bias[6])
        h2c = relu(h1a * w[25] + h1b * w[26] + h1c * w[27] + h1d * w[28] + bias[7])
        h2d = relu(h1a * w[29] + h1b * w[30] + h1c * w[31] + h1d * w[32] + bias[8])
        h3a = elu(h2a * w[33] + h2b * w[34] + h2c * w[35] + h2d * w[36] + bias[9])
        h3b = elu(h2a * w[37] + h2b * w[38] + h2c * w[39] + h2d * w[40] + bias[10])
        h3c = elu(h2a * w[41] + h2b * w[42] + h2c * w[43] + h2d * w[44] + bias[11])
        h3d = elu(h2a * w[45] + h2b * w[46] + h2c * w[47] + h2d * w[48] + bias[12])
        h4a = relu(h3a * w[48] + h3b * w[49] + h3c * w[50] + h3d * w[51] + bias[13])
        h4b = relu(h3a * w[52] + h3b * w[53] + h3c * w[54] + h3d * w[55] + bias[14])
        h4c = relu(h3a * w[56] + h3b * w[57] + h3c * w[58] + h3d * w[59] + bias[15])
        h4d = relu(h3a * w[60] + h3b * w[61] + h3c * w[62] + h3d * w[63] + bias[16])
        h4e = relu(h3a * w[64] + h3b * w[65] + h3c * w[66] + h3d * w[67] + bias[17])
        h4f = relu(h3a * w[68] + h3b * w[69] + h3c * w[70] + h3d * w[71] + bias[18])
        h5a = relu(h4a * w[72] + h4b * w[73] + h4c * w[74] + h4d * w[75] + h4e * w[76] + h4f * w[77] + bias[19])
        h5b = relu(h4a * w[78] + h4b * w[79] + h4c * w[80] + h4d * w[81] + h4e * w[82] + h4f * w[83] + bias[20])
        h5c = relu(h4a * w[84] + h4b * w[85] + h4c * w[86] + h4d * w[87] + h4e * w[88] + h4f * w[89] + bias[21])
        h6a = relu(h5a * w[90] + h5b * w[91] + h5c * w[92] + bias[22])
        h6b = relu(h5a * w[93] + h5b * w[94] + h5c * w[95] + bias[23])
        h6c = relu(h5a * w[96] + h5b * w[97] + h5c * w[98] + bias[0])
        vertical = sigmoid(h6a * w[99] + h6b * w[100] + h6c * w[101])
        horizontal = sigmoid(h6a * w[102] + h6b * w[103] + h6c * w[104])
        diagonal = sigmoid(h6a * w[105] + h6b * w[106] + h6c * w[107])
        solid = sigmoid(h6a * w[108] + h6b * w[109] + h6c * w[0])
        cost = calculate_cost(vertical, horizontal, diagonal, solid, pattern)
        print(str(answer))
        print(str(round(vertical, 2)), ", ", str(round(horizontal, 2)), ", ", str(round(diagonal, 2)), ", ", str(round(solid, 2)), ", Cost:", str(round(cost, 2)))

print("Initializing...")
epochs_passed = []
records_list = []
final_epochs = []
cost_list = []
records = 0
recordless = 0
record = 4
basew = [0 for _ in range(110)]
basebias = [0 for _ in range(24)]

print("Training...")
for i in range(iterations):
    epoch()
    epochs_passed.append(i + 1)
    records_list.append(record)

plt.plot(epochs_passed, records_list)
plt.xlabel('Epochs')
plt.ylabel('Best Cost (Record)')
plt.title('Best Cost (Record) vs. Epochs')
plt.show()

print("--- TRAINING COMPLETE ---")

print("Evaluating final patterns...")

for i in range(10):
    final()
    final_epochs.append(i + 1)
    cost_list.append(cost)
    
plt.plot(final_epochs, cost_list)
plt.xlabel('Final Epochs')
plt.ylabel('Best Cost (Record)')
plt.title('Best Cost (Record) vs. Epochs')
plt.show()

save_list_to_file(basew, r"C:\Users\Brendan Honzik\Dropbox\PC\Desktop\values\weights.txt")
save_list_to_file(basebias, r"C:\Users\Brendan Honzik\Dropbox\PC\Desktop\values\bias.txt")
