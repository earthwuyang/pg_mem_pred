def Qerror(dataset, output, memory): # meandian Q-Error
    op = dataset.memory_scaler.inverse_transform(output.cpu().reshape(-1, 1)).reshape(-1)
    mem = dataset.memory_scaler.inverse_transform(memory.cpu().reshape(-1, 1)).reshape(-1)
    count = 0 
    for i in range(len(output)):
        count += max(op[i], mem[i]) / (min(op[i], mem[i]) + 1e-10 )
        # count += max(output[i], memory[i]) / (min(output[i], memory[i]) + 1e-10 )
    return count/len(output)

def MedianQerror(dataset, output, memory): # meandian Q-Error
    output = dataset.memory_scaler.inverse_transform(output.cpu().reshape(-1, 1)).reshape(-1)
    memory = dataset.memory_scaler.inverse_transform(memory.cpu().reshape(-1, 1)).reshape(-1)
    qerror_list = []
    for i in range(len(output)):
        qerror_list.append(max(output[i], memory[i]) / (min(output[i], memory[i]) + 1e-10 ))
    qerror_list = sorted(qerror_list)
    return qerror_list[0], qerror_list[int(len(qerror_list)/2)], qerror_list[int(len(qerror_list)*0.95)], qerror_list[-1]

def MRE(dataset,output, memory): # mean relative error
    op = dataset.memory_scaler.inverse_transform(output.cpu().reshape(-1, 1)).reshape(-1)
    mem = dataset.memory_scaler.inverse_transform(memory.cpu().reshape(-1, 1)).reshape(-1)
    count = 0 
    mre_list = []
    # output = torch.exp(output)
    # memory = torch.exp(memory)      
    for i in range(len(output)):    
        count += abs(op[i] - mem[i]) / (mem[i] + 1e-10)
        mre_list.append(abs(op[i] - mem[i]) / (mem[i] + 1e-10))
    mre_list = sorted(mre_list)

    return count/len(output)


def MAE(output, memory): # mean absolute error  

    count = 0 
    mae_list = []
    output = torch.exp(output)
    memory = torch.exp(memory)
    for i in range(len(output)):
        mae_list.append(abs(output[i] - memory[i]))
        count += abs(output[i] - memory[i])
    mae_list = sorted(mae_list)
    mae_0 = mae_list[0]
    mae_20 = mae_list[int(len(mae_list)*0.2)]
    mae_50 = mae_list[int(len(mae_list)*0.5)]
    mae_90 = mae_list[int(len(mae_list)*0.9)]
    # print(f"mae_0: {mae_0}, mae_20: {mae_20}, mae_50: {mae_50}, mae_90: {mae_90}")
    return count/len(output)