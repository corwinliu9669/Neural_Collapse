import torch
import numpy as np
import torch.nn.functional as F
#from sklearn.metrics.pairwise import cosine_similarity
def evaluate_batch(model, data_loader, device):
	model.eval()
	correct = num = correct_t5 =0
	for iter, pack in enumerate(data_loader):
		data, target = pack['image'].to(device), pack['label'].to(device)
		logits = model(data)
		_, pred = logits.max(1)
		_, pred_t5 = torch.topk(logits, 5, dim=1)
		correct += pred.eq(target).sum().item()
		correct_t5 += pred_t5.eq(torch.unsqueeze(target, 1).repeat(1, 5)).sum().item()
		num += data.shape[0]
	print('Correct : ', correct)
	print('Num : ', num)
	print('Test ACC : ', correct / num)
	print('Top 5 ACC : ', correct_t5 / num)
	torch.cuda.empty_cache()
	model.train()
	return correct / num

def descent_lr(epoch, optimizer, lr, lr_decay, epoch_schedule):
        index = 0
        for k in epoch_schedule:
            if epoch > k:
                index += 1
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr * lr_decay ** index
        print('***********************************')
        print('epoch:', epoch)
        print('learning rate:', param_group['lr'])
        print('***********************************')

def norm(x):
    return np.linalg.norm(x, ord=2, axis=1, keepdims=False)


def neural_collapse_embedding(model, data_loader, device, num_class):
    model.eval()
    embedding = []
    label = []
    for iter, pack in enumerate(data_loader):
        data, target = pack['image'].to(device), pack['label'].to(device)
        embed = model.forward_embedding(data)
        embedding_arr = embed.detach().cpu().numpy()
        embedding.append(embedding_arr)
        label_arr = target.cpu().numpy()
        label.append(label_arr)
    embedding_np = np.concatenate(embedding, 0)
    print( embedding_np.shape)
    label_np = np.concatenate(label, 0)
    class_embedding = []
    intra_variation = []
    class_mean = []
    global_mean = np.mean(embedding_np, 0, keepdims=True)
    class_weights = np.zeros((num_class, 512))
    corr_intra = []
    corr_inter = []
    for k in range(num_class):
        tmp_index = [i for i in range(len(label_np)) if int(label_np[i]) ==k]
        class_embedding.append(embedding_np[tmp_index])
        class_mean.append(np.mean(embedding_np[tmp_index], 0))
        class_weights[k] = class_mean[k]
        corr_intra.append(np.matmul((embedding_np[tmp_index] - np.mean(embedding_np[tmp_index], 0, keepdims=True)).transpose(), (embedding_np[tmp_index] - np.mean(embedding_np[tmp_index], 0, keepdims=True)) ))
        corr_inter.append(np.matmul((np.mean(embedding_np[tmp_index], 0, keepdims=True) - global_mean).transpose(), (np.mean(embedding_np[tmp_index], 0, keepdims=True) - global_mean) ))
    corr_intra = np.mean(np.array(corr_intra), 0)
    corr_inter = np.mean(np.array(corr_inter), 0)
    intra_v = np.matrix.trace(corr_intra * np.linalg.pinv(corr_inter, rcond=1e-6)) /num_class
    equal_norm_activation = np.std(norm(np.array(class_mean) - global_mean)) / np.mean(norm(np.array(class_mean) - global_mean))
    class_mean_matrix = np.array(class_mean) - global_mean
   # cosine_sim = cosine_similarity(class_mean_matrix, class_mean_matrix).ravel()
   # cosine_sim = np.array([i for i in cosine_sim if int(i)<1])
 #   equa_ang = np.std(cosine_sim)
 #   equa_ang_2 = np.mean(np.abs(cosine_sim + 1/(num_class-1)))
    model.train()
    return equal_norm_activation, class_weights/ np.linalg.norm(class_weights, ord='fro'), 0, 0, intra_v

def weight_feature(weight, num_class=64):
    weight_norm = np.linalg.norm(weight, ord=2, axis=1, keepdims=False).ravel()
    equinorm_weight = np.std(weight_norm) / np.mean(weight_norm)
    normalized_weight =  weight / np.linalg.norm(weight, ord='fro')
    weight_n = weight - np.mean(weight, 0)
    cosine_sim = cosine_similarity(weight_n, weight_n).ravel()
    cosine_sim = np.array([i for i in cosine_sim if int(i)<1])
    equa_ang_w = np.std(cosine_sim)
    equa_ang_w_2 = np.mean(np.abs(cosine_sim + 1/(num_class-1)))
    return equinorm_weight, normalized_weight, equa_ang_w, equa_ang_w_2


