import collections
import numpy as np
from args import args
import torch
from torch import nn
from sklearn.metrics import precision_score, f1_score, recall_score, matthews_corrcoef
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def meta_loop(loader,
              model,
              inner_step,
              inner_lr,
              optimizer,
              attenuator,
              is_train=True,
              is_save=False,
              name=None):
    meta_loss = []
    criterion = nn.CrossEntropyLoss()
    support_loss_list = []
    previous_grads = None
    original_weight = collections.OrderedDict(model.named_parameters())
    gradient_changes = []
    fast_weight = collections.OrderedDict(model.named_parameters())

    for i in range(args.num_block):
        fast_weight.pop(f'layers.{i}.AddNorm1.normalize.weight', None)
        fast_weight.pop(f'layers.{i}.AddNorm2.normalize.weight', None)
        fast_weight.pop(f'layers.{i}.AddNorm1.normalize.bias', None)
        fast_weight.pop(f'layers.{i}.AddNorm2.normalize.bias', None)

    for task_ids, task in enumerate(loader):
        fast_weight = collections.OrderedDict(model.named_parameters())
        for i in range(args.num_block):
            fast_weight.pop(f'layers.{i}.AddNorm1.normalize.weight', None)
            fast_weight.pop(f'layers.{i}.AddNorm2.normalize.weight', None)
            fast_weight.pop(f'layers.{i}.AddNorm1.normalize.bias', None)
            fast_weight.pop(f'layers.{i}.AddNorm2.normalize.bias', None)
        all_support_input_ids, all_support_mask, all_support_label = [], [], []
        for support_pairs in task['support_set']:
            support_input_ids = support_pairs['sequence'].to(device)
            support_mask = support_pairs['attention_mask'].to(device)
            support_label = support_pairs['label'].to(device)

            all_support_input_ids.append(support_input_ids)
            all_support_mask.append(support_mask)
            all_support_label.append(support_label)
        all_support_input_ids = torch.stack(all_support_input_ids, dim=0).squeeze(dim=1)
        all_support_mask = torch.stack(all_support_mask, dim=0).squeeze(dim=1)
        all_support_label = torch.stack(all_support_label, dim=0).squeeze(dim=1)

        for inner_epoch in range(inner_step):
            support_logit = model.functional_forward(all_support_input_ids, all_support_mask, fast_weight)
            support_loss = criterion(support_logit, all_support_label)
            support_loss_list.append(support_loss)

            # Attenuator
            if inner_epoch == 0:
                grads = torch.autograd.grad(support_loss, fast_weight.values(), allow_unused=True)
                current_grads = grads
                if previous_grads is not None:
                    gradient_change = torch.norm(torch.cat(
                        [g1.view(-1) - g2.view(-1) for (g1, g2)
                         in zip(current_grads, previous_grads)]), p=2)
                    gradient_changes.append(gradient_change.item())
                previous_grads = current_grads
                layer_wise_mean_grads = []
                for i in range(len(grads)):
                    layer_wise_mean_grads.append(grads[i].mean())
                layer_wise_mean_grads = torch.stack(layer_wise_mean_grads)
                gamma_params = attenuator(layer_wise_mean_grads)
                attenuated_fast_weight = collections.OrderedDict()
                for ((name, param), gamma) in zip(fast_weight.items(), gamma_params):
                    attenuated_fast_weight[name] = gamma * param
                fast_weight = collections.OrderedDict((name, param - inner_lr * grad)
                                                      for ((name, param), grad) in
                                                      zip(attenuated_fast_weight.items(), grads))
            else:
                grads = torch.autograd.grad(support_loss, fast_weight.values(), allow_unused=True)
                current_grads = grads
                if previous_grads is not None:
                    gradient_change = torch.norm(torch.cat(
                        [g1.view(-1) - g2.view(-1) for (g1, g2)
                         in zip(current_grads, previous_grads)]), p=2)
                    gradient_changes.append(gradient_change.item())
                previous_grads = current_grads
                fast_weight = collections.OrderedDict((name, param - inner_lr * grad)
                                                      for ((name, param), grad) in
                                                      zip(fast_weight.items(), grads))

    for i in range(args.num_block):
        fast_weight[f'layers.{i}.AddNorm1.normalize.weight'] = original_weight[
            f'layers.{i}.AddNorm1.normalize.weight']
        fast_weight[f'layers.{i}.AddNorm2.normalize.weight'] = original_weight[
            f'layers.{i}.AddNorm2.normalize.weight']
        fast_weight[f'layers.{i}.AddNorm1.normalize.bias'] = original_weight[
            f'layers.{i}.AddNorm1.normalize.bias']
        fast_weight[f'layers.{i}.AddNorm2.normalize.bias'] = original_weight[
            f'layers.{i}.AddNorm2.normalize.bias']
    if is_save:
        torch.save(fast_weight, f'model/{name}.pth')

    for task_ids, task in enumerate(loader):
        all_query_input_ids, all_query_mask, all_query_label = [], [], []
        for query_pairs in task['query_set']:
            query_input_ids = query_pairs['sequence'].to(device)
            query_mask = query_pairs['attention_mask'].to(device)
            query_label = query_pairs['label'].to(device)
            all_query_input_ids.append(query_input_ids)
            all_query_mask.append(query_mask)
            all_query_label.append(query_label)
        all_query_input_ids = torch.stack(all_query_input_ids, dim=0).squeeze(dim=1)
        all_query_mask = torch.stack(all_query_mask, dim=0).squeeze(dim=1)
        all_query_label = torch.stack(all_query_label, dim=0).squeeze(dim=1)

        query_logit = model.functional_forward(all_query_input_ids, all_query_mask, fast_weight)
        query_loss = criterion(query_logit, all_query_label)

        epsilon = args.epsilon
        eta = args.eta
        query_loss = - query_loss ** eta * torch.log(torch.clamp(1 - query_loss + epsilon, min=epsilon))

        z_star, z_logit = model.bottleneck(all_query_input_ids, all_query_mask)
        prior_distribution = torch.distributions.Normal(0, 1)
        regularization = args.beta * torch.distributions.kl_divergence(z_star, prior_distribution)

        query_loss = torch.mean(query_loss + regularization, dim=0)
        meta_loss.append(query_loss)

    optimizer.zero_grad()
    meta_loss = torch.stack(meta_loss).sum()
    if is_train:
        meta_loss.backward()
        optimizer.step()

    return meta_loss


def meta_loop_test(dataloader,
                   model,
                   inner_step,
                   inner_lr,
                   is_save=True):
    meta_loss = []
    meta_acc, meta_precision, meta_f1, meta_recall, meta_mcc = [], [], [], [], []
    criterion = nn.CrossEntropyLoss()
    original_weight = collections.OrderedDict(model.named_parameters())
    fast_weight = collections.OrderedDict(model.named_parameters())
    for i in range(args.num_block):
        fast_weight.pop(f'layers.{i}.AddNorm1.normalize.weight', None)
        fast_weight.pop(f'layers.{i}.AddNorm2.normalize.weight', None)
        fast_weight.pop(f'layers.{i}.AddNorm1.normalize.bias', None)
        fast_weight.pop(f'layers.{i}.AddNorm2.normalize.bias', None)
    for task_idx, task in enumerate(dataloader):
        all_support_input_ids, all_support_mask, all_support_label = [], [], []
        all_query_input_ids, all_query_mask, all_query_label = [], [], []
        for support_pairs in task['support_set']:
            support_input_ids = support_pairs['sequence'].to(device)
            support_mask = support_pairs['attention_mask'].to(device)
            support_label = support_pairs['label'].to(device)

            all_support_input_ids.append(support_input_ids)
            all_support_mask.append(support_mask)
            all_support_label.append(support_label)
        for query_pairs in task['query_set']:
            query_input_ids = query_pairs['sequence'].to(device)
            query_mask = query_pairs['attention_mask'].to(device)
            query_label = query_pairs['label'].to(device)
            all_query_input_ids.append(query_input_ids)
            all_query_mask.append(query_mask)
            all_query_label.append(query_label)
        all_support_input_ids = torch.stack(all_support_input_ids, dim=0).squeeze(dim=1)
        all_support_mask = torch.stack(all_support_mask, dim=0).squeeze(dim=1)
        all_support_label = torch.stack(all_support_label, dim=0).squeeze(dim=1)
        all_query_input_ids = torch.stack(all_query_input_ids, dim=0).squeeze(dim=1)
        all_query_mask = torch.stack(all_query_mask, dim=0).squeeze(dim=1)
        all_query_label = torch.stack(all_query_label, dim=0).squeeze(dim=1)

        for _ in range(inner_step):
            support_logit = model.functional_forward(all_support_input_ids, all_support_mask, fast_weight)
            support_loss = criterion(support_logit, all_support_label)

            grads = torch.autograd.grad(support_loss, fast_weight.values(), allow_unused=True)
            fast_weight = collections.OrderedDict((name, param - inner_lr * grad)
                                                  for ((name, param), grad) in zip(fast_weight.items(), grads))

        for i in range(args.num_block):
            fast_weight[f'layers.{i}.AddNorm1.normalize.weight'] = original_weight[
                f'layers.{i}.AddNorm1.normalize.weight']
            fast_weight[f'layers.{i}.AddNorm2.normalize.weight'] = original_weight[
                f'layers.{i}.AddNorm2.normalize.weight']
            fast_weight[f'layers.{i}.AddNorm1.normalize.bias'] = original_weight[
                f'layers.{i}.AddNorm1.normalize.bias']
            fast_weight[f'layers.{i}.AddNorm2.normalize.bias'] = original_weight[
                f'layers.{i}.AddNorm2.normalize.bias']
        if is_save:
            torch.save(fast_weight, 'model/AB.pth')

        query_logit = model.functional_forward(all_query_input_ids, all_query_mask, fast_weight)
        query_prediction = torch.max(query_logit, dim=1)[1]

        query_loss = criterion(query_logit, all_query_label)

        labels = all_query_label.cpu().detach()
        predictions = query_prediction.cpu().detach()

        query_acc = torch.eq(all_query_label, query_prediction).sum() / len(all_query_label)
        precision = precision_score(labels, predictions)
        f1 = f1_score(labels, predictions)
        mcc = matthews_corrcoef(labels, predictions)
        recall = recall_score(labels, predictions)

        meta_loss.append(query_loss)
        meta_precision.append(precision)
        meta_f1.append(f1)
        meta_recall.append(recall)
        meta_mcc.append(mcc)
        meta_acc.append(query_acc.data.cpu().numpy())

    meta_loss_list = meta_loss
    meta_acc_list = meta_acc
    meta_recall_list = meta_recall
    meta_precision_list = meta_precision
    meta_f1_list = meta_f1
    meta_mcc_list = meta_mcc

    meta_loss = torch.stack(meta_loss).mean()
    meta_acc = np.mean(meta_acc)
    meta_precision = np.mean(meta_precision)
    meta_f1 = np.mean(meta_f1)
    meta_mcc = np.mean(meta_mcc)
    meta_recall = np.mean(meta_recall)

    return meta_loss, meta_acc, meta_precision, meta_recall, meta_f1, meta_mcc, meta_loss_list, \
        meta_acc_list, meta_recall_list, meta_precision_list, meta_f1_list, meta_mcc_list

