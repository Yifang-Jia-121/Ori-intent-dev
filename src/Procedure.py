import numpy as np
import torch

import utils
import sampler
import world


def BPR_train_original(dataset, recommend_model, bpr, epoch):
    Recmodel = recommend_model
    Recmodel.forecast = False
    Recmodel.train()
    S = sampler.Sample_interaction(dataset, world.ng_num)

    users = torch.Tensor(S[:, 0]).long().to(world.device)
    posItems = torch.Tensor(S[:, 1]).long().to(world.device)
    negItems = torch.Tensor(S[:, 2]).long().to(world.device)
    posTids = torch.Tensor(S[:, 3]).long().to(world.device)
    users, posItems, negItems, posTids = utils.shuffle(users, posItems, negItems, posTids)
    batch_size = world.config['bpr_batch_size']
    total_batch = (len(users) + batch_size - 1) // batch_size
    aver_loss = 0.
    for (batch_i,
         (batch_users,
          batch_pos,
          batch_neg,
          batch_Tid)) in enumerate(utils.minibatch(users,
                                                   posItems,
                                                   negItems,
                                                   posTids,
                                                   batch_size=world.config['bpr_batch_size'])):
        cri = bpr.stepOneBatch(batch_users, batch_pos, batch_neg, batch_Tid)
        aver_loss += cri
    bpr.stepOneEpoch()
    aver_loss = aver_loss / total_batch
    return aver_loss


def test_one_batch(X):
    sorted_items = X[0].numpy()
    groundTrue = X[1]
    r = utils.getLabel(groundTrue, sorted_items)
    pre, recall, hr, ndcg = [], [], [], []
    for k in world.topks:
        ret = utils.RecallPrecision_ATk(groundTrue, r, k)
        pre.append(ret['precision'])
        recall.append(ret['recall'])
        hr.append(utils.HR_ATk(groundTrue, r, k))
        ndcg.append(utils.NDCGatK_r(groundTrue, r, k))
    return {'recall': np.array(recall),
            'precision': np.array(pre),
            'hr': np.array(hr),
            'ndcg': np.array(ndcg)}

def Test(dataset, Recmodel, cold=False, satisfication=False,
         counterfactual_k=None):
    u_batch_size = world.config['test_u_batch_size']
    if cold:
        testDict: dict = dataset.coldTestDict
    elif satisfication:
        testDict: dict = dataset.satisfactoryTestDict
    else:
        testDict: dict = dataset.testDict

    Recmodel.forecast = True
    item_counts = dataset.item_counts
    Recmodel = Recmodel.eval()
    max_K = max(world.topks)
    results = {'hr': np.zeros(len(world.topks)),
               'ndcg': np.zeros(len(world.topks))}
    with torch.no_grad():
        users = list(testDict.keys())
        try:
            assert u_batch_size <= len(users) / 10
        except AssertionError:
            print(f"test_u_batch_size is too big for this dataset, try a small one {len(users) // 10}")
        users_list = []
        rating_list = []
        groundTrue_list = []
        total_batch = (len(users) + u_batch_size - 1) // u_batch_size
        # Recmodel.save_all_ratings()
        for batch_users in utils.minibatch(users, batch_size=u_batch_size):
            allPos = dataset.getUserPosItems(batch_users, False)
            groundTrue = [testDict[u] for u in batch_users]
            batch_users_gpu = torch.as_tensor(batch_users, dtype=torch.long, device=world.device)

            rating = Recmodel.getUsersRating(
                batch_users_gpu, counterfactual_k=counterfactual_k)
            exclude_index = []
            exclude_items = []
            for range_i, items in enumerate(allPos):
                exclude_index.extend([range_i] * len(items))
                exclude_items.extend(items)
            rating[exclude_index, exclude_items] = -(1 << 10)
            _, rating_K = torch.topk(rating, k=max_K)
            del rating
            users_list.append(batch_users)
            rating_list.append(rating_K.cpu())
            groundTrue_list.append(groundTrue)
        assert total_batch == len(users_list)
        X = zip(rating_list, groundTrue_list)
        pre_results = []
        for x in X:
            pre_results.append(test_one_batch(x))
        for result in pre_results:
            results['hr'] += result['hr']
            results['ndcg'] += result['ndcg']
        results['hr'] /= float(len(users))
        results['ndcg'] /= float(len(users))

        for i, k in enumerate(world.topks):
            results[f'hr@{k}'] = results['hr'][i]
            results[f'ndcg@{k}'] = results['ndcg'][i]
        del results['hr'], results['ndcg']
        all_rating = torch.cat(rating_list, dim=0).cpu().numpy()
        for k in world.topks:
            ret = utils.diversity_at_k(all_rating, item_counts, dataset.niche_items, k)
            results[f'novelty@{k}'] = ret['novelty']
            results[f'niche_rate@{k}'] = ret['niche_rate']
        # print(results)
        return results


def Evaluate(dataset, Recmodel, epoch, cold=False, w=None,
             counterfactual_k=None, niche_items=None):
    u_batch_size = world.config['test_u_batch_size']
    valDict: dict = dataset.valDict
    Recmodel = Recmodel.eval()
    max_K = max(world.topks)
    item_counts = dataset.item_counts
    if niche_items is None:
        niche_items = dataset.niche_items
    Recmodel.forecast = True
    results =  {
               'hr': np.zeros(len(world.topks)),
               'ndcg': np.zeros(len(world.topks))}
    with torch.no_grad():
        users = list(valDict.keys())
        try:
            assert u_batch_size <= len(users) / 10
        except AssertionError:
            print(f"test_u_batch_size is too big for this dataset, try a small one {len(users) // 10}")
        users_list = []
        rating_list = []
        groundTrue_list = []
        total_batch = (len(users) + u_batch_size - 1) // u_batch_size
        # Recmodel.save_all_ratings()
        for batch_users in utils.minibatch(users, batch_size=u_batch_size):
            allPos = dataset.getUserPosItems(batch_users)
            groundTrue = [valDict[u] for u in batch_users]
            batch_users_gpu = torch.as_tensor(batch_users, dtype=torch.long, device=world.device)

            rating = Recmodel.getUsersRating(
                batch_users_gpu, counterfactual_k=counterfactual_k)
            exclude_index = []
            exclude_items = []
            for range_i, items in enumerate(allPos):
                exclude_index.extend([range_i] * len(items))
                exclude_items.extend(items)
            rating[exclude_index, exclude_items] = -(1 << 10)
            _, rating_K = torch.topk(rating, k=max_K)
            del rating
            users_list.append(batch_users)
            rating_list.append(rating_K.cpu())
            groundTrue_list.append(groundTrue)
        assert total_batch == len(users_list)
        X = zip(rating_list, groundTrue_list)
        pre_results = []
        for x in X:
            pre_results.append(test_one_batch(x))
        for result in pre_results:
            results['hr'] += result['hr']
            results['ndcg'] += result['ndcg']
        results['hr'] /= float(len(users))
        results['ndcg'] /= float(len(users))

        for i, k in enumerate(world.topks):
            results[f'hr@{k}'] = results['hr'][i]
            results[f'ndcg@{k}'] = results['ndcg'][i]
        del results['hr'], results['ndcg']
        # del results['recall'], results['precision'], results['hr'], results['ndcg']
        all_rating = torch.cat(rating_list, dim=0).cpu().numpy()
        for k in world.topks:
            ret = utils.diversity_at_k(all_rating, item_counts, niche_items, k)
            results[f'novelty@{k}'] = ret['novelty']
            results[f'niche_rate@{k}'] = ret['niche_rate']
        return results


def EvaluateKs(dataset, Recmodel, counterfactual_ks, niche_items=None):
    counterfactual_ks = list(dict.fromkeys(counterfactual_ks))
    if not counterfactual_ks:
        raise ValueError('counterfactual_ks must not be empty')

    u_batch_size = world.config['test_u_batch_size']
    val_dict = dataset.valDict
    Recmodel = Recmodel.eval()
    Recmodel.forecast = True
    max_K = max(world.topks)
    item_counts = dataset.item_counts
    if niche_items is None:
        niche_items = dataset.niche_items

    results_by_k = {}
    rating_lists = {counterfactual_k: []
                    for counterfactual_k in counterfactual_ks}
    ground_true_list = []
    users = list(val_dict.keys())

    with torch.no_grad():
        for batch_users in utils.minibatch(users, batch_size=u_batch_size):
            all_pos = dataset.getUserPosItems(batch_users)
            ground_true = [val_dict[u] for u in batch_users]
            batch_users_gpu = torch.as_tensor(
                batch_users, dtype=torch.long, device=world.device)
            ratings = Recmodel.getUsersRatings(
                batch_users_gpu, counterfactual_ks)

            exclude_index = []
            exclude_items = []
            for range_i, items in enumerate(all_pos):
                exclude_index.extend([range_i] * len(items))
                exclude_items.extend(items)

            for counterfactual_k, rating in zip(counterfactual_ks, ratings):
                rating[exclude_index, exclude_items] = -(1 << 10)
                _, rating_K = torch.topk(rating, k=max_K)
                rating_lists[counterfactual_k].append(rating_K.cpu())
            ground_true_list.append(ground_true)

    for counterfactual_k in counterfactual_ks:
        results = {
            'hr': np.zeros(len(world.topks)),
            'ndcg': np.zeros(len(world.topks)),
        }
        for rating_batch, ground_true in zip(
                rating_lists[counterfactual_k], ground_true_list):
            batch_results = test_one_batch((rating_batch, ground_true))
            results['hr'] += batch_results['hr']
            results['ndcg'] += batch_results['ndcg']

        results['hr'] /= float(len(users))
        results['ndcg'] /= float(len(users))
        for index, topk in enumerate(world.topks):
            results[f'hr@{topk}'] = results['hr'][index]
            results[f'ndcg@{topk}'] = results['ndcg'][index]
        del results['hr'], results['ndcg']

        all_rating = torch.cat(
            rating_lists[counterfactual_k], dim=0).numpy()
        for topk in world.topks:
            diversity = utils.diversity_at_k(
                all_rating, item_counts, niche_items, topk)
            results[f'novelty@{topk}'] = diversity['novelty']
            results[f'niche_rate@{topk}'] = diversity['niche_rate']
        results_by_k[counterfactual_k] = results

    return results_by_k
