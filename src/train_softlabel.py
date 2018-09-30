# -*- coding: utf-8 -*-

import argparse
import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import network
import loss
import pre_process as prep
import torch.utils.data as util_data
import lr_schedule
from data_list import ImageList
from torch.autograd import Variable
import time

optim_dict = {"SGD": optim.SGD}


def image_classification_predict(loader, model, test_10crop=True, gpu=True):
    start_test = True
    if test_10crop:
        iter_test = [iter(loader['test' + str(i)]) for i in xrange(10)]
        for i in xrange(len(loader['test0'])):
            data = [iter_test[j].next() for j in xrange(10)]
            inputs = [data[j][0] for j in xrange(10)]
            if gpu:
                for j in xrange(10):
                    inputs[j] = Variable(inputs[j].cuda())
            else:
                for j in xrange(10):
                    inputs[j] = Variable(inputs[j])
            outputs = []
            for j in xrange(10):
                outputs.append(model(inputs[j]))
            outputs = sum(outputs)
            if start_test:
                all_output = outputs.data.float()
                start_test = False
            else:
                all_output = torch.cat((all_output, outputs.data.float()), 0)
    else:
        iter_val = iter(loader["test"])
        for i in xrange(len(loader['test'])):
            data = iter_val.next()
            inputs = data[0]
            if gpu:
                inputs = Variable(inputs.cuda())
            else:
                inputs = Variable(inputs)
            outputs = model(inputs)
            if start_test:
                all_output = outputs.data.cpu().float()
                start_test = False
            else:
                all_output = torch.cat((all_output, outputs.data.cpu().float()), 0)
    _, predict = torch.max(all_output, 1)
    return all_output, predict


def image_classification_test(loader, model, test_10crop=True, gpu=True):
    start_test = True
    if test_10crop:
        iter_test = [iter(loader['test' + str(i)]) for i in xrange(10)]
        for i in xrange(len(loader['test0'])):
            data = [iter_test[j].next() for j in xrange(10)]
            inputs = [data[j][0] for j in xrange(10)]
            labels = data[0][1]
            if gpu:
                for j in xrange(10):
                    inputs[j] = Variable(inputs[j].cuda())
                labels = Variable(labels.cuda())
            else:
                for j in xrange(10):
                    inputs[j] = Variable(inputs[j])
                labels = Variable(labels)
            outputs = []
            for j in xrange(10):
                outputs.append(model(inputs[j]))
            outputs = sum(outputs)
            if start_test:
                all_output = outputs.data.float()
                all_label = labels.data.float()
                start_test = False
            else:
                all_output = torch.cat((all_output, outputs.data.float()), 0)
                all_label = torch.cat((all_label, labels.data.float()), 0)
    else:
        model.train(False)
        iter_test = iter(loader["test"])
        for i in xrange(len(loader['test'])):
            data = iter_test.next()
            inputs = data[0]
            labels = data[1]
            if gpu:
                inputs = Variable(inputs.cuda())
                labels = Variable(labels.cuda())
            else:
                inputs = Variable(inputs)
                labels = Variable(labels)
            outputs = model(inputs)
            if start_test:
                all_output = outputs.data.float()
                all_label = labels.data.float()
                start_test = False
            else:
                all_output = torch.cat((all_output, outputs.data.float()), 0)
                all_label = torch.cat((all_label, labels.data.float()), 0)

    _, predict = torch.max(all_output, 1)
    accuracy = torch.sum(torch.squeeze(predict).float() == all_label) / float(all_label.size()[0])
    return accuracy


def image_classification_residual_test(loader, base_network,classifier_layer,residual_layer1,
                                       test_10crop=True, gpu=True):
    start_test = True
    if test_10crop:
        iter_test = [iter(loader['test' + str(i)]) for i in xrange(10)]
        for i in xrange(len(loader['test0'])):
            data = [iter_test[j].next() for j in xrange(10)]
            inputs = [data[j][0] for j in xrange(10)]
            labels = data[0][1]
            if gpu:
                for j in xrange(10):
                    inputs[j] = Variable(inputs[j].cuda())
                labels = Variable(labels.cuda())
            else:
                for j in xrange(10):
                    inputs[j] = Variable(inputs[j])
                labels = Variable(labels)
            outputs = []
            for j in xrange(10):
                features_base=base_network(inputs[j])
                features_residual1=residual_layer1(features_base)
                residual_total1=features_base+features_residual1
                outputs.append(classifier_layer(residual_total1))

            outputs = sum(outputs)
            if start_test:
                all_output = outputs.data.float()
                all_label = labels.data.float()
                start_test = False
            else:
                all_output = torch.cat((all_output, outputs.data.float()), 0)
                all_label = torch.cat((all_label, labels.data.float()), 0)
    else:
        iter_test = iter(loader["test"])
        for i in xrange(len(loader['test'])):
            data = iter_test.next()
            inputs = data[0]
            labels = data[1]
            if gpu:
                inputs = Variable(inputs.cuda())
                labels = Variable(labels.cuda())
            else:
                inputs = Variable(inputs)
                labels = Variable(labels)
            outputs=[]
            features_base = base_network(inputs[j])
            features_residual1 = residual_layer1(features_base)

            residual_total1 = features_base + features_residual1
            outputs.append(classifier_layer(residual_total1))

            if start_test:
                all_output = outputs.data.float()
                all_label = labels.data.float()
                start_test = False
            else:
                all_output = torch.cat((all_output, outputs.data.float()), 0)
                all_label = torch.cat((all_label, labels.data.float()), 0)

    _, predict = torch.max(all_output, 1)
    accuracy = torch.sum(torch.squeeze(predict).float() == all_label) / float(all_label.size()[0])
    return accuracy



def transfer_classification(config,modelname,taskname,probability,b,c):
    prep_dict = {}
    for prep_config in config["prep"]:
        prep_dict[prep_config["name"]] = {}
        if prep_config["type"] == "image":
            prep_dict[prep_config["name"]]["test_10crop"] = prep_config["test_10crop"]
            prep_dict[prep_config["name"]]["train"] = prep.image_train(resize_size=prep_config["resize_size"],
                                                                       crop_size=prep_config["crop_size"])
            if prep_config["test_10crop"]:
                prep_dict[prep_config["name"]]["test"] = prep.image_test_10crop(resize_size=prep_config["resize_size"],
                                                                                crop_size=prep_config["crop_size"])
            else:
                prep_dict[prep_config["name"]]["test"] = prep.image_test(resize_size=prep_config["resize_size"],
                                                                         crop_size=prep_config["crop_size"])

    ## set loss
    class_criterion = nn.CrossEntropyLoss()
    loss_config = config["loss"]
    transfer_criterion = loss.loss_dict[loss_config["name"]]
    if "params" not in loss_config:
        loss_config["params"] = {}

    dsets = {}
    dset_loaders = {}
    for data_config in config["data"]:
        dsets[data_config["name"]] = {}
        dset_loaders[data_config["name"]] = {}
        if data_config["type"] == "image":
            dsets[data_config["name"]]["train"] = ImageList(open(data_config["list_path"]["train"]).readlines(),
                                                            transform=prep_dict[data_config["name"]]["train"])
            dset_loaders[data_config["name"]]["train"] = util_data.DataLoader(dsets[data_config["name"]]["train"],
                                                                              batch_size=data_config["batch_size"][
                                                                                  "train"], shuffle=True, num_workers=4)
            if "test" in data_config["list_path"]:
                if prep_dict[data_config["name"]]["test_10crop"]:
                    for i in range(10):
                        dsets[data_config["name"]]["test" + str(i)] = ImageList(
                            open(data_config["list_path"]["test"]).readlines(),
                            transform=prep_dict[data_config["name"]]["test"]["val" + str(i)]
                            )
                        dset_loaders[data_config["name"]]["test" + str(i)] = util_data.DataLoader(
                            dsets[data_config["name"]]["test" + str(i)], batch_size=data_config["batch_size"]["test"],
                            shuffle=False, num_workers=4)
                else:
                    dsets[data_config["name"]]["test"] = ImageList(open(data_config["list_path"]["test"]).readlines(),
                                                                   transform=prep_dict[data_config["name"]]["test"])
                    dset_loaders[data_config["name"]]["test"] = util_data.DataLoader(dsets[data_config["name"]]["test"],
                                                                                     batch_size=
                                                                                     data_config["batch_size"]["test"],
                                                                                     shuffle=False, num_workers=4)
            else:
                if prep_dict[data_config["name"]]["test_10crop"]:
                    for i in range(10):
                        dsets[data_config["name"]]["test" + str(i)] = ImageList(
                            open(data_config["list_path"]["train"]).readlines(),
                            transform=prep_dict[data_config["name"]]["test"]["val" + str(i)])
                        dset_loaders[data_config["name"]]["test" + str(i)] = util_data.DataLoader(
                            dsets[data_config["name"]]["test" + str(i)], batch_size=data_config["batch_size"]["test"],
                            shuffle=False, num_workers=4)
                else:
                    dsets[data_config["name"]]["test"] = ImageList(open(data_config["list_path"]["train"]).readlines(),
                                                                   transform=prep_dict[data_config["name"]]["test"])
                    dset_loaders[data_config["name"]]["test"] = util_data.DataLoader(dsets[data_config["name"]]["test"],
                                                                                     batch_size=
                                                                                     data_config["batch_size"]["test"],
                                                                                     shuffle=False, num_workers=4)
    class_num = 31
    net_config = config["network"]
    base_network=torch.load('../save/nobottle_base_network_'+modelname+'.pkl')  #原本单gpu
    if net_config["use_bottleneck"]:
        bottleneck_layer = nn.Linear(base_network.output_num(), net_config["bottleneck_dim"])
        classifier_layer = nn.Linear(bottleneck_layer.out_features, class_num)
    else:
        classifier_layer = nn.Linear(base_network.output_num(), class_num)

    # initialization
    if net_config["use_bottleneck"]:
        bottleneck_layer.weight.data.normal_(0, 0.005)
        bottleneck_layer.bias.data.fill_(0.1)
        bottleneck_layer = nn.Sequential(bottleneck_layer, nn.ReLU(), nn.Dropout(0.5))
    classifier_layer.weight.data.normal_(0, 0.01)
    classifier_layer.bias.data.fill_(0.0)
    residual_fc1_1 = nn.Linear(base_network.output_num(), base_network.output_num())
    residual_bn1_1 = nn.BatchNorm1d(base_network.output_num())
    residual_fc1_2 = nn.Linear(base_network.output_num(), base_network.output_num())
    residual_fc1_1.weight.data.normal_(0, 0.005)
    residual_fc1_1.bias.data.fill_(0.1)
    residual_fc1_2.weight.data.normal_(0, 0.005)
    residual_fc1_2.bias.data.fill_(0.1)
    residual_layer1 = nn.Sequential(residual_fc1_1, residual_bn1_1,nn.ReLU(), residual_fc1_2)
    classifier_layer = torch.load('../save/nobottle_classifier_layer_'+modelname+'.pkl')
    use_gpu = torch.cuda.is_available()
    if use_gpu:
        classifier_layer = classifier_layer.cuda()
        base_network = base_network.cuda()
        residual_layer1 = residual_layer1.cuda()

    if net_config["use_bottleneck"]:
        parameter_list=[{"params": classifier_layer.parameters(), "lr": 10}]

    else:
        parameter_list = [
            {"params": base_network.parameters(), "lr": 1},
            {"params": classifier_layer.parameters(), "lr": 10},
            {"params": residual_layer1.parameters(), "lr": 0.1},
        ]
        kernel_muls = [2.0,  2.0]
        kernel_nums = [5, 1]
        fix_sigma_list = [None, 1.68]

    if loss_config["name"] == "DRCN":
        softmax_layer = nn.Softmax()
        if use_gpu:
            softmax_layer = softmax_layer.cuda()

    ## set optimizer
    optimizer_config = config["optimizer"]
    optimizer = optim_dict[optimizer_config["type"]](parameter_list, **(optimizer_config["optim_params"]))
    param_lr = []
    for param_group in optimizer.param_groups:
        param_lr.append(param_group["lr"])
    schedule_param = optimizer_config["lr_param"]
    lr_scheduler = lr_schedule.schedule_dict[optimizer_config["lr_type"]]

    ## train
    len_train_source = len(dset_loaders["source"]["train"]) - 1
    len_train_target = len(dset_loaders["target"]["train"]) - 1

    lensource = len(dsets['source']['test0'])
    lentarget=len(dsets['target']['test0'])
    print lensource
    print lentarget
    saveold=[]
    savenew=[]
    count=0
    for i in range(config["num_iterations"]+1):
        ## test in the train
        if i % config["test_interval"] == 0:
        # if False:
            base_network.train(False)
            classifier_layer.train(False)
            if net_config["use_bottleneck"]:
                bottleneck_layer.train(False)
                residual_layer1.train(False)
                residual_layer2.train(False)
                base_network=nn.Sequential(base_network)
                classifier_layer=nn.Sequential(classifier_layer)
                residual_layer1=nn.Sequential(residual_layer1)
                residual_layer2=nn.Sequential(residual_layer2)

                accuracy=image_classification_residual_test(dset_loaders["target"],
                                                base_network,classifier_layer,residual_layer1,residual_layer2,
                                                test_10crop=prep_dict["target"]["test_10crop"], gpu=use_gpu)

                print 'iter{0}accuracy:'.format(i),accuracy
            else:
                residual_layer1.train(False)
                base_network = nn.Sequential(base_network)
                classifier_layer = nn.Sequential(classifier_layer)
                residual_layer1 = nn.Sequential(residual_layer1)
                accuracy = image_classification_residual_test(dset_loaders["target"],
                                                              base_network, classifier_layer, residual_layer1,
                                                              test_10crop=prep_dict["target"]["test_10crop"],
                                                              gpu=use_gpu)
                print 'iter{0}accuracy:'.format(i), accuracy

        base_network.train(True)
        classifier_layer.train(True)
        residual_layer1.train(True)

        optimizer = lr_scheduler(param_lr, optimizer, i, **schedule_param)
        optimizer.zero_grad()
        if i % len_train_source == 0:
            iter_source = iter(dset_loaders["source"]["train"])
        if i % len_train_target == 0:
            iter_target = iter(dset_loaders["target"]["train"])
        inputs_source, labels_source = iter_source.next()
        inputs_target, labels_target = iter_target.next()
        if use_gpu:
            inputs_source, inputs_target, labels_source = Variable(inputs_source).cuda(), Variable(
                inputs_target).cuda(), Variable(labels_source).cuda()
        else:
            inputs_source, inputs_target, labels_source = Variable(inputs_source), Variable(inputs_target), Variable(
                labels_source)

        inputs = torch.cat((inputs_source, inputs_target), dim=0)
        features_base = base_network(inputs)

        features_residual1 = residual_layer1(features_base)
        residual_total1=features_base+features_residual1
        outputs = classifier_layer(features_base)

        classifier_loss = class_criterion(outputs.narrow(0, 0, inputs.size(0) / 2), labels_source)

        if loss_config["name"] == "DRCN":
            softmax_out = softmax_layer(outputs)
            output_residual=classifier_layer(residual_total1)
            softmax_residual=softmax_layer(output_residual)

            transfer_loss = transfer_criterion([features_base.narrow(0, 0, features_base.size(0) / 2),
                                                softmax_out.narrow(0, 0, softmax_out.size(0) / 2)
                                                ],
                                               [residual_total1.narrow(0, features_base.size(0) / 2, features_base.size(0) / 2),
                                                softmax_residual.narrow(0, softmax_residual.size(0) / 2,softmax_residual.size(0) / 2)
                                                ],kernel_muls,kernel_nums,fix_sigma_list,
                                               **loss_config["params"])
            outputs_result = []
            outputs_result.append(output_residual.narrow(0, residual_total1.size(0) / 2, residual_total1.size(0) / 2))
            outputs_result = sum(outputs_result)
            all_output = outputs_result.data.float()
            all_label = labels_source.data.float()

            category_num = softmax_out.size(1)
            import  copy
            _, predict = torch.max(all_output, 1)

            while (len(savenew) != category_num):
                savenew.append(0)

            softmax_target=softmax_residual.narrow(0, softmax_residual.size(0) / 2,softmax_residual.size(0) / 2).cpu().data.numpy()
            for u in range(len(softmax_target)):
                count += 1
                for uu in range(len(softmax_target[0])):
                    savenew[uu]+=softmax_target[u][uu]
                if(count % lentarget==0):
                    saveold = copy.deepcopy(savenew)
                    for p in range(len(savenew)):
                        savenew[p]=0
            sum_conditional_loss = 0
            for k in xrange(category_num):
                source_index = []
                for index, source_k in enumerate(all_label):
                    if (source_k == k):
                        source_index.append(index)
                conditional_loss1 = 0
                if (len(source_index) > 0):
                    source_concat1 = None
                    target_concat1 = None
                    for index in xrange(len(source_index)):
                        if source_concat1 is None:
                            source_concat1 = features_base.narrow(0, source_index[index], 1)
                        else:
                            source_concat1 = torch.cat((source_concat1,
                                                        features_base.narrow(0, source_index[index], 1)), 0)
                    target_softmax=softmax_residual.narrow(0, softmax_residual.size(0) / 2,softmax_residual.size(0) / 2)
                    prob_total=0
                    for z in range(target_softmax.size(0)):
                        prob=target_softmax.data.float()[z][k]
                        if(prob>probability):
                            prob_total+=prob
                            if target_concat1 is None:
                                target_concat1 = residual_total1.narrow(0,
                                                                        z + residual_total1.size(0) / 2,
                                                                        1)* prob
                            else:
                                target_concat1 = torch.cat((target_concat1,
                                                            residual_total1.narrow(0, z + residual_total1.size(0) / 2,
                                                                                   1)* prob  ), 0)

                    if(target_concat1 is not None):
                        conditional_loss1 = loss.PROB(source_concat1, target_concat1,prob_total)
                c1=1.0
                if (count >= lentarget):
                    c1 = float(saveold[k] / float(max(saveold)))
                sum_conditional_loss += c1 * conditional_loss1


        total_loss = classifier_loss+ b*transfer_loss +c * sum_conditional_loss
        total_loss.backward()
        optimizer.step()


if __name__ == "__main__":
    print 'hello'
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'

    probability = 0.4
    b,c=0.1,0.05
    print 'pro=',probability,'b=',b,'c=',c
    sourcelist = 'amazon_list.txt'
    targetlist = 'webcam10_list.txt'
    config = {}
    config["num_iterations"] = 11000
    config["test_interval"] = 500
    config["prep"] = [{"name": "source", "type": "image", "test_10crop": True, "resize_size": 256, "crop_size": 224},
                      {"name": "target", "type": "image", "test_10crop": True, "resize_size": 256, "crop_size": 224}]
    config["loss"] = {"name": "DRCN", "trade_off": 0.2}
    config["data"] = [{"name": "source", "type": "image", "list_path": {"train": "../data/office/" + sourcelist},
                       "batch_size": {"train": 36, "test": 4}},
                      {"name": "target", "type": "image", "list_path": {"train": "../data/office/" + targetlist},
                       "batch_size": {"train": 36, "test": 4}}]
    config["network"] = {"name": "ResNet50", "use_bottleneck": False, "bottleneck_dim": 256}
    config["optimizer"] = {"type": "SGD",
                           "optim_params": {"lr": 1.0, "momentum": 0.9, "weight_decay": 0.0005, "nesterov": True},
                           "lr_type": "inv", "lr_param": {"init_lr": 0.0003, "gamma": 0.0003, "power": 0.75}}
    transfer_classification(config, 'res50_amazon', sourcelist + ' to ' + targetlist, probability,b,c)
