import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import precision_recall_curve, mean_squared_error, mean_absolute_error, r2_score, mean_squared_log_error, log_loss
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, confusion_matrix,average_precision_score, ConfusionMatrixDisplay, roc_curve, auc
import math
from matplotlib import pyplot as plt
import seaborn as sns


# def Prob2Score(prob,basePoint,PDO):
#     y = np.log(prob/(1 - prob))
#     if y == math.inf: y=1000
#     if y == -math.inf: y=-1000
#     tmp = int(basePoint - PDO * y)
#     score = max(350,min(950,tmp))
#     return score
def Prob2Score(prob,basePoint,PDO):
    y = np.log(prob/(1 - prob))
    if y == math.inf: y=1000
    if y == -math.inf: y=-1000
    score = int(basePoint + PDO * y)
    # score = max(350,min(950,tmp))
    # print(y, score, prob)
    return score

def lgm_train_plot(model, prcss, metrics, max_num_features=20, imp_type='gain'):
    import lightgbm as lgb
    from matplotlib import pyplot as plt
    i = 0
    while i < len(metrics):
        fig, ((ax1, ax2, ax3)) = plt.subplots(nrows=1, ncols=3, figsize=(15,3))
        lgb.plot_metric(prcss, metric=metrics[i] if isinstance(metrics[i], str) else metrics[i].__name__, ax=ax1)
        if i+1 < len(metrics):
            lgb.plot_metric(prcss, metric=metrics[i+1] if isinstance(metrics[i+1], str) else metrics[i+1].__name__, ax=ax2)
        if i+2 < len(metrics):
            lgb.plot_metric(prcss, metric=metrics[i+2] if isinstance(metrics[i+1], str) else metrics[i+2].__name__ , ax=ax3)
        i += 3
        plt.figure()
    lgb.plot_importance(model,max_num_features=max_num_features,importance_type=imp_type, figsize=(3,3))
    plt.title('feature_importance')
    plt.show()
def calc_ks(data, target, score, group_num, reverse=False, plot=True):

    data = data.dropna(subset = [target, score])
    data['bad'] = data[target]
    data['score'] = data[score]
    data['good'] = 1 - data.bad
    data['bucket'] = pd.qcut(data.score, group_num, duplicates = 'drop')
    grouped = data.groupby('bucket')  
    agg1 = pd.DataFrame()
    agg1['min_scr'] =  grouped.score.min()
    agg1['max_scr'] = grouped.score.max()
    agg1['bads'] = grouped.bad.sum()
    agg1['goods'] = grouped.good.sum()
    agg1['total'] = agg1.bads + agg1.goods   

    agg2 = (agg1.sort_values(by = 'min_scr', ascending = reverse)).reset_index(drop = True)
    agg2['odds'] = (agg2.goods / agg2.bads).apply('{0:.2f}'.format)
    agg2['bad_rate'] = (agg2.bads / agg2.total).apply('{0:.2%}'.format)
    agg2['bad_cum'] = (agg2.bads / agg2.bads.sum()).cumsum()*100
    agg2['good_cum'] = (agg2.goods / agg2.goods.sum()).cumsum()*100
    agg2['bad_cum_pct'] = (agg2.bads / agg2.bads.sum()).cumsum().apply('{0:.2%}'.format)
    agg2['good_cum_pct'] = (agg2.goods / agg2.goods.sum()).cumsum().apply('{0:.2%}'.format)
    agg2['ks'] = np.round(((agg2.bads / agg2.bads.sum()).cumsum() - (agg2.goods / agg2.goods.sum()).cumsum()), 4) * 100
    agg2['tile0'] = range(1, len(agg2.ks) + 1)
    agg2['pop'] = 1.0*agg2['tile0']/len(agg2['tile0']) 
    flag = lambda x: '<----' if x == agg2.ks.max() else ''
    agg2['max'] = agg2.ks.apply(flag)
    ks_value = agg2.ks.max()
    ks_pop = agg2['pop'][agg2.ks.idxmax()]
    if plot:
        print ('ks_value is ' + str(np.round(ks_value, 4)) + ' at pop = ' + str(np.round(ks_pop, 4)))
        plt.figure(figsize=(3,3))
        # chart
        plt.plot(agg2['pop'], agg2.good_cum, label='cum_good',
                             color='blue', linestyle='-', linewidth=2)

        plt.plot(agg2['pop'], agg2.bad_cum, label='cum_bad',
                            color='red', linestyle='-', linewidth=2)

        plt.plot(agg2['pop'], agg2.ks, label='ks',
                       color='green', linestyle='-', linewidth=2)

        plt.title('KS=%s ' %np.round(ks_value, 4) +  
                    'at Pop=%s' %np.round(ks_pop, 4), fontsize=15)
        plt.legend()
        plt.show()    
#     pd.set_option('display.max_columns', 15)
    ksdf = agg2[['pop', 
                 'min_scr', 
                 'max_scr', 
                 'bads', 
                 'goods', 
                 'total', 
                 'odds', 
                 'bad_rate', 
                 'bad_cum_pct', 
                 'good_cum_pct',
                 'ks',
                 'max']]    
    return [ks_value, ks_pop, ksdf]

def plot_prc(y_true, y_score):
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    prc_auc = auc(recall,precision)
    plt.figure(figsize=(3,3))
    plt.plot(precision, recall, label='PRC_AUC = %0.2f'% prc_auc)
    plt.xlabel('Precision')
    plt.ylabel('Recall')
    plt.grid(True)
    ax = plt.gca()
    plt.xlim([-0.1,1.0])
    plt.ylim([-0.1,1.01])
    ax.set_aspect('equal')
    plt.legend(loc='lower right')
    plt.show() 
    return prc_auc

def plot_roc(y_true, y_score):
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    roc_auc = auc(fpr,tpr)
    # Plot ROC
    plt.figure(figsize=(3,3))
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b',label='ROC_AUC = %0.2f'% roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0,1],[0,1],'r--')
    plt.xlim([-0.1,1.0])
    plt.ylim([-0.1,1.01])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show() 
def evaluateClassfier_bc(model, 
                      true,
                      predicted_label,
                      predicted_prob,                     
                      enable_score = False,
                      ks_group_num = 20,
                      message = "Performance Evaluation "):
    
    Accuracy = accuracy_score(true, predicted_label)
    Recall = recall_score(true, predicted_label)
    Precision = precision_score(true, predicted_label)
    F1 = f1_score(true, predicted_label)
    AUC_PRC = average_precision_score(true, predicted_prob)
    AUC_ROC = roc_auc_score(true, predicted_prob)
    Log_loss = log_loss(true, predicted_prob)   

    cm = confusion_matrix(true, predicted_label)
    print(message)
    print("Log_loss:", round(Log_loss,4))  
    print("Accuracy :", round(Accuracy,4))
    print("Recall :", round(Recall,4))
    print("Precision :", round(Precision,4))
    print("F1:", round(F1,4))
    print("AUC_PRC:", round(AUC_PRC,4))
    print("AUC_ROC:", round(AUC_ROC,4))
  
    disp = ConfusionMatrixDisplay(confusion_matrix=cm) #display_labels=model.classes_)
    fig, ax = plt.subplots(figsize=(3,3))
    disp.plot(ax=ax)
    plt.show()
    


    print("KS Table : \n")
    #KS
    data = pd.DataFrame({'label': true,
                         # default using the second possiblity -> 1's possiblity 
                         'score': [Prob2Score(i,600,100)  for i in predicted_prob],
                         'prob':  predicted_prob,
                         })  
    if enable_score:
        KS = calc_ks(data, 'label', 'score', group_num=ks_group_num, reverse=False)
    else:
        KS = calc_ks(data, 'label', 'prob', group_num=ks_group_num, reverse=False)
    display(KS[2])

    
    print("ROC PLOT :")
    #ROC curve
    plot_roc(true, predicted_prob)
    
    #PRC curve
    print("PRC PLOT :")
    plot_prc(true, predicted_prob)
    print()

    return {'Log_loss': Log_loss,
            'Accuracy': Accuracy,
            'AUC_ROC': AUC_ROC,
            'AUC_PRC': AUC_PRC,
            'Recall': Recall,
            'Precision': Precision,
            'F1': F1,
            'KS_form': KS,
            'KS': KS[0],
            'KS_pop': KS[1]}

def model_evaluation_bc(X_train,
                       Y_train, 
                       X_val,
                       Y_val,
                       X_test,
                       Y_test,
                       model, 
                       prob_threshold=0.5,
                       ks_group_num = 20,
                       enable_score = False):
    train_predict = pd.DataFrame()
    eval_predict = pd.DataFrame()
    test_predict = pd.DataFrame()

    result = {'predict': [train_predict, eval_predict, test_predict]}
    if X_train is not None:
        train_predict['cutoff'] = prob_threshold
        predicted_train_y_p = model.predict_proba(X_train)[:, 1]
        predicted_train_y_pl = [1 if i >= prob_threshold else 0 for i in predicted_train_y_p]
        train_predict['prob_predict'] = predicted_train_y_p
        train_predict['prob_predict_pl'] = predicted_train_y_pl
        train_predict['label'] = Y_train.reset_index(drop=True)
        #plot prob distribution 
        sns.displot(kind='kde', data=train_predict, x='prob_predict', hue='label', common_norm=False,height=3, aspect=1)
        plt.figure(figsize=(3,3))
        train_eval_prob = evaluateClassfier_bc(
                                    model,
                                    true=Y_train,
                                    predicted_label=predicted_train_y_pl,
                                    predicted_prob=predicted_train_y_p,
                                    ks_group_num = ks_group_num,
                                    message = model.__class__.__name__+" Train Set (Threshold: {})".format(prob_threshold))
        result['train_eval_prob'] = train_eval_prob
        print("*"*30)

    if X_val is not None:
        eval_predict['cutoff'] = prob_threshold
        predicted_valid_y_p = model.predict_proba(X_val)[:,1]
        predicted_valid_y_pl = [1 if i >= prob_threshold else 0 for i in predicted_valid_y_p]
        eval_predict['prob_predict'] = predicted_valid_y_p
        eval_predict['prob_predict_pl'] = predicted_valid_y_pl
        eval_predict['label'] = Y_val.reset_index(drop=True)

        #plot prob distribution 
        sns.displot(kind='kde', data=eval_predict, x='prob_predict', hue='label', common_norm=False, height=3, aspect=1)
        plt.figure(figsize=(3,3))
        valid_eval_prob = evaluateClassfier_bc(
                                    model,
                                    true=Y_val,
                                    predicted_label=predicted_valid_y_pl,
                                    predicted_prob=predicted_valid_y_p,
                                    ks_group_num = ks_group_num,
                                    message = model.__class__.__name__+" Validation Set (Threshold: {})".format(prob_threshold))
        result['valid_eval_prob'] = valid_eval_prob
        print("*"*30)
    
    if X_test is not None:
        test_predict['cutoff'] = prob_threshold
        predicted_valid_y_p = model.predict_proba(X_test)[:,1]
        predicted_valid_y_pl = [1 if i >= prob_threshold else 0 for i in predicted_valid_y_p]
        test_predict['prob_predict'] = predicted_valid_y_p
        test_predict['prob_predict_pl'] = predicted_valid_y_pl
        test_predict['label'] = Y_test.reset_index(drop=True)

        #plot prob distribution 
        sns.displot(kind='kde', data=test_predict, x='prob_predict', hue='label', common_norm=False, height=3, aspect=1)
        plt.figure(figsize=(3,3))
        test_eval_prob = evaluateClassfier_bc(
                                    model,
                                    true=Y_test,
                                    predicted_label=predicted_valid_y_pl,
                                    predicted_prob=predicted_valid_y_p,
                                    ks_group_num = ks_group_num,
                                    message = model.__class__.__name__+" Test Set (Threshold: {})".format(prob_threshold))
        result['test_eval_prob'] = test_eval_prob
      
    summary = pd.DataFrame()
    if X_train is not None:
        s_t = pd.DataFrame({k : v for k, v in train_eval_prob.items() if not k.endswith('form')}, index=['Train'])
        summary = pd.concat([summary, s_t])
    if X_val is not None:
        s_v = pd.DataFrame({k : v for k, v in valid_eval_prob.items() if not k.endswith('form')}, index=['Valid'])
        summary = pd.concat([summary, s_v])
    if X_test is not None:
        s_te = pd.DataFrame({k : v for k, v in test_eval_prob.items() if not k.endswith('form')}, index=['Test'])
        summary = pd.concat([summary, s_te])
    display(summary)
    
    return result

def f1_score_eval(y_true, y_pred, weight):
    y_pred = list(map(lambda x: 1 if x>=0.5 else 0, y_pred))
    y_true = list(y_true)
    fs = f1_score(y_true, y_pred)
    return 'f1_score_eval', fs, True

def recall_eval(y_true, y_pred, weight):
    y_pred = list(map(lambda x: 1 if x>=0.5 else 0, y_pred))
    y_true = list(y_true)
    rs = recall_score(y_true, y_pred)
    return 'recall_eval', rs, True

def precision_eval(y_true, y_pred, weight):
    y_pred = list(map(lambda x: 1 if x>=0.5 else 0, y_pred))
    y_true = list(y_true)
    ps = precision_score(y_true, y_pred)
    return 'precision_eval', ps, True

def accuracy_eval(y_true, y_pred, weight):
    y_pred = list(map(lambda x: 1 if x>=0.5 else 0, y_pred))
    y_true = list(y_true)
    acs = accuracy_score(y_true, y_pred)
    return 'accuracy_eval', acs, True