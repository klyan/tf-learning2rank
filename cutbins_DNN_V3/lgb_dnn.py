# coding=utf-8
import sys
reload(sys)
sys.setdefaultencoding('utf8')
import tensorflow as tf
import pandas as pd
import numpy as np
import json
import argparse
import os
import datetime
import math
import xgboost as xgb
import lightgbm as lgb
import time
import random
from sklearn.metrics import roc_auc_score
from pandas import Series
import gc
import datetime
import bisect
from scipy.sparse import coo_matrix
from sklearn import metrics
import shutil
import pickle
import math

def sigmoid(inX):
    return 1.0 / (1 + math.exp(-inX))

def get_threshold(tree_json):
    threshold = []
    if "right_child" in tree_json.keys():
        threshold.extend(get_threshold(tree_json['right_child']))
    if "left_child" in tree_json.keys():
        threshold.extend(get_threshold(tree_json['left_child']))
    if "threshold" in tree_json.keys():
        threshold.extend([tree_json['threshold']])
    return threshold


def get_bucketized_cols_by_tree(train_data, train_label, numeric_columns):
    ligbmodel = lgb.LGBMClassifier(num_leaves = 12, max_depth=4, n_estimators= 1)
    columns = []
    bucketized_cols = []
    column_bins = {}
    for idx, col in enumerate(numeric_columns):
        if col not in train_data.columns:
            print(col)
            continue
        ligbmodel.fit(pd.DataFrame(train_data[col].fillna(0)), train_label)
        split = sorted(get_threshold(ligbmodel.booster_.dump_model()["tree_info"][0]["tree_structure"]))
        numeric_feature_column = tf.feature_column.numeric_column(col)
        bucketized_cols.extend([tf.feature_column.bucketized_column(source_column = numeric_feature_column, boundaries = split)])
        column_bins[col] = split
    with open('/data/kai.zhang/dnn/onlineModel/feature_quantile.txt', 'w') as fw:
        json.dump(column_bins, fw)
    return bucketized_cols


def getNumCateFeatures(raw):
    col_nunique = raw.nunique()
    category_fea = list(set(list(col_nunique[col_nunique <= 15].index)) - set(['label','real_order','sample_weight']))
    numeric_fea = list(set(list(raw.columns)) - set(category_fea) - set(['label','real_order','sample_weight']))
    return category_fea, numeric_fea


def getModelFeatures(category_fea, numeric_fea):
    model_feature_columns = []
    for key in category_fea:
        model_feature_columns.append(tf.feature_column.numeric_column(key=key))
    bucketized_cols = get_bucketized_cols_by_tree(data_train, data_train_click, numeric_fea)
    model_feature_columns.extend(bucketized_cols)
    return model_feature_columns


def reorderDataColumns(category_feas, numeric_feas):
    DATA_COLUMNS = []
    for fea_cate in category_feas:
        s = 'fea_cate_' + fea_cate
        DATA_COLUMNS.append(s)
    for fea_cate in numeric_feas:
        s = 'fea_real_' + fea_cate
        DATA_COLUMNS.append(s)
    return DATA_COLUMNS

def rename_col_for_input_order():
    cnt = 0
    col_mapping = {}
    for idx, col in enumerate(DATA_COLUMNS):
        if col.startswith('fea_'):
            sp = col.split('_')
            new_col = '_'.join(sp[:-1])+'{0:03}_'.format(cnt) + sp[-1]
            col_mapping[col] = new_col
            cnt += 1
        else:
            col_mapping[col] = col
    return col_mapping



my_model_dir = '/data/kai.zhang/dnn/model'

shutil.rmtree("/data/kai.zhang/dnn/model", ignore_errors=True)
os.mkdir("/data/kai.zhang/dnn/model")


pd.set_option('display.max_columns', None)
os.chdir("/data/kai.zhang/dnn")


#/data/kai.zhang/dnn/model/feature_quantile.txt

#raw = pd.read_csv('/data/kai.zhang/dnn/dnn_sampe.csv',sep=',')

raw = pd.read_csv('/data/kai.zhang/dnn/dnn0514.csv',sep='\t', na_values=['\N','NULL','null'], error_bad_lines=False)

raw.columns = ["queryid", "label", "real_order", "sample_weight", "isfoodlevel1", "ishuishop", "distance", "ocr", "seg", "lastview", "yestdayctr", "realclick", "loccvr", "todayctr", "discount", "isdealshop", "logceilpictotal", "ctr", "densitythirty", "pricepref", "catprefwithgeo", "isnewuser", "ispermanentcity", "guessstar", "spl", "istopshop", "topshop", "istakeawayshop", "allcategoryctr", "distancelarge3km", "distanceless3km", "distanceless2km", "distanceless1km", "distanceless500m", "distanceless200m", "fclick", "fclickpv", "fclickpriceless50", "fclickpriceless100", "fclickpriceless200", "fclickpriceless300", "fclickpricelarge300", "crossclickprice", "crossdistance", "crossorderprice", "repeatclickratio", "shopclickusers", "shopclickcount", "click5ratio", "click1ratio", "poirepeatratio", "ctravgratio", "staravgratio", "popscoreavgratio", "scoreavgratio", "branchcntavgratio", "timeseekshophoursctr", "isfavorclickcate2", "cate2clickrecency", "cate2clickfrequency", "cate2orderrecency", "cate2orderfrequency", "cate2ordermonetary", "dividedistancectr", "productdistancectr", "similarshops"]


deletes = ["shop_id","locatecityid","queryid"]
deletes= list(set(raw.columns).intersection([x.lower() for x in deletes]))
raw.drop(deletes, axis=1, inplace=True)


raw[['crossclickprice', 'crossdistance', 'crossorderprice']] = raw[['crossclickprice', 'crossdistance', 'crossorderprice']].fillna(value=-1)
raw[['crossclickprice', 'crossdistance', 'crossorderprice']] = raw[['crossclickprice', 'crossdistance', 'crossorderprice']].fillna(value=-1)
raw.fillna(value=0, inplace=True)


selected = ["guessstar", "ispermanentcity", "isdealshop", "isnewuser", "isfavorclickcate2", "ishuishop", "istopshop", "istakeawayshop", "isfoodlevel1"]

raw[selected].dtypes

col_nunique = raw[selected].nunique()



for col in raw.columns:
    if col in ['label', 'real_order']:
        continue
    raw[col] = raw[col].astype('float64')



category_feas, numeric_feas = getNumCateFeatures(raw)

DATA_COLUMNS = reorderDataColumns(category_feas, numeric_feas)

COLUMN_ALIAS_MAPPING = rename_col_for_input_order()



remove_col = ["shopclicknum","viewedlastshoppos","sample_weight"]
remove_col = [x.lower() for x in remove_col]
removed= list(set(raw.columns).intersection(remove_col))

random.seed(100)
test_index = random.sample(raw.index, int(len(raw.index)*0.1))

#test data
data_test = raw.loc[test_index].drop(removed, axis=1, inplace=False).drop(["label","real_order"], axis=1, inplace=False)
data_test_click = raw.loc[test_index].label
data_test_order = raw.loc[test_index].real_order

#all data
data = raw.drop(test_index)
filter_ori= (data["shopclicknum"] > 0) | ((data["shopclicknum"] <= 0) & (data["viewedlastshoppos"]>=10))
data = data[filter_ori]

random.seed(100)
train_index = random.sample(data.index, int(len(data.index)*0.9))

del raw
import gc
gc.collect()

#train data
data_train_click = data.loc[train_index].label
data_train_order = data.loc[train_index].real_order
data_train = data.loc[train_index].drop(["label","real_order"], axis=1, inplace=False).drop(removed, axis=1, inplace=False)

#eval data
data_dev_click = data.drop(train_index).label
data_dev_order = data.drop(train_index).real_order
data_dev = data.drop(train_index).drop(["label","real_order"], axis=1, inplace=False).drop(removed, axis=1, inplace=False)



#data_train.drop("order_weight",axis=1, inplace=True)



model_feature_columns = []
for key in category_feas:
    #raw = raw.rename(columns={key: 'fea_real_' + key})
    data_train.rename(columns={key: COLUMN_ALIAS_MAPPING['fea_cate_' + key]} ,inplace=True)
    data_dev.rename(columns={key: COLUMN_ALIAS_MAPPING['fea_cate_' + key]} , inplace=True)
    data_test.rename(columns={key: COLUMN_ALIAS_MAPPING['fea_cate_' + key]} ,inplace=True)
    model_feature_columns.append(tf.feature_column.numeric_column(key=COLUMN_ALIAS_MAPPING['fea_cate_' + key]))


numeric_fea1 = []
for key in numeric_feas:
    #raw = raw.rename(columns={key: 'fea_real_' + key})
    data_train.rename(columns={key: COLUMN_ALIAS_MAPPING['fea_real_' + key]}, inplace=True)
    data_dev.rename(columns={key: COLUMN_ALIAS_MAPPING['fea_real_' + key]}, inplace=True)
    data_test.rename(columns={key: COLUMN_ALIAS_MAPPING['fea_real_' + key]}, inplace=True)
    numeric_fea1.append(COLUMN_ALIAS_MAPPING['fea_real_' + key])

bucketized_cols = get_bucketized_cols_by_tree(data_train, data_train_click, numeric_fea1)
model_feature_columns.extend(bucketized_cols)




data_train.rename(columns=COLUMN_ALIAS_MAPPING, inplace=True)
data_dev.rename(columns=COLUMN_ALIAS_MAPPING, inplace=True)
data_test.rename(columns=COLUMN_ALIAS_MAPPING, inplace=True)

#model_feature_columns = getModelFeatures(COLUMN_ALIAS_MAPPING)

print("columns length:",len(model_feature_columns), data_train.shape[1])

run_config = tf.estimator.RunConfig( tf_random_seed = 666
                                , save_summary_steps = 200
                                , session_config=tf.ConfigProto(device_count={'GPU': 1})
)


is_test = True

def my_model(features, labels, mode, params):
    if 'order_weight' in features.keys():
        sample_weight = features.pop('order_weight')
    else:
        sample_weight = 1.0
    nets = {}
    net = tf.feature_column.input_layer(features, params['feature_columns'])
    for idx, units in enumerate(params['hidden_units']):
        nets['net_'+str(idx)] = net
        net = tf.layers.dense(net, units=units, activation=params["activation"]) #tf.nn.relu
        #net = tf.layers.dropout(net, rate=0.2)
        #tf.contrib.layers.l2_regularizer 
    # Compute logits (1 per class).
    logits = tf.layers.dense(net, params['n_classes'], activation=None)

    # Compute predictions.
    export_outputs={'my_export_outputs':  tf.estimator.export.PredictOutput({'prediction': logits})}

    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            'probabilities': tf.nn.softmax(logits),
            'probabilities1': tf.sigmoid(logits),
            'logits': logits
        }
        if is_test:
            for k, v in nets.items():
                predictions[k] = v
        return tf.estimator.EstimatorSpec(mode, predictions=predictions, export_outputs=export_outputs)

    # Compute loss.
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits, weights = sample_weight)

    #loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits logits)
    metrics_auc = tf.metrics.auc(labels=labels, predictions=tf.nn.softmax(logits)[:,1])
    metrics = {'auc': metrics_auc}          #tf.summary.scalar('auc', metrics_auc)
    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=metrics, export_outputs=export_outputs)

    # Create training op.
    assert mode == tf.estimator.ModeKeys.TRAIN
    optimizer = tf.train.AdagradOptimizer(learning_rate=0.1)
    train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step()) 
    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op, export_outputs=export_outputs)


def my_model(features, labels, mode, params):
    if 'order_weight' in features.keys():
        sample_weight = features.pop('order_weight')
    else:
        sample_weight = 1.0
    nets = {}
    net = tf.feature_column.input_layer(features, params['feature_columns'])
    for idx, units in enumerate(params['hidden_units']):
        nets['net_'+str(idx)] = net
        net = tf.layers.dense(net, units=units, activation=params["activation"])  # tf.nn.relu
        # net = tf.layers.dropout(net, rate=0.2)
        # tf.contrib.layers.l2_regularizer
    logits = tf.layers.dense(net, params['n_classes'], activation=None)

    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {'probabilities': tf.nn.softmax(logits),
                       'logits': logits}
        if is_test:
            for k, v in nets.items():
                predictions[k] = v
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)

    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)  # , weights=sample_weight)
    accuracy = tf.reduce_mean(tf.to_float(tf.equal(tf.argmax(logits, axis=1), labels)))
    if mode == tf.estimator.ModeKeys.EVAL:
        auc = tf.metrics.auc(labels=labels, predictions=tf.nn.softmax(logits)[:, 1])
        return tf.estimator.EstimatorSpec(mode, loss=loss,
                                          eval_metric_ops={'auc': auc})

    assert mode == tf.estimator.ModeKeys.TRAIN
    for var in tf.trainable_variables():
        tf.summary.histogram(var.name, var)
    tf.summary.scalar('loss', loss)
    tf.summary.scalar('accuracy', accuracy)
    logging_hook = tf.train.LoggingTensorHook({"loss": loss,
                                               "accuracy": accuracy}, every_n_iter=100)
    summary_hook = tf.train.SummarySaverHook(summary_op=tf.summary.merge_all(), save_steps=100)
    optimizer = tf.train.AdagradOptimizer(learning_rate=0.01)
    train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op, training_hooks=[logging_hook, summary_hook])


shutil.rmtree("/data/kai.zhang/dnn/model", ignore_errors=True)
os.mkdir("/data/kai.zhang/dnn/model")

tf.set_random_seed(1234)
classifier = tf.estimator.Estimator(
        model_fn = my_model,
        model_dir = my_model_dir,
        params={
            'feature_columns': model_feature_columns,
            'hidden_units': [256, 256, 256],
            'n_classes': 2,
            'activation':tf.nn.relu, #tf.nn.leaky_relu
            'sample_weight':1
        },
        config = run_config
)

selected = ["fea_cate000_guessstar", "fea_cate001_ispermanentcity", "fea_cate002_isdealshop", "fea_cate003_isnewuser", "fea_cate004_isfavorclickcate2", "fea_cate005_ishuishop", "fea_cate006_istopshop", "fea_cate007_istakeawayshop", "fea_cate008_isfoodlevel1"]

###添加权重
for wei in xrange(5,20,2):
    print "current weight:: ", wei
    wei = 30
    data_sample_weight = data.loc[train_index].real_order * wei + 1
    data_train["order_weight"] = data_sample_weight
    data_train["order_weight"] = data_train["order_weight"].astype('float64') 
    data_train_tmp = data_train.drop("order_weight", axis=1)



    train_input_fn = tf.estimator.inputs.pandas_input_fn(x=data_train, y=data_train_click, batch_size = 1024, num_epochs=1, shuffle=True)

    #shutil.rmtree("/data/kai.zhang/dnn/model/", ignore_errors=True)
    classifier.train(input_fn=train_input_fn)

    with open(my_model_dir + '/deep.model', 'w') as fw:
        pickle.dump(classifier, fw)

    #train
    train_result = classifier.evaluate(input_fn=tf.estimator.inputs.pandas_input_fn(x=data_train_tmp, y=data_train_click, batch_size = 1024, num_epochs=1, shuffle=True))
    print "train click auc:", train_result

    # Eval
    eval_input_fn = tf.estimator.inputs.pandas_input_fn(x=data_dev, y=data_dev_click, batch_size = 1024, num_epochs=1, shuffle=False)
    eval_result = classifier.evaluate(input_fn=eval_input_fn)
    print("eval click auc:", eval_result) #0.71657276


    #test
    test_input_fn = tf.estimator.inputs.pandas_input_fn(x=data_test, y=data_test_click, batch_size = 1024, num_epochs=1, shuffle=False)
    print "Test click auc: ", classifier1.evaluate(input_fn=test_input_fn)  #0.71754426



    ##order auc 
    train_order_result = classifier.evaluate(input_fn=tf.estimator.inputs.pandas_input_fn(x=data_train_tmp, y=data_train_order, batch_size = 1024, num_epochs=1, shuffle=False))
    print "train order auc: ", train_order_result

    # Eval
    order_eval_result = classifier.evaluate(input_fn=tf.estimator.inputs.pandas_input_fn(x=data_dev, y=data_dev_order, batch_size = 1024, num_epochs=1, shuffle=False))
    print "eval order auc: ", order_eval_result

    #test
    order_test_result = classifier.evaluate(input_fn=tf.estimator.inputs.pandas_input_fn(x=data_test, y=data_test_order, batch_size = 1024, num_epochs=1, shuffle=False))
    print "Test order auc: ", order_test_result




###以下是草稿

feature_spec = tf.feature_column.make_parse_example_spec(model_feature_columns)
serving_input_fn = tf.estimator.export.build_parsing_serving_input_receiver_fn(feature_spec)
export_dir = classifier.export_savedmodel("/data/kai.zhang/dnn/saveModel", serving_input_receiver_fn = serving_input_fn, as_text = True)


tf.train.latest_checkpoint("/data/kai.zhang/dnn/model")


reader = pywrap_tensorflow.NewCheckpointReader(file_name)
if all_tensors:
  var_to_shape_map = reader.get_variable_to_shape_map()
  for key in sorted(var_to_shape_map):
    print("tensor_name: ", key)
    print(reader.get_tensor(key))
        
#https://github.com/Fematich/mlengine-boilerplate/blob/master/trainer/model.py




test_result = classifier.predict(input_fn=test_input_fn)
train_stop = datetime.datetime.now()

for idx, pred in enumerate(test_result):
    if idx < 10: 
        print pred["probabilities"], pred["logits"]



###sklearn
y_predicted = np.array(list(p['probabilities'][1] for p in test_result))
y_predicted = y_predicted.reshape(np.array(data_test_click).shape)
y = np.array(data_test_click)
metrics.roc_auc_score(y, y_predicted)#验证集上的auc值


###验证线上打分与本地是否一致
from pandas import Series
import math
def sigmoid(x):
  return 1 / (1 + math.exp(-x))




new1 = pd.DataFrame(dict((k,0.0) for k in list(data_test.columns)),index=[0])  
check_input_fn = tf.estimator.inputs.pandas_input_fn(x=new1, y=Series([1]), batch_size = 1024, num_epochs=1, shuffle=False)
check_result = classifier.predict(input_fn=check_input_fn)

for idx, pred in enumerate(check_result):
    print(pred)
    print(pred["probabilities"], pred["logits"], sigmoid(pred["logits"][0]), sigmoid(pred["logits"][1]))




def predict_detail(test_file, model_dir, need_valid=True):
    classifier = pickle.load(open(my_model_dir + '/deep.model', 'rb'))
    test_input_fn = tf.estimator.inputs.pandas_input_fn(x=data_test, y=data_test_click, batch_size=1024, num_epochs=1, shuffle=False)
    test_result = classifier.predict(input_fn=test_input_fn)
    need_valid = True
    is_test = True
    for idx, pred in enumerate(test_result):
        if need_valid and idx < 2:
            print(pred, 1 / (1 + np.exp(-1 * pred['logits'][1])))






###lgb
dtrain = lgb.Dataset(data_train, label=data_train_click, weight=data_sample_weight, free_raw_data=False)
dtest = lgb.Dataset(data_test, label=data_test_click, free_raw_data=False)

gc.collect()

param = {'num_leaves':8,'num_boost_round':300, 'objective':'binary','metric':'auc',"learning_rate" : 0.3, "boosting":"gbdt"}

bst = lgb.train(param, dtrain, valid_sets=[dtrain, dtest],  verbose_eval=50)
pred_train = bst.predict(data_train)
pred_test = bst.predict(data_test)



gbdt = xgb.XGBClassifier(n_estimators=300, max_depth=3)
gbdt.fit(data_train, data_train_click, sample_weight= data_sample_weight, eval_metric='auc')

pred_train = gbdt.predict_proba(data_train)
pred_dev = gbdt.predict_proba(data_dev)
pred_test = gbdt.predict_proba(data_test)

print('train auc: %g' % roc_auc_score(data_train_click, pred_train[:,1]))
print('valid auc: %g' % roc_auc_score(data_dev_click, pred_dev[:,1]))
print('valid auc: %g' % roc_auc_score(data_test_click, pred_test[:,1]))

print('train auc: %g' % roc_auc_score(data_train_order, pred_train[:,1]))
print('valid auc: %g' % roc_auc_score(data_dev_order, pred_dev[:,1]))
print('valid auc: %g' % roc_auc_score(data_test_order, pred_test[:,1]))





curl 'http://localhost:6051/search/mainshop?query=term(categoryids,10),term(shoptype,10),geo(poi,121.416145:31.217739,1600)&notquery=term(power,1),term(weddingcloneshop,1)&sort=desc(daocannew)&stat=count(discountstatus),count(shoppower),count(segmentscene)&limit=0,5&fl=avgprice,score1,score2,score3,hasgroup,dist(poi,121.416145:31.217739),shopid&info=mappingmodel:FoodCat1DNNSeq1Test,querytypeandchannelid:1_10,dpid:-5559610562723745782,userid:1775625,needcpmad:true,AdDownGradeRule:true,contentadnumber:100,needrulefeaturererank:true,toptype:2,recallresultnumber:350,shouldblockad:false,modelreranksize:300,admergeenabled:true,clientip:10.76.175.231,elevateids:,userip:203.76.219.100,bu_constrained:1,maxcontinuousadnumber:3,app:PointShopSearch,wifi:,locatecityid:1,isfoodcatelocal:false,bizname:mainshop,forceextrudecontentad:true,generalsearchmall:exp2,isfixedpositionad:true,cityid:1,rewrite:GeoDynamicRangeProcessor%3A%5B5000%5D%3BLocationURecDistProcessor%3A%5B5000%2C1600%5D%3Bunrelated-pre%3A%5B%5Bcategoryids%2C+regionids%5D%5D%3B,geoFieldName:poi,isoverseastraffic:false,ModelScoreNormKey:10,mobiledeviceid:29c7b7f8835a49eb73b0b0e727ee8da298e1dbfa,pagecity:1,ab_itemchain:localdistscore%405.0%7CshopStatus%40100%7CbaseScore%401.0%7CshopBusinessNew%400.4%7Csegscore%400.2,userlng:121.4161453926712,needexpandrange:true,sorttype:1,keys:,useragent:MApi+1.2+%28dpscope+9.9.11+appstore%3B+iPhone+11.2.6+iPhone10%2C3%3B+a0d0%29,poi:121.416145%3A31.217739+5000,adenabled:true,bizClientIp:10.69.61.173,platform:MAPI,userlat:31.21773943604781,livecity:1,categoryids:10,mobileplatform:2,needrecdist:true,querymaincategoryid:10,hotarea:true,pageonemaxadnumber:8,foodexp_pagenum:a,needrankinfo:true,unrelatedguidefields:categoryids%3Bregionids,clientversion:9.9.11,pagemodule:mainshoplist,dynamicrange:true,needpositionpromotion:false,recalladnumber:200,originsort:DESC_dpscore,deleteemptyfield:true,nearby_poi:121.416145%3A31.217739%2C5000,modelscoredetail:true,rulefeaturedetail:true,needcheckrulelog:true' | python -m json.tool 



InitFeatureValues=[0,0.0,0,0,0,0,1,0,1.0,1.25,10.746268656716419,3.0,0.01715,0.002,2128.31,9.156939595249067,0.113,0.0,1.376947,1.5078,0.0,0.07908,9.149574,-1.0,0.14672,6.0,-1.0,0.015292727272727347,9038.0,0.0,-1.0,2.24E-4,0.022411089180474734,13.428223947143554,0.815114,1.394512,3.0,0.0,0.411292,19.0,1,219.8684262204234,0.0,0.09798613995401421,6695.0,0.33138721247631614,0.005675,8760.0,0.0,232.56362915039062,0.0996679946249559,8.0,0.05774,6.0,9.0,0.5,17.0,5.0,0.740679,5.0,0.03406327445962531,2.0]

InitFeatures = ['fea_cate000_guessstar', 'fea_cate001_ispermanentcity', 'fea_cate002_isdealshop', 'fea_cate003_isnewuser', 'fea_cate004_isfavorclickcate2', 'fea_cate005_ishuishop', 'fea_cate006_istopshop', 'fea_cate007_istakeawayshop', 'fea_cate008_isfoodlevel1', 'fea_real009_crossclickprice', 'fea_real010_crossorderprice', 'fea_real011_fclickpricelarge300', 'fea_real012_seg', 'fea_real013_densitythirty', 'fea_real014_cate2orderrecency', 'fea_real015_logceilpictotal', 'fea_real016_spl', 'fea_real017_realclick', 'fea_real018_ctravgratio', 'fea_real019_scoreavgratio', 'fea_real020_loccvr', 'fea_real021_timeseekshophoursctr', 'fea_real022_popscoreavgratio', 'fea_real023_distanceless3km', 'fea_real024_catprefwithgeo', 'fea_real025_fclickpriceless100', 'fea_real026_distanceless2km', 'fea_real027_pricepref', 'fea_real028_shopclickcount', 'fea_real029_cate2clickfrequency', 'fea_real030_distanceless200m', 'fea_real031_ocr', 'fea_real032_todayctr', 'fea_real033_productdistancectr', 'fea_real034_click1ratio', 'fea_real035_staravgratio', 'fea_real036_fclickpriceless50', 'fea_real037_topshop', 'fea_real038_branchcntavgratio', 'fea_real039_fclick', 'fea_real040_discount', 'fea_real041_dividedistancectr', 'fea_real042_lastview', 'fea_real043_allcategoryctr', 'fea_real044_shopclickusers', 'fea_real045_crossdistance', 'fea_real046_click5ratio', 'fea_real047_cate2clickrecency', 'fea_real048_cate2ordermonetary', 'fea_real049_distance', 'fea_real050_similarshops', 'fea_real051_distanceless1km', 'fea_real052_ctr', 'fea_real053_distancelarge3km', 'fea_real054_cate2orderfrequency', 'fea_real055_poirepeatratio', 'fea_real056_fclickpv', 'fea_real057_fclickpriceless200', 'fea_real058_repeatclickratio', 'fea_real059_distanceless500m', 'fea_real060_yestdayctr', 'fea_real061_fclickpriceless300']


#InitFeatureValues, InitFeatures

z1 = json.dumps(dict(zip(InitFeatures, InitFeatureValues)))



new1 = pd.DataFrame({"fea_real037_topshop": 0.0, "fea_real053_distancelarge3km": 6.0, "fea_real012_seg": 0.01715, "fea_real030_distanceless200m": -1.0, "fea_real036_fclickpriceless50": 3.0, "fea_real051_distanceless1km": 8.0, "fea_real018_ctravgratio": 1.376947, "fea_real055_poirepeatratio": 0.5, "fea_real013_densitythirty": 0.002, "fea_real025_fclickpriceless100": 6.0, "fea_real028_shopclickcount": 9038.0, "fea_real060_yestdayctr": 0.03406327445962531, "fea_cate006_istopshop": 1, "fea_real048_cate2ordermonetary": 0.0, "fea_real056_fclickpv": 17.0, "fea_real024_catprefwithgeo": 0.14672, "fea_real049_distance": 232.56362915039062, "fea_real016_spl": 0.113, "fea_cate003_isnewuser": 0, "fea_real050_similarshops": 0.0996679946249559, "fea_real027_pricepref": 0.015292727272727347, "fea_real009_crossclickprice": 1.25, "fea_real029_cate2clickfrequency": 0.0, "fea_real011_fclickpricelarge300": 3.0, "fea_real047_cate2clickrecency": 8760.0, "fea_real031_ocr": 0.000224, "fea_real014_cate2orderrecency": 2128.31, "fea_real044_shopclickusers": 6695.0, "fea_real033_productdistancectr": 13.428223947143554, "fea_cate005_ishuishop": 0, "fea_real046_click5ratio": 0.005675, "fea_real032_todayctr": 0.022411089180474734, "fea_real052_ctr": 0.05774, "fea_real039_fclick": 19.0, "fea_real010_crossorderprice": 10.746268656716419, "fea_cate000_guessstar": 0, "fea_cate008_isfoodlevel1": 1.0, "fea_real042_lastview": 0.0, "fea_real021_timeseekshophoursctr": 0.07908, "fea_real035_staravgratio": 1.394512, "fea_real043_allcategoryctr": 0.09798613995401421, "fea_real061_fclickpriceless300": 2.0, "fea_cate002_isdealshop": 0, "fea_real034_click1ratio": 0.815114, "fea_real020_loccvr": 0.0, "fea_real054_cate2orderfrequency": 9.0, "fea_real017_realclick": 0.0, "fea_cate007_istakeawayshop": 0, "fea_real045_crossdistance": 0.33138721247631614, "fea_real058_repeatclickratio": 0.740679, "fea_real057_fclickpriceless200": 5.0, "fea_real019_scoreavgratio": 1.5078, "fea_cate004_isfavorclickcate2": 0, "fea_cate001_ispermanentcity": 0.0, "fea_real038_branchcntavgratio": 0.411292, "fea_real015_logceilpictotal": 9.156939595249067, "fea_real023_distanceless3km": -1.0, "fea_real022_popscoreavgratio": 9.149574, "fea_real059_distanceless500m": 5.0, "fea_real026_distanceless2km": -1.0, "fea_real040_discount": 1, "fea_real041_dividedistancectr": 219.8684262204234},index=[0])




check_input_fn = tf.estimator.inputs.pandas_input_fn(x=new1, y=Series([1]), batch_size = 1024, num_epochs=1, shuffle=False)

check_result = classifier.predict(input_fn=check_input_fn)

for idx, pred in enumerate(check_result):
    print(pred, sigmoid(pred['logits'][1]))




