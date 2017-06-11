import pyspark.mllib.regression
import pyspark.mllib.tree
from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier as RF
from pyspark.ml.feature import StringIndexer, VectorIndexer, VectorAssembler, SQLTransformer
from pyspark.ml.evaluation import MulticlassClassificationEvaluator, BinaryClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
import numpy as np
from  pyspark.mllib.evaluation import BinaryClassificationMetrics as meteric
from pyspark.context import SparkContext
from pyspark import SQLContext;
from pyspark.sql.types import DoubleType, StringType
from pyspark.sql.functions import UserDefinedFunction
import pandas as pd;
from pyspark.ml.feature import OneHotEncoder
import matplotlib.pyplot as mp

# dictionaries to map Categorical values to numbers

# mapping for marital status , is_self_employed
yes_no_map = {"No": 1.0, "Yes": 0.0}

# mapping for gender
genderMap = {"M": 0.0, "F": 1.0, "m": 0.0, "f": 1.0}

# mapping for qualification
qualificationMap = {"Graduate": 1.0, "Not Graduate": 0.0}

#mapping for property area
propertyAreaMap = {"Rural": 0.0, "Urban":1.0}






sc = SparkContext()
sqlContext = SQLContext(sc)

data = sqlContext.read.load('processed_data.csv',format='com.databricks.spark.csv',header=True,inferSchema='true')
data.cache();


modified_property_area = UserDefinedFunction(lambda m: 'Rural' if m == 'Semiurban' else m,StringType())
#print(modified_property_area)
data = data.withColumn('property_area_modified',modified_property_area(data['property_area']))



marital_status_udf = UserDefinedFunction(lambda m: yes_no_map[m],DoubleType())
gender_udf = UserDefinedFunction(lambda g: genderMap.get(g,0), DoubleType())
self_employed_udf = UserDefinedFunction(lambda se: yes_no_map.get(se,0), DoubleType())
qualification_udf = UserDefinedFunction(lambda q: qualificationMap.get(q,0), DoubleType())
self_employed_encode = UserDefinedFunction(lambda at: propertyAreaMap.get(at,0), DoubleType() )

data = data.withColumn('marital_status_encoded',marital_status_udf(data['marital_status']))\
     .withColumn('gender_encoded',gender_udf(data['gender'])) \
     .withColumn('qualification_encoded', qualification_udf(data['qualification'])) \
     .withColumn('is_self_employed_encoded', self_employed_udf(data['is_self_employed'])) \
     .withColumn('property_area_encoded', self_employed_encode(data['property_area_modified']))\
     .drop('loan_id') \
     .drop('marital_status')\
    .drop('gender') \
     .drop('is_self_employed')\
     .drop('property_area') \
     .drop('property_area_modified')\
     .drop('qualification')

data = data.drop('loan_id')

feature_columns = data.columns
feature_columns.remove('status')
data.toPandas().to_csv("new_data.csv")
assembler_features = VectorAssembler(inputCols=feature_columns, outputCol='features')
prediction_column = StringIndexer(inputCol='status', outputCol='label')
tmp = [assembler_features , prediction_column]

pipeline = Pipeline(stages=tmp)
sum=0.0

for i in range(10):
    all_data = pipeline.fit(data).transform(data)
    (training_data, test_data) = all_data.randomSplit([0.8,0.2])
    test_data.drop('status')

    rf = RF(labelCol='label', featuresCol='features',numTrees=200)


    fit = rf.fit(training_data)
    transformed = fit.transform(test_data)

    results = transformed.select(['probability','label'])

    results_collect = results.collect()
    result_list = [(float(i[0][0]), 1.0 - float(i[1])) for i in results_collect]
    scores = sc.parallelize(result_list)

    meterics = meteric(scores)
    sum=sum+meterics.areaUnderROC

print(sum/10)


