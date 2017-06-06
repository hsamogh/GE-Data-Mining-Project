

from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier as RF
from pyspark.ml.feature import StringIndexer, VectorIndexer, VectorAssembler, SQLTransformer
from pyspark.ml.evaluation import MulticlassClassificationEvaluator, BinaryClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
import numpy as np
from pyspark.context import SparkContext
from pyspark import SQLContext;
from pyspark.sql.types import DoubleType, StringType
from pyspark.sql.functions import UserDefinedFunction
import pandas as pd;
from pyspark.ml.feature import OneHotEncoder
import matplotlib.pyplot as mp

# dictionaries to map Categorical values to numbers

# mapping for marital status , is_self_employed
yes_no_map = {"No": 1, "Yes": 0}

# mapping for gender
genderMap = {"M": 0, "F": 1, "m": 0, "f": 1}

# mapping for qualification
qualificationMap = {"Graduate": 1, "Not Graduate": 0}

#mapping for property area
propertyAreaMap = {"Rural": 0, "Urban":0}






sc = SparkContext()
sqlContext = SQLContext(sc)

data = sqlContext.read.load('processed_data.csv',format='com.databricks.spark.csv',header=True,inferSchema='true')
data.cache();


modified_property_area = UserDefinedFunction(lambda m: 'Rural' if m == 'Semiurban' else m,StringType())
#print(modified_property_area)
data = data.withColumn('property_area_modified',modified_property_area(data['property_area']))


marital_status_udf = UserDefinedFunction(lambda m: yes_no_map[m],DoubleType())
gender_udf = UserDefinedFunction(lambda g: genderMap[g], DoubleType())
self_employed_udf = UserDefinedFunction(lambda se: yes_no_map[se], DoubleType())
qualification_udf = UserDefinedFunction(lambda q : qualificationMap[q],DoubleType())
self_employed_encode = UserDefinedFunction(lambda at : propertyAreaMap[at], DoubleType() )

data = data.withColumn('marital_status_encoded',marital_status_udf(data['marital_status']))\
    .withColumn('gender_encoded',gender_udf(data['gender'])) \
    .withColumn('qualification_encoded', qualification_udf(data['qualification'])) \
    .withColumn('is_self_employed_modified', self_employed_udf(data['is_self_employed'])) \
    .withColumn('is_self_employed_encoded', self_employed_encode(data['is_self_employed_modified']))\
    .drop('loan_id') \
    .drop('marital_status')\
    .drop('gender') \
    .drop('is_self_employed')\
    .drop('is_self_employed')

data.printSchema();

(train_data, test_data) = data.randomSplit([0.8,0.2])


