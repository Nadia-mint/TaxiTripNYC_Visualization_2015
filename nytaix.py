from pyspark.mllib.clustering import KMeans, KMeansModel
from numpy import array
from pyspark.sql import SQLContext
from pyspark.sql.types import *

fileName = "yellow_tripdata_2015-01.csv"

sqlContext = SQLContext(sc)
taxiFile = sc.textFile(fileName)
header = taxiFile.first()

pickup_fields = [StructField(field_name, StringType(), True) for field_name in header.split(',') if "pickup" in field_name]
pickup_fields[0].dataType = TimestampType()
pickup_fields[1].dataType = FloatType()
pickup_fields[2].dataType = FloatType()
pickup_fields[0].name = "pickup_datetime"

dropoff_fields = [StructField(field_name, StringType(), True) for field_name in header.split(',') if "dropoff" in field_name]
dropoff_fields[0].dataType = TimestampType()
dropoff_fields[1].dataType = FloatType()
dropoff_fields[2].dataType = FloatType()
dropoff_fields[0].name = "dropoff_datetime"

pickup_schema = StructType(pickup_fields)
dropoff_schema = StructType(dropoff_fields)

taxiHeader = taxiFile.filter(lambda l: "ID"in l) # find the header row, which contains "VenderID"
taxiNoHeader = taxiFile.subtract(taxiHeader) # drop the first row

# we could not have used the header variable already calculated, 
# since header is just a local variable, 
# it cannot be subtracted from an RDD.

from datetime import *
from dateutil.parser import parse

# parse("1/1/2015  00:34:42") # a test

pickup_temp = taxiNoHeader.map(lambda k: k.split(",")).map(lambda p: (parse(p[1]), float(p[5]), float(p[6])))

dropoff_temp = taxiNoHeader.map(lambda k: k.split(",")).map(lambda p: (parse(p[2]), float(p[9]), float(p[10])))

# pickup_temp.top(2) # take a look

start_datetime = parse("1/1/2015 9:00:00")
end_datetime = parse("1/1/2015 12:00:00")

pickup_df = sqlContext.createDataFrame(pickup_temp, pickup_schema) # 12748986 records
# pickup_df.head(5) # look at the first 5 rows
# pickup_df.show()
# pickup_df.printSchema()
pickup_df = pickup_df.filter(pickup_df.pickup_datetime >= start_datetime)
pickup_df = pickup_df.filter(pickup_df.pickup_datetime <= end_datetime)
# there are missing longitude and latitude data filled with 0 and typos
pickup_df = pickup_df.filter(pickup_df.pickup_longitude > -75)
pickup_df = pickup_df.filter(pickup_df.pickup_longitude < -72)

dropoff_df = sqlContext.createDataFrame(dropoff_temp, dropoff_schema)
dropoff_df = dropoff_df.filter(dropoff_df.dropoff_datetime >= start_datetime)
dropoff_df = dropoff_df.filter(dropoff_df.dropoff_datetime <= end_datetime)
dropoff_df = dropoff_df.filter(dropoff_df.dropoff_longitude > -75)
dropoff_df = dropoff_df.filter(dropoff_df.dropoff_longitude < -72)

# try some queries
# taxi_df.groupBy("VendorID").count().show()
# taxi_df.filter(taxi_df.store_and_fwd_flag == '').count()

pickup_array = pickup_df.map(lambda x: array([float(x[1]), float(x[2])])).cache()
pickup_array.count()
pickup_model = KMeans.train(pickup_array, 10, maxIterations=10, runs=4, initializationMode="random")
pickup_centers = pickup_model.clusterCenters

'''
output:
[array([-73.98227551,  40.72726685]), 
 array([-73.94894565,  40.8091506 ]), 
 array([-73.97328617,  40.75436116]), 
 array([-73.98660181,  40.76263573]), 
 array([-73.87124998,  40.76696224]), 
 array([-73.97492762,  40.78705872]), 
 array([-73.95253233,  40.77328378]), 
 array([-73.78222692,  40.64736145]), 
 array([-73.99468622,  40.74292497]), 
 array([-74.0047403 ,  40.70852686])]
'''

dropoff_array = dropoff_df.map(lambda x: array([float(x[1]), float(x[2])]))
dropoff_array.cache()
dropoff_array.count() # 28699
dropoff_model = KMeans.train(dropoff_array, 10, maxIterations=10, runs=4, initializationMode="random")
dropoff_centers = dropoff_model.clusterCenters

'''
[array([-73.99213925,  40.75029304]), 
 array([-73.86811142,  40.76454869]), 
 array([-73.93448195,  40.82711368]), 
 array([-74.03472788,  40.70102565]), 
 array([-73.97541618,  40.75794618]), 
 array([-73.99530914,  40.72460149]), 
 array([-73.7803851 ,  40.65906982]), 
 array([-73.95075102,  40.77396579]), 
 array([-73.97614803,  40.7838726 ]), 
 array([-73.95674052,  40.68348334])]
'''


'''
clusters = []
cost = []
for k in range(2, 21):
	thisCluster = KMeans.train(tfVectors, k, maxIterations=10, runs=10, initializationMode="random")
	clusters.append(thisCluster)
	cost.append(thisCluster.computeCost(tfVectors))

plt.plot(range(2,21) , cost)
plt.ylabel('cost')
plt.xlabel('k')
plt.show()

model = clusters[5]
prediction = model.predict(tfIdfVectors)
labels = prediction.collect()

clusters.k
clusters.clusterCenters()
clusters.computeCost(tfVectors)
prediction = clusters.predict(tfIdfVectors)
prediction.take(10)
# clusters.save(sc, "output")
# sameModel = KMeansModel.load(sc, "output")
'''