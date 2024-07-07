import matplotlib.pyplot as plt
import pandas as pd

# Sample data representing the distribution of different LULC classes
data = {
    'LULC_Class': ['Water', 'Vegetation', 'Agriculture', 'Settlement', 'Barren'],
    'Area (sq km)': [50, 200, 150, 100, 25]
}

# Create a DataFrame
df = pd.DataFrame(data)

# Plotting the bar graph
plt.figure(figsize=(10, 6))
plt.bar(df['LULC_Class'], df['Area (sq km)'], color=['blue', 'green', 'yellow', 'red', 'brown'])

# Adding titles and labels
plt.title('Distribution of LULC Classes')
plt.xlabel('LULC Class')
plt.ylabel('Area (sq km)')

# Show the plot
plt.show()

import ee
import geemap

# Initialize the Earth Engine module.
ee.Initialize()

# Define a function to mask clouds based on the SCL band of Sentinel-2
def maskS2clouds(image):
    scl = image.select('SCL')
    cloudShadow = scl.eq(3)
    clouds = scl.eq(7)
    cirrus = scl.eq(10)
    mask = cloudShadow.Or(clouds).Or(cirrus)
    return image.updateMask(mask.Not())

# Load the Sentinel-2 image collection.
collection = ee.ImageCollection('COPERNICUS/S2_SR') \
    .filterDate('2022-01-01', '2022-12-31') \
    .filterBounds(ee.FeatureCollection('projects/ee-nasimaktar/assets/StudyArea')) \
    .map(maskS2clouds)

# Create a median composite.
composite = collection.median().clip(ee.FeatureCollection('projects/ee-nasimaktar/assets/StudyArea'))

# Define the visualization parameters.
vis_params = {
    'min': 0,
    'max': 3000,
    'bands': ['B4', 'B3', 'B2'],
    'gamma': 1.4
}

# Load the training data.
training_data = ee.FeatureCollection('projects/ee-nasimaktar/assets/TRD1')

# Filter out features that do not have the 'class' property.
training_data = training_data.filter(ee.Filter.notNull(['class']))

# Split the data into training (80%) and testing (20%) sets.
training_split = training_data.randomColumn()
training = training_split.filter(ee.Filter.lt('random', 0.8))
testing = training_split.filter(ee.Filter.gte('random', 0.8))

# Select bands for classification.
bands = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B11', 'B12']

# Sample the input imagery to get a FeatureCollection of training data.
training_samples = composite.select(bands).sampleRegions(
    collection=training,
    properties=['class'],
    scale=10
)

# Sample the input imagery to get a FeatureCollection of testing data.
testing_samples = composite.select(bands).sampleRegions(
    collection=testing,
    properties=['class'],
    scale=10,
    geometries=True
)

# Train the Random Forest classifier.
classifier = ee.Classifier.smileRandomForest(100).train(
    features=training_samples,
    classProperty='class',
    inputProperties=bands
)

# Classify the composite image.
classified = composite.select(bands).classify(classifier).clip(ee.FeatureCollection('projects/ee-nasimaktar/assets/StudyArea'))

# Define the visualization parameters for classification result.
class_vis_params = {
    'min': 0,
    'max': 4,  # Classes: Water (0), Vegetation (1), Agriculture (2), Settlement (3), Barren (4)
    'palette': ['blue', 'green', 'yellow', 'red', 'grey']
}

# Use geemap to display the original image, training data, and classification result.
Map = geemap.Map(center=[25.0, 87.8], zoom=10)
Map.addLayer(composite, vis_params, 'Sentinel-2 Composite')
Map.addLayer(training_data, {}, 'Training Data')
Map.addLayer(classified, class_vis_params, 'LULC Classification')
Map.addLayer(ee.FeatureCollection('projects/ee-nasimaktar/assets/StudyArea'), {}, 'Study Area')
Map

# Accuracy Assessment on Testing Data.
testAccuracy = classified.sampleRegions(
  collection=testing_samples,
  properties=['class'],
  scale=10
)

# Compute the confusion matrix.
confusionMatrix = testAccuracy.errorMatrix('class', 'classification')

# Print the confusion matrix.
print('Confusion Matrix:', confusionMatrix.getInfo())
print('Overall Accuracy:', confusionMatrix.accuracy().getInfo())
print('Kappa Coefficient:', confusionMatrix.kappa().getInfo())
