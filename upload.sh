# Configure AWS
aws configure

# upload
aws s3 cp data/ s3://tinyml/data/ --recursive --exclude "*" --include "*.tfrecords"